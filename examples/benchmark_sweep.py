"""Sweep the FPS benchmark across environment counts and shard counts, and plot the curve.

The single-point benchmark answers "how fast is this configuration". A sweep answers the question a
scaling curve is for: how throughput moves as the environment count grows, and how many Showdown
servers it takes to keep up. Each cell of the grid is one environment count paired with one shard
count, run as its own process so a configuration that runs out of memory, ports, or file
descriptors records a failure and the sweep carries on to the next.

    python examples/benchmark_sweep.py --output-dir runs/sweep

By default it covers 512 to 32768 environments against 16, 32, 64 and 128 servers, which is 24
runs and, at the larger sizes, hours of wall time. The sweep is resumable: a cell whose result is
already on disk is loaded rather than re-run, so it can be stopped and restarted, or widened by
adding sizes to a directory that already holds some.

The grid has cells that are deliberately lopsided. 32768 environments on 16 servers is 2048
battles per single-threaded Showdown process and will be server-bound; 512 environments on 128
servers is four battles each and wastes them. Both are run because the point of the sweep is to
show where those regimes begin, and each cell's environments-per-shard is recorded alongside its
throughput so the curve can be read with that in mind.
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_ENV_COUNTS = (512, 1024, 4096, 8192, 16384, 32768)
DEFAULT_SHARD_COUNTS = (16, 32, 64, 128)
# The columns lifted from each cell's benchmark result into the combined table, in order.
REPORTED_FIELDS = (
    "num_envs", "shards", "envs_per_shard", "envs_per_worker", "episodes_finished",
    "construction_seconds", "reset_seconds", "step_seconds",
    "effective_steps_per_second", "ongoing_battle_steps_per_second",
    "episodes_per_second", "steps_per_episode",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/sweep"), help="Per-cell JSON and the combined outputs go here.")
    parser.add_argument("--env-counts", type=int, nargs="+", default=list(DEFAULT_ENV_COUNTS))
    parser.add_argument("--shard-counts", type=int, nargs="+", default=list(DEFAULT_SHARD_COUNTS))
    parser.add_argument("--steps", type=int, default=300, help="Measured steps per cell; needs to exceed one episode (~65) to measure the effective rate.")
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--envs-per-worker", type=int, default=32)
    parser.add_argument("--step-timeout", type=float, default=120.0)
    parser.add_argument("--startup-timeout", type=float, help="Passed through; defaults to a budget that grows with the shard count.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cell-timeout", type=float, default=7200.0, help="Kill a cell that has not finished in this many seconds and record it as failed.")
    parser.add_argument("--rerun", action="store_true", help="Re-run cells whose result is already on disk instead of loading it.")
    parser.add_argument("--no-plot", action="store_true", help="Write the table but not the curve.")
    return parser.parse_args()


def cell_output_path(output_dir: Path, num_envs: int, shards: int) -> Path:
    return output_dir / f"fps_{num_envs}_shards_{shards}.json"


def run_cell(num_envs: int, shards: int, args: argparse.Namespace, output_path: Path) -> dict:
    """Run one benchmark configuration in its own process and return its result or a failure.

    A returned dict either carries the benchmark's measurements with status "ok", or names why the
    cell did not produce them. Isolating each cell in a subprocess is what lets an out-of-memory or
    out-of-ports configuration fail on its own rather than ending the sweep.
    """
    command = [
        sys.executable, str(Path(__file__).with_name("benchmark_fps.py")),
        "--num-envs", str(num_envs), "--shards", str(shards),
        "--steps", str(args.steps), "--warmup-steps", str(args.warmup_steps),
        "--envs-per-worker", str(args.envs_per_worker), "--step-timeout", str(args.step_timeout),
        "--seed", str(args.seed), "--output", str(output_path),
    ]
    if args.startup_timeout is not None:
        command += ["--startup-timeout", str(args.startup_timeout)]

    started = time.perf_counter()
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=args.cell_timeout, check=False)
    except subprocess.TimeoutExpired:
        return {"num_envs": num_envs, "shards": shards, "status": "timeout", "wall_seconds": round(time.perf_counter() - started, 1)}
    wall_seconds = round(time.perf_counter() - started, 1)

    if completed.returncode != 0 or not output_path.exists():
        # The last stderr line is usually the exception that ended the run, e.g. a startup timeout
        # or a refused port; the whole stream is kept beside the sweep for anything more.
        stderr_lines = [line for line in completed.stderr.splitlines() if line.strip()]
        (output_path.with_suffix(".stderr.log")).write_text(completed.stderr, encoding="utf-8")
        return {
            "num_envs": num_envs, "shards": shards, "status": "failed",
            "returncode": completed.returncode, "wall_seconds": wall_seconds,
            "error": stderr_lines[-1] if stderr_lines else "no output and no error text",
        }

    measurements = json.loads(output_path.read_text())
    measurements["status"] = "ok"
    measurements["wall_seconds"] = wall_seconds
    return measurements


def load_cell(output_path: Path) -> dict | None:
    """Return a previously saved cell result, or None if it is absent or unreadable."""
    if not output_path.exists():
        return None
    try:
        measurements = json.loads(output_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    measurements.setdefault("status", "ok")
    return measurements


def with_derived_fields(result: dict) -> dict:
    """Add the columns the table reports but the per-cell benchmark does not, such as the ratio."""
    if result.get("num_envs") and result.get("shards"):
        result["envs_per_shard"] = round(result["num_envs"] / result["shards"], 1)
    return result


def write_table(rows: list[dict], destination: Path) -> None:
    """Write the sweep as a CSV whose columns are the reported fields plus the run status.

    A failure's error text can hold commas, such as a list of addresses that stayed occupied, so
    the rows go through the csv writer rather than a bare join that would split them into columns.
    """
    columns = list(REPORTED_FIELDS) + ["status", "wall_seconds", "error"]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow(["" if row.get(column) is None else row.get(column) for column in columns])


def plot_curve(rows: list[dict], destination: Path) -> bool:
    """Plot effective and ongoing throughput against the environment count, one line per shard count.

    Returns whether a plot was produced; a sweep with no successful cell has nothing to draw.
    """
    successful = [row for row in rows if row.get("status") == "ok" and row.get("effective_steps_per_second")]
    if not successful:
        return False

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shard_counts = sorted({row["shards"] for row in successful})
    figure, (effective_axis, ongoing_axis) = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True)
    for axis, field, title in (
        (effective_axis, "effective_steps_per_second", "Effective throughput (episode boundaries included)"),
        (ongoing_axis, "ongoing_battle_steps_per_second", "Throughput between episode boundaries"),
    ):
        for shards in shard_counts:
            points = sorted(
                ((row["num_envs"], row[field]) for row in successful if row["shards"] == shards and row.get(field)),
                key=lambda point: point[0],
            )
            if points:
                xs, ys = zip(*points)
                axis.plot(xs, ys, marker="o", label=f"{shards} servers")
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xlabel("environments")
        axis.set_ylabel("environment steps / second")
        axis.set_title(title, fontsize=10)
        axis.grid(True, which="both", linewidth=0.3, alpha=0.5)
        axis.legend(title="Showdown servers", fontsize=8)

    figure.suptitle("Pokemon codesign environment throughput scaling")
    figure.tight_layout()
    figure.savefig(destination, dpi=130)
    plt.close(figure)
    return True


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    grid = [(num_envs, shards) for num_envs in args.env_counts for shards in args.shard_counts]
    print(f"sweeping {len(grid)} cells: envs={args.env_counts} x shards={args.shard_counts}", flush=True)

    rows = []
    for index, (num_envs, shards) in enumerate(grid, start=1):
        output_path = cell_output_path(args.output_dir, num_envs, shards)
        prefix = f"[{index}/{len(grid)}] envs={num_envs} shards={shards} ({num_envs / shards:.0f}/shard)"

        cached = None if args.rerun else load_cell(output_path)
        if cached is not None:
            result = with_derived_fields(cached)
            print(f"{prefix} -> cached {result.get('effective_steps_per_second')} FPS", flush=True)
        else:
            print(f"{prefix} -> running...", flush=True)
            result = with_derived_fields(run_cell(num_envs, shards, args, output_path))
            if result["status"] == "ok":
                print(f"{prefix} -> {result['effective_steps_per_second']} effective FPS "
                      f"({result['ongoing_battle_steps_per_second']} ongoing) in {result['wall_seconds']:.0f}s", flush=True)
            else:
                print(f"{prefix} -> {result['status'].upper()}: {result.get('error', '')}", flush=True)
        rows.append(result)

    table_path = args.output_dir / "sweep_results.csv"
    json_path = args.output_dir / "sweep_results.json"
    write_table(rows, table_path)
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {table_path} and {json_path}", flush=True)

    succeeded = [row for row in rows if row.get("status") == "ok"]
    failed = [row for row in rows if row.get("status") != "ok"]
    print(f"{len(succeeded)}/{len(rows)} cells succeeded", flush=True)
    if failed:
        for row in failed:
            print(f"  FAILED envs={row['num_envs']} shards={row['shards']}: {row['status']} {row.get('error', '')}", flush=True)
    if succeeded:
        best = max(succeeded, key=lambda row: row["effective_steps_per_second"])
        print(f"peak effective FPS: {best['effective_steps_per_second']} at "
              f"{best['num_envs']} envs / {best['shards']} servers", flush=True)

    if not args.no_plot:
        plot_path = args.output_dir / "scaling_curve.png"
        if plot_curve(rows, plot_path):
            print(f"wrote {plot_path}", flush=True)
        else:
            print("no successful cells to plot", flush=True)


if __name__ == "__main__":
    main()
