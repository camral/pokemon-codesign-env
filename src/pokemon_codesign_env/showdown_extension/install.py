import argparse
import subprocess
from importlib.resources import files
from pathlib import Path


def install_showdown_extension(showdown_path: Path, build: bool = True) -> Path:
    """Install the Pokemon co-design formats into a Pokemon Showdown server."""
    showdown_path = Path(showdown_path).resolve()
    config_path = showdown_path / "config"
    package_path = showdown_path / "package.json"
    if not config_path.is_dir() or not package_path.is_file():
        raise ValueError(f"Not a Pokemon Showdown installation: {showdown_path}")

    resources = files(__package__)
    extension = resources.joinpath("custom-formats.ts").read_text(encoding="utf-8")
    wrapper = resources.joinpath("custom-formats-wrapper.ts").read_text(encoding="utf-8")
    extension_path = config_path / "pokemon-codesign-formats.ts"
    custom_formats_path = config_path / "custom-formats.ts"

    if custom_formats_path.exists():
        current = custom_formats_path.read_text(encoding="utf-8")
        if current != wrapper and "pokemon-codesign-formats" not in current:
            raise FileExistsError(
                f"{custom_formats_path} already defines custom formats. Import Formats from "
                "'./pokemon-codesign-formats' and spread them into its Formats array."
            )

    extension_path.write_text(extension, encoding="utf-8")
    if not custom_formats_path.exists():
        custom_formats_path.write_text(wrapper, encoding="utf-8")

    if build:
        subprocess.run(["node", "build"], cwd=showdown_path, check=True)
    return extension_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install the Pokemon co-design custom formats into Pokemon Showdown."
    )
    parser.add_argument("showdown_path", type=Path)
    parser.add_argument("--no-build", action="store_true")
    args = parser.parse_args()
    path = install_showdown_extension(args.showdown_path, build=not args.no_build)
    print(f"Installed Pokemon co-design Showdown formats at {path}")


if __name__ == "__main__":
    main()
