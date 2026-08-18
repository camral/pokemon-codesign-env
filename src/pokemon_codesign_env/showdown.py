import json
import subprocess
from pathlib import Path


def get_format_rules(format_id: str, showdown_path: Path) -> dict[str, object]:
    """Return the resolved rules for a Pokemon Showdown format."""
    script = """
        const { Dex } = require('./dist/sim/dex');
        const { TeamValidator } = require('./dist/sim/team-validator');

        const formatId = process.argv[1];
        const format = Dex.formats.get(formatId);
        if (!format.exists) {
            throw new Error(`Unknown Pokemon Showdown format: ${formatId}`);
        }

        const dex = Dex.forFormat(format);
        const ruleTable = TeamValidator.get(format.id).ruleTable;
        const usesAwakeningValues = dex.currentMod === 'gen7letsgo';
        const usesStatPoints = dex.currentMod.startsWith('champions');

        console.log(JSON.stringify({
            format_id: format.id,
            format_name: format.name,
            generation: dex.gen,
            min_team_size: ruleTable.minTeamSize,
            max_team_size: ruleTable.maxTeamSize,
            picked_team_size: ruleTable.pickedTeamSize,
            max_move_count: ruleTable.maxMoveCount,
            min_level: ruleTable.minLevel,
            max_level: ruleTable.maxLevel,
            default_level: ruleTable.defaultLevel,
            adjust_level: ruleTable.adjustLevel,
            adjust_level_down: ruleTable.adjustLevelDown,
            ev_limit: ruleTable.evLimit,
            ev_per_stat_cap: usesAwakeningValues ? 200 : usesStatPoints ? 32 : 255,
            iv_max: dex.gen <= 2 ? 30 : 31,
        }));
    """

    result = subprocess.run(
        ["node", "-e", script, format_id],
        cwd=showdown_path,
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode:
        message = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Could not resolve Pokemon Showdown format {format_id!r}: {message}")

    return json.loads(result.stdout)
