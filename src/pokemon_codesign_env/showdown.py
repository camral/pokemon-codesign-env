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

    result = subprocess.run(["node", "-e", script, format_id], cwd=showdown_path, capture_output=True, check=False, text=True)
    if result.returncode:
        message = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Could not resolve Pokemon Showdown format {format_id!r}: {message}")

    return json.loads(result.stdout)


def get_parameter_values(
    format_id: str,
    param: str,
    showdown_path: Path,
    species: str | None = None,
    generations_allowed: tuple[int, ...] | None = None,
    types_allowed: tuple[str, ...] | None = None,
) -> list[str | None]:
    """Return concrete Pokemon Showdown values for a team parameter."""
    script = """
        const { Dex } = require('./dist/sim/dex');
        const { TeamValidator } = require('./dist/sim/team-validator');

        const formatId = process.argv[1];
        const param = process.argv[2];
        const requestedSpecies = process.argv[3];
        const filters = JSON.parse(process.argv[4]);
        const format = Dex.formats.get(formatId);
        if (!format.exists) {
            throw new Error(`Unknown Pokemon Showdown format: ${formatId}`);
        }

        const dex = Dex.forFormat(format);
        const validator = TeamValidator.get(format.id);
        const generations = filters.generations;
        const types = filters.types.map(type => type.toLowerCase());
        const isEligibleSpecies = species => (
            species.exists &&
            !validator.ruleTable.isBannedSpecies(species) &&
            (!generations.length || generations.includes(species.gen)) &&
            (!types.length || species.types.some(type => types.includes(type.toLowerCase())))
        );

        let speciesPool = dex.species.all().filter(isEligibleSpecies);
        if (requestedSpecies) {
            const selectedSpecies = dex.species.get(requestedSpecies);
            speciesPool = isEligibleSpecies(selectedSpecies) ? [selectedSpecies] : [];
        }

        let values;
        if (param === 'species') {
            values = speciesPool.map(species => species.name);
        } else if (param === 'moves') {
            const moveIds = new Set();
            for (const species of speciesPool) {
                for (const learnsetData of dex.species.getFullLearnset(species.id)) {
                    for (const moveId of Object.keys(learnsetData.learnset)) moveIds.add(moveId);
                }
            }
            values = Array.from(moveIds)
                .map(moveId => dex.moves.get(moveId))
                .filter(move => move.exists && !validator.checkMove({ name: 'Pokemon' }, move, {}))
                .map(move => move.name);
        } else if (param === 'ability') {
            const abilityNames = new Set();
            for (const species of speciesPool) {
                for (const ability of Object.values(species.abilities)) abilityNames.add(ability);
            }
            values = Array.from(abilityNames)
                .map(abilityName => dex.abilities.get(abilityName))
                .filter(ability => ability.exists && !validator.checkAbility({ name: 'Pokemon' }, ability, {}))
                .map(ability => ability.name);
            if (dex.gen <= 2) values = [null];
        } else if (param === 'item') {
            values = dex.items.all()
                .filter(item => item.exists && !validator.checkItem({ name: 'Pokemon' }, item, {}))
                .map(item => item.name);
            if (!validator.checkItem({ name: 'Pokemon' }, dex.items.get(''), {})) values.unshift(null);
        } else if (param === 'nature') {
            values = dex.natures.all()
                .filter(nature => nature.exists && !validator.checkNature({ name: 'Pokemon' }, nature, {}))
                .map(nature => nature.name);
            if (dex.gen <= 2) values = [null];
        } else if (param === 'gender') {
            const genders = new Set();
            for (const species of speciesPool) {
                if (species.gender) {
                    genders.add(species.gender);
                } else {
                    genders.add('M');
                    genders.add('F');
                }
            }
            values = Array.from(genders);
        } else {
            throw new Error(`Unsupported Pokemon parameter: ${param}`);
        }

        values.sort((left, right) => (left || '').localeCompare(right || ''));
        console.log(JSON.stringify(values));
    """

    filters = {"generations": list(generations_allowed or ()), "types": list(types_allowed or ())}
    result = subprocess.run(
        ["node", "-e", script, format_id, param, species or "", json.dumps(filters)],
        cwd=showdown_path,
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode:
        message = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Could not resolve Pokemon Showdown values for {param!r}: {message}")

    return json.loads(result.stdout)
