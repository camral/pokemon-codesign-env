import json
import subprocess
from functools import lru_cache
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
            supports_tera_type: (
                dex.gen === 9 && !dex.currentMod.startsWith('champions') &&
                !ruleTable.has('terastalclause')
            ) || ruleTable.has('bonustypemod'),
        }));
    """

    result = subprocess.run(["node", "-e", script, format_id], cwd=showdown_path, capture_output=True, check=False, text=True)
    if result.returncode:
        message = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Could not resolve Pokemon Showdown format {format_id!r}: {message}")

    return json.loads(result.stdout)


@lru_cache(maxsize=None)
def _get_parameter_catalog(
    format_id: str, showdown_path: str, generations_allowed: tuple[int, ...], types_allowed: tuple[str, ...]
) -> dict:
    """Load every configurable value for a format in one Showdown process."""
    script = """
        const { Dex } = require('./dist/sim/dex');
        const { TeamValidator } = require('./dist/sim/team-validator');

        const formatId = process.argv[1];
        const filters = JSON.parse(process.argv[2]);
        const format = Dex.formats.get(formatId);
        if (!format.exists) {
            throw new Error(`Unknown Pokemon Showdown format: ${formatId}`);
        }

        const dex = Dex.forFormat(format);
        const validator = TeamValidator.get(format.id);
        const generations = filters.generations;
        const types = filters.types.map(type => type.toLowerCase());
        const restrictToFormatGeneration = format.id.endsWith('pokemonleague');
        const isEligibleSpecies = species => (
            species.exists &&
            !validator.ruleTable.isBannedSpecies(species) &&
            (!restrictToFormatGeneration || species.gen <= dex.gen) &&
            (!generations.length || generations.includes(species.gen)) &&
            (!types.length || species.types.some(type => types.includes(type.toLowerCase())))
        );

        const speciesPool = dex.species.all().filter(isEligibleSpecies);
        const legalTypes = dex.types.all().filter(type => type.exists && !type.isNonstandard).map(type => type.name);
        const supportsTera = (
            dex.gen === 9 && !dex.currentMod.startsWith('champions') && !validator.ruleTable.has('terastalclause')
        ) || validator.ruleTable.has('bonustypemod');
        const catalog = {
            species: speciesPool.map(species => species.name),
            aliases: {},
            moves: {},
            ability: {},
            gender: {},
            tera_type: {},
            item: dex.items.all()
                .filter(item => item.exists && !validator.checkItem({ name: 'Pokemon' }, item, {}))
                .map(item => item.name),
            nature: dex.gen <= 2 ? [null] : dex.natures.all()
                .filter(nature => nature.exists && !validator.checkNature({ name: 'Pokemon' }, nature, {}))
                .map(nature => nature.name),
        };
        if (!validator.checkItem({ name: 'Pokemon' }, dex.items.get(''), {})) catalog.item.unshift(null);

        for (const species of speciesPool) {
            catalog.aliases[species.id] = species.name;
            catalog.aliases[species.name.toLowerCase()] = species.name;

            const moveIds = new Set();
            for (const learnsetData of dex.species.getFullLearnset(species.id)) {
                for (const moveId of Object.keys(learnsetData.learnset)) moveIds.add(moveId);
            }
            catalog.moves[species.name] = Array.from(moveIds)
                .map(moveId => dex.moves.get(moveId))
                .filter(move => move.exists && move.gen <= dex.gen && !validator.checkMove({ name: 'Pokemon' }, move, {}))
                .map(move => move.name)
                .sort((left, right) => left.localeCompare(right));

            if (dex.gen <= 2) {
                catalog.ability[species.name] = [null];
            } else {
                catalog.ability[species.name] = Array.from(new Set(Object.values(species.abilities)))
                    .map(abilityName => dex.abilities.get(abilityName))
                    .filter(ability => ability.exists && !validator.checkAbility({ name: 'Pokemon' }, ability, {}))
                    .map(ability => ability.name);
            }
            catalog.gender[species.name] = species.gender ? [species.gender] : ['M', 'F'];
            if (!supportsTera) {
                catalog.tera_type[species.name] = [null];
            } else if (species.requiredTeraType) {
                catalog.tera_type[species.name] = [species.requiredTeraType];
            } else {
                catalog.tera_type[species.name] = [species.types[0], ...legalTypes.filter(type => type !== species.types[0])];
            }
        }

        for (const param of ['species', 'item', 'nature']) catalog[param].sort((left, right) => (left || '').localeCompare(right || ''));
        console.log(JSON.stringify(catalog));
    """

    filters = {"generations": list(generations_allowed), "types": list(types_allowed)}
    result = subprocess.run(
        ["node", "-e", script, format_id, json.dumps(filters)],
        cwd=showdown_path,
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode:
        message = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Could not resolve Pokemon Showdown parameter catalog: {message}")

    return json.loads(result.stdout)


def get_parameter_values(
    format_id: str,
    param: str,
    showdown_path: Path,
    species: str | None = None,
    generations_allowed: tuple[int, ...] | None = None,
    types_allowed: tuple[str, ...] | None = None,
) -> list[str | None]:
    """Return cached concrete Pokemon Showdown values for a team parameter."""
    showdown_path = str(Path(showdown_path).resolve())
    generations_allowed = tuple(sorted(set(generations_allowed or ())))
    types_allowed = tuple(sorted({pokemon_type.lower() for pokemon_type in types_allowed or ()}))
    catalog = _get_parameter_catalog(format_id, showdown_path, generations_allowed, types_allowed)

    if param == "species" and species is not None:
        canonical_species = catalog["aliases"].get(species, catalog["aliases"].get(species.lower()))
        return [] if canonical_species is None else [canonical_species]
    if param in {"species", "item", "nature"}:
        return list(catalog[param])
    if param not in {"moves", "ability", "gender", "tera_type"}:
        raise ValueError(f"Unsupported Pokemon parameter: {param}")
    if species is not None:
        canonical_species = catalog["aliases"].get(species, catalog["aliases"].get(species.lower()))
        return [] if canonical_species is None else list(catalog[param][canonical_species])

    values = {value for species_values in catalog[param].values() for value in species_values}
    return sorted(values, key=lambda value: value or "")


def validate_team(format_id: str, packed_team: str, showdown_path: Path) -> None:
    """Validate a packed team with the local Pokemon Showdown installation."""
    result = subprocess.run(
        [str(showdown_path / "pokemon-showdown"), "validate-team", format_id],
        cwd=showdown_path,
        input=packed_team,
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode:
        message = result.stderr.strip() or result.stdout.strip()
        raise ValueError(f"Pokemon Showdown rejected the generated team: {message}")
