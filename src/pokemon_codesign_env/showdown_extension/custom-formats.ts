import { toID } from '../sim/dex';
import type { Battle } from '../sim/battle';
import type { Pokemon } from '../sim/pokemon';

const STATE_PREFIX = 'pcg1';
const STATUS_CODES: { [code: string]: string } = {
	b: 'brn',
	f: 'frz',
	a: 'par',
	p: 'psn',
	s: 'slp',
	t: 'tox',
};

function restorePokemonState(this: Battle, pokemon: Pokemon) {
	const state = toID(pokemon.set.pokeball);
	if (!state.startsWith(STATE_PREFIX)) return false;

	const encoded = state.slice(STATE_PREFIX.length);
	if (encoded.length < 4 || encoded.length % 2) {
		throw new Error(`Invalid Pokemon co-design state: ${state}`);
	}

	const hp = parseInt(encoded.slice(0, 3), 36);
	const status = STATUS_CODES[encoded.charAt(3)] || '';
	const pp = [];
	for (let index = 4; index < encoded.length; index += 2) {
		pp.push(parseInt(encoded.slice(index, index + 2), 36));
	}
	if (!Number.isFinite(hp) || pp.some(value => !Number.isFinite(value))) {
		throw new Error(`Invalid Pokemon co-design state: ${state}`);
	}

	pokemon.hp = this.clampIntRange(hp, 0, pokemon.maxhp);
	for (const [index, value] of pp.entries()) {
		const moveSlot = pokemon.baseMoveSlots[index];
		if (moveSlot) moveSlot.pp = this.clampIntRange(value, 0, moveSlot.maxpp);
	}

	if (!pokemon.hp) {
		pokemon.fainted = true;
		pokemon.status = 'fnt' as ID;
	} else if (status) {
		const condition = this.dex.conditions.get(status);
		pokemon.status = condition.id;
		pokemon.statusState = this.initEffectState({ id: condition.id, target: pokemon });
		if (condition.duration) pokemon.statusState.duration = condition.duration;
		if (condition.durationCallback) {
			pokemon.statusState.duration = condition.durationCallback.call(this, pokemon, pokemon, null);
		}
		this.singleEvent('Start', condition, pokemon.statusState, pokemon);
	}
	return true;
}

function restoreLeagueState(this: Battle) {
	for (const side of this.sides) {
		let persistentTeam = false;
		for (const pokemon of side.pokemon) {
			if (restorePokemonState.call(this, pokemon)) persistentTeam = true;
		}
		if (!persistentTeam) continue;

		side.pokemon.sort((left, right) => Number(left.fainted) - Number(right.fainted));
		for (const [position, pokemon] of side.pokemon.entries()) pokemon.position = position;
		side.pokemonLeft = side.pokemon.filter(pokemon => !pokemon.fainted).length;
		side.totalFainted = side.pokemon.length - side.pokemonLeft;
	}
}

const leagueFormats = Array.from({ length: 9 }, (_, index) => {
	const generation = index + 1;
	return {
		name: `[Gen ${generation}] Pokemon League`,
		mod: `gen${generation}`,
		searchShow: false,
		debug: true,
		battle: { trunc: Math.trunc },
		ruleset: generation >= 5 ?
			[`[Gen ${generation}] Custom Game`, '!Team Preview'] : [`[Gen ${generation}] Custom Game`],
		onBattleStart: restoreLeagueState,
	};
});

export const Formats: import('../sim/dex-formats').FormatList = [
	{
		section: "Pokemon Co-Design",
	},
	...leagueFormats,
];
