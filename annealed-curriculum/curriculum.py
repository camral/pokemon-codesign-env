"""Team population and annealing primitives for the curriculum baseline."""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from poke_env.data import GenData
from poke_env.data.normalize import to_id_str
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.dqn import DQN
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork

from pokemon_codesign_env.team_designer import TeamDesigner
from pokemon_codesign_env.team_pool import validate_team_batch


@dataclass(frozen=True)
class TeamCandidate:
    """A complete design and its immutable packed Showdown representation."""

    parameters: dict
    packed_team: str


class TeamPopulation:
    """Sample and mutate complete teams while leaving moves outside the search action."""

    def __init__(self, search_config, seed: int = 0):
        self.search_config = search_config
        self.rng = np.random.default_rng(seed)
        self.designer = TeamDesigner(search_config, validate_teams=False)
        self.attributes = (*self.designer.optimizable_attributes["pokemon_0"], "level")

    def _parameter_space(self, attribute: str, species: str | None = None) -> dict:
        if attribute == "level":
            return {"type": "integer", "min": max(1, self.search_config.format.min_level), "max": min(100, self.search_config.format.max_level)}
        return self.designer.get_parameter_space(attribute, species)

    def _sample_parameters(self) -> dict:
        parameters = {}
        for pokemon_index in range(self.search_config.team_size):
            pokemon = f"pokemon_{pokemon_index}"
            values = {}
            if "species" in self.attributes:
                values["species"] = self.designer.sample_parameter(self._parameter_space("species"))
            species = values.get("species")
            for attribute in self.attributes:
                if attribute != "species":
                    values[attribute] = self.designer.sample_parameter(self._parameter_space(attribute, species))
            parameters[pokemon] = values
        return parameters

    def _mutate_parameters(self, base: dict, probability: float) -> dict:
        values = deepcopy(base)
        mutation_mask = self.rng.random((self.search_config.team_size, len(self.attributes))) < probability
        for pokemon_index in range(self.search_config.team_size):
            pokemon = f"pokemon_{pokemon_index}"
            old_species = values[pokemon].get("species")
            if "species" in self.attributes and mutation_mask[pokemon_index, self.attributes.index("species")]:
                values[pokemon]["species"] = self.designer.sample_parameter(self._parameter_space("species"))
            species = values[pokemon].get("species")
            species_changed = species != old_species
            for attribute_index, attribute in enumerate(self.attributes):
                should_mutate = mutation_mask[pokemon_index, attribute_index]
                should_refresh_dependency = species_changed and attribute in {"ability", "gender", "tera_type"}
                if attribute != "species" and (should_mutate or should_refresh_dependency):
                    values[pokemon][attribute] = self.designer.sample_parameter(self._parameter_space(attribute, species))
        return values

    def _serialize(self, parameters: list[dict]) -> list[TeamCandidate]:
        pending = [values if self._species_are_trackable(values) else self._sample_trackable_parameters() for values in parameters]
        candidates = []
        attempts = 0
        while pending and attempts < 20:
            packed_teams = []
            for values in pending:
                design_values = {
                    pokemon: {name: value for name, value in pokemon_values.items() if name != "level"}
                    for pokemon, pokemon_values in values.items()
                }
                self.designer.set_optimizable_parameter_values(design_values)
                self.designer.rng = np.random.default_rng(int(self.rng.integers(2**32)))
                observations = self.designer.sample_observations()
                for pokemon, pokemon_values in values.items():
                    observations[pokemon]["level"] = pokemon_values["level"]
                self.designer.sampled_observations = observations
                packed_teams.append(self.designer.yield_team())
            errors = validate_team_batch(self.search_config.format.showdown_path, self.search_config.format.format_id, packed_teams)
            candidates.extend(TeamCandidate(values, team) for values, team, error in zip(pending, packed_teams, errors, strict=True) if error is None)
            pending = [self._sample_trackable_parameters() for error in errors if error is not None]
            attempts += 1
        if pending:
            raise RuntimeError(f"Could not produce {len(pending)} valid curriculum teams after {attempts} batches")
        return candidates

    def _sample_trackable_parameters(self) -> dict:
        for _ in range(100):
            parameters = self._sample_parameters()
            if self._species_are_trackable(parameters):
                return parameters
        raise RuntimeError("Could not sample a poke-env-trackable team after 100 attempts")

    def _species_are_trackable(self, parameters: dict) -> bool:
        """Avoid species-name collisions that poke-env cannot distinguish."""
        pokedex = GenData.from_gen(self.search_config.format.generation).pokedex
        species = [pokemon["species"] for pokemon in parameters.values()]
        base_species = [to_id_str(pokedex.get(to_id_str(name), {}).get("baseSpecies", name)) for name in species]
        for index, base_name in enumerate(base_species):
            for other_index, other_name in enumerate(species):
                aliases = {to_id_str(part) for part in other_name.split("-")}
                if index != other_index and (base_name == to_id_str(other_name) or base_name in aliases):
                    return False
        return True

    def sample(self, size: int) -> list[TeamCandidate]:
        """Return ``size`` independently sampled complete teams."""
        return self._serialize([self._sample_parameters() for _ in range(size)])

    def mutate(self, reservoir: list[TeamCandidate], size: int, probability: float) -> list[TeamCandidate]:
        """Vectorize mutation decisions, then resolve the ragged species-dependent domains."""
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"Mutation probability must be in [0, 1], got {probability}")
        bases = self.rng.integers(len(reservoir), size=size)
        return self._serialize([self._mutate_parameters(reservoir[int(index)].parameters, probability) for index in bases])


class TeamEmbeddingExtractor(BaseFeaturesExtractor):
    """Embed every species ID through one table and every move ID through another."""

    def __init__(
        self, observation_space, species_positions: tuple[int, ...], move_positions: tuple[int, ...], species_count: int, move_count: int,
        embedding_dim: int = 16,
    ):
        categorical_size = int(np.prod(observation_space["categorical"].shape))
        embedded_positions = set(species_positions) | set(move_positions)
        scalar_positions = tuple(index for index in range(categorical_size) if index not in embedded_positions)
        scalar_size = len(scalar_positions) + int(np.prod(observation_space["continuous"].shape)) + int(np.prod(observation_space["binary"].shape))
        features_dim = (len(species_positions) + len(move_positions)) * embedding_dim + scalar_size
        super().__init__(observation_space, features_dim)
        self.species_positions = species_positions
        self.move_positions = move_positions
        self.scalar_positions = scalar_positions
        self.species_embedding = torch.nn.Embedding(species_count + 1, embedding_dim)
        self.move_embedding = torch.nn.Embedding(move_count + 1, embedding_dim)

    def forward(self, observations):
        categorical = observations["categorical"].long()
        parts = []
        if self.species_positions:
            parts.append(self.species_embedding(categorical[:, self.species_positions]).flatten(start_dim=1))
        if self.move_positions:
            parts.append(self.move_embedding(categorical[:, self.move_positions]).flatten(start_dim=1))
        if self.scalar_positions:
            parts.append(categorical[:, self.scalar_positions].float())
        parts.extend((observations["continuous"].float(), observations["binary"].float()))
        return torch.cat(parts, dim=1)


class MaskedQNetwork(QNetwork):
    """Assign impossible actions negative infinity before selection and TD targets."""

    def extract_features(self, observations, features_extractor=None):
        """Preserve raw integer IDs instead of SB3's automatic one-hot conversion."""
        return (features_extractor or self.features_extractor)(observations)

    def forward(self, observations):
        values = super().forward(observations)
        legal = observations["action_mask"].bool()
        no_legal_action = ~legal.any(dim=1)
        if no_legal_action.any():
            legal = legal.clone()
            legal[no_legal_action, 0] = True
        return values.masked_fill(~legal, torch.finfo(values.dtype).min)


class EmbeddedMaskedDQNPolicy(DQNPolicy):
    """Build both online and target critics with shared-ID embedding extractors."""

    def make_q_net(self):
        return MaskedQNetwork(**self._update_features_extractor(self.net_args, features_extractor=None)).to(self.device)


class MaskedDQN(DQN):
    """Restrict epsilon-random exploration to actions reported legal by the environment."""

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        if not deterministic and self._last_obs is not None and self.exploration_rate > 0 and np.random.rand() < self.exploration_rate:
            masks = np.asarray(observation["action_mask"])
            if masks.ndim == 1:
                masks = masks[None, :]
            actions = np.asarray([self.action_space.sample(mask=np.asarray(mask, dtype=np.int8)) for mask in masks])
            return actions, state
        return self.policy.predict(observation, state, episode_start, deterministic)


def embedding_positions(observation_encoder) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Locate species and move IDs in the encoder's categorical vector."""
    species = []
    moves = []
    for name, (group, index) in observation_encoder.feature_indices.items():
        if group == "categorical" and "species_id" in name:
            species.append(index)
        elif group == "categorical" and "move_id" in name:
            moves.append(index)
    return tuple(species), tuple(moves)


def annealed_value(initial: float, final: float, progress: float) -> float:
    """Linearly interpolate a curriculum value over clipped phase-one progress."""
    progress = min(max(progress, 0.0), 1.0)
    return initial + (final - initial) * progress


def aggregate_fitness(values: np.ndarray, method: str) -> np.ndarray:
    """Aggregate the Y Q estimates associated with every candidate."""
    if method == "minimum":
        return values.min(axis=1)
    if method == "percentile10":
        return np.percentile(values, 10, axis=1)
    if method == "bottom-quartile":
        count = max(1, int(np.ceil(values.shape[1] / 4)))
        return np.sort(values, axis=1)[:, :count].mean(axis=1)
    raise ValueError(f"Unknown fitness aggregation {method!r}")
