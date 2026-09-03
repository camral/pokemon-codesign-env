"""Pokemon design graph used by House of Dextra graph heuristic search."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import numpy as np

from .team_design import TeamDesignVectorAdapter

HASH_BITS = 32
TEAM_SLOTS = 6
STATS = ("hp", "atk", "def", "spa", "spd", "spe")
HASHED_ATTRIBUTES = ("species", "item", "nature", "ability", "gender", "tera_type")


def _binary_identity(value: str | None) -> np.ndarray:
    """Encode a category without imposing numeric proximity between category IDs."""

    if value is None:
        return np.zeros(HASH_BITS, dtype=np.float32)
    digest = hashlib.sha256(value.encode()).digest()[: HASH_BITS // 8]
    return np.unpackbits(np.frombuffer(digest, dtype=np.uint8)).astype(np.float32)


class PokemonTeamGraphAdapter:
    """Seven-node star graph backed exclusively by ``TeamDesignVectorAdapter`` values."""

    def __init__(self, vector_adapter: TeamDesignVectorAdapter):
        if vector_adapter.search_config.team_size != TEAM_SLOTS:
            raise ValueError("House of Dextra requires exactly six Pokemon slots")
        self.vector_adapter = vector_adapter
        self.search_config = vector_adapter.search_config
        self.edges = np.asarray(
            [
                [node for node in range(1, TEAM_SLOTS + 1) for _ in range(2)],
                [value for node in range(1, TEAM_SLOTS + 1) for value in (node, 0)],
            ],
            dtype=np.int64,
        )
        # Correct the alternating source endpoints to root->slot and slot->root.
        self.edges[0] = [value for node in range(1, TEAM_SLOTS + 1) for value in (0, node)]

    @property
    def feature_dimension(self) -> int:
        return 2 + TEAM_SLOTS + HASH_BITS * (len(HASHED_ATTRIBUTES) + 4) + 1 + 12

    def initial_design(self, rng: np.random.Generator) -> dict[str, dict]:
        return self.vector_adapter.decode(rng.uniform(size=self.vector_adapter.dimension))

    def mutate(self, design: dict[str, dict], rng: np.random.Generator) -> dict[str, dict]:
        """Change one canonical field while preserving its local constraints."""

        self.vector_adapter.designer.set_optimizable_parameter_values(design)
        fields = self.vector_adapter.fields
        field_index = int(rng.integers(len(fields)))
        field = fields[field_index]
        if field.parameter in {"evs", "ivs"}:
            transferred = self._transfer_stat_points(design, field, rng)
            if transferred is not None:
                return transferred

        vector = self.vector_adapter.encode(design)
        start = sum(item.width for item in fields[:field_index])
        coordinate = start + int(rng.integers(field.width))
        original = design[field.pokemon][field.parameter]
        for _ in range(64):
            candidate_vector = vector.copy()
            candidate_vector[coordinate] = rng.random()
            candidate = self.vector_adapter.decode(candidate_vector)
            if candidate[field.pokemon][field.parameter] != original:
                return candidate
        return self.vector_adapter.decode(vector)

    def _transfer_stat_points(self, design, field, rng):
        candidate = deepcopy(design)
        allocation = candidate[field.pokemon][field.parameter]
        space = self.vector_adapter.designer.get_parameter_space(field.parameter)
        cap = int(space["max_per_stat"])
        pairs = [
            (donor, receiver)
            for donor in STATS
            for receiver in STATS
            if donor != receiver and allocation[donor] > 0 and allocation[receiver] < cap
        ]
        if not pairs:
            return None
        donor, receiver = pairs[int(rng.integers(len(pairs)))]
        amount = int(rng.integers(1, min(allocation[donor], cap - allocation[receiver]) + 1))
        allocation[donor] -= amount
        allocation[receiver] += amount
        self.vector_adapter.designer.set_optimizable_parameter_values(candidate)
        return candidate

    def encode(self, design: dict[str, dict]) -> tuple[np.ndarray, np.ndarray]:
        """Return root-plus-six-slot features and bidirectional star edges."""

        self.vector_adapter.designer.set_optimizable_parameter_values(design)
        nodes = np.zeros((TEAM_SLOTS + 1, self.feature_dimension), dtype=np.float32)
        nodes[0, 0] = 1.0
        for slot in range(TEAM_SLOTS):
            values = design[f"pokemon_{slot}"]
            features = [np.asarray([0.0, 1.0], dtype=np.float32)]
            slot_identity = np.zeros(TEAM_SLOTS, dtype=np.float32)
            slot_identity[slot] = 1.0
            features.append(slot_identity)
            for attribute in HASHED_ATTRIBUTES:
                features.append(_binary_identity(values.get(attribute)))
            moves = list(values["moves"][:4])
            moves.extend([None] * (4 - len(moves)))
            features.extend(_binary_identity(move) for move in moves)
            features.append(np.asarray([values["happiness"] / 255.0], dtype=np.float32))
            features.append(
                np.asarray(
                    [
                        values["evs"][stat] / max(1, self.search_config.ev_per_stat_cap)
                        for stat in STATS
                    ],
                    dtype=np.float32,
                )
            )
            features.append(
                np.asarray(
                    [values["ivs"][stat] / max(1, self.search_config.iv_max) for stat in STATS],
                    dtype=np.float32,
                )
            )
            nodes[slot + 1] = np.concatenate(features)
        return nodes, self.edges.copy()

    def serialize(self, design: dict[str, dict]) -> dict[str, dict]:
        return json.loads(self.vector_adapter.serialize(design))

    def schema(self) -> dict:
        return {
            "version": 1,
            "nodes": 7,
            "topology": "bidirectional_star",
            "feature_dimension": self.feature_dimension,
            "categorical_encoding": f"sha256-{HASH_BITS}-bit",
            "team_design_schema": self.vector_adapter.schema(),
        }
