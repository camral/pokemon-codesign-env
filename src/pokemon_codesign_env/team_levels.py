"""Shared level sampling for learner and random opponent teams."""

from __future__ import annotations

import numpy as np
from poke_env.teambuilder import Teambuilder

LEVEL_MEAN = 50
LEVEL_STD = 0.0


def sample_level(
    rng: np.random.Generator,
    *,
    minimum: int,
    maximum: int,
    mean: float = LEVEL_MEAN,
    std: float = LEVEL_STD,
) -> int:
    """Sample one rounded normal level and clip it to format limits."""

    return int(np.clip(round(float(rng.normal(mean, std))), minimum, maximum))


def resample_packed_team_levels(
    packed_team: str,
    rng: np.random.Generator,
    *,
    minimum: int,
    maximum: int,
    mean: float = LEVEL_MEAN,
    std: float = LEVEL_STD,
) -> str:
    """Return a packed team whose Pokemon levels use the shared distribution."""

    pokemon = Teambuilder.parse_packed_team(packed_team)
    for member in pokemon:
        member.level = sample_level(rng, minimum=minimum, maximum=maximum, mean=mean, std=std)
    return Teambuilder.join_team(pokemon)
