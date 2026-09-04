import gymnasium
from gymnasium import spaces
import numpy as np

from pathlib import Path

from poke_env.environment import SinglesEnv

from pokemon_codesign_env.configs import TeamSearchConfig, FormatConfig, ObservationConfig
from pokemon_codesign_env.envs import MaxBasePowerPlayerEnv,PokemonLeagueEnv,RandomPlayerEnv,SimpleHeuristicsPlayerEnv,get_pokemon_league
from pokemon_codesign_env.observations import HumanVisibleBattleEncoder
from pokemon_codesign_env.spaces import TeamDesignSpace
from pokemon_codesign_env.team_designer import TeamDesigner

OPPONENTS = {
    "random": RandomPlayerEnv,
    "max_base_power": MaxBasePowerPlayerEnv,
    "simple_heuristics": SimpleHeuristicsPlayerEnv,
    "league": PokemonLeagueEnv,
}


class PokEnv(gymnasium.Env):
    """A Pokemon Showdown co-design environment.

    Extends the Gymnasium API with an embodiment design space. `embodiment_space` describes the
    team designs this environment accepts; `reset` takes the embodiment to battle with. The team
    is fixed once a battle begins, so this environment does not support transition-level
    embodiment changes and `step` ignores an embodiment with a warning.
    """

    metadata = {"render_modes": ["human"]}
    SUPPORTS_TRANSITION_EMBODIMENT = False

    def __init__(
        self,
        opponent: str = "random",
        format_config: FormatConfig | None = None,
        embodiment_config: TeamSearchConfig | None = None,
        observation_config: ObservationConfig | None = None,
        is_eval_env: bool = False,
        max_action_space_size: int | None = None,
        showdown_path: Path | None = None,
        **env_kwargs,
    ):
        super().__init__()

        if opponent not in OPPONENTS:
            raise ValueError(f"opponent must be one of {sorted(OPPONENTS)}, got {opponent!r}")
        self.opponent = opponent
        self.environment_class = OPPONENTS[opponent]

        self.is_eval_env = is_eval_env

        # Resolve the battle format. League environments select theirs from a generation instead
        # of a class attribute, so they cannot use the BATTLE_FORMAT preset.
        if format_config is None:
            format_config = FormatConfig.from_showdown(self._default_battle_format(env_kwargs), showdown_path)
        self.format_config = format_config

        # The design space and the battle must describe the same format, or the designed team is
        # legal for a format this environment never plays.
        if embodiment_config is None:
            embodiment_config = TeamSearchConfig(format=format_config)
        elif embodiment_config.format.format_id != format_config.format_id:
            raise ValueError(
                f"embodiment_config designs teams for {embodiment_config.format.format_id!r} but the "
                f"environment battles in {format_config.format_id!r}"
            )
        self.embodiment_config = embodiment_config
        self.observation_config = observation_config or ObservationConfig()

        # The embodiment design space, and the compiler that turns a member of it into a team.
        self.embodiment_space = TeamDesignSpace(embodiment_config)
        self.team_designer = TeamDesigner(embodiment_config)

        # The wrapped battle environment, which owns the Showdown connection.
        self.observation_encoder = HumanVisibleBattleEncoder.from_showdown(format_config, observation_config=self.observation_config)
        self.env = self.environment_class(team=self.team_designer,observation_encoder=self.observation_encoder,showdown_path=showdown_path,**env_kwargs)

        # Bound and pad the action space. Its natural size is fixed by the generation rather than
        # by the team, so this only ever pads when one policy has to span several formats.
        self.natural_action_space_size = SinglesEnv.get_action_space_size(format_config.generation)
        if max_action_space_size is None:
            max_action_space_size = self.natural_action_space_size
        if self.natural_action_space_size > max_action_space_size:
            raise ValueError(f"{format_config.format_name} requires an action space of {self.natural_action_space_size}, which exceeds max_action_space_size={max_action_space_size}")
        self.max_action_space_size = max_action_space_size
        self.action_space = spaces.Discrete(max_action_space_size)

        # The action mask travels inside the observation, so it has to be padded with the action
        # space. Padded entries are always illegal.
        inner_spaces = self.env.observation_space.spaces
        self.observation_space = spaces.Dict({**{name: space for name, space in inner_spaces.items() if name != "action_mask"},"action_mask": spaces.MultiBinary(max_action_space_size)})

        # Reset must run before step, which PokEnv enforces itself because the environment is not
        # wrapped in gymnasium's OrderEnforcing, which would otherwise reject the extended
        # reset(embodiment=) and step(embodiment=) signatures.
        self._started = False

        # Set by make() to the Showdown server this environment owns, so closing the environment
        # also shuts the server down.
        self.server_context = None

    def close(self) -> None:
        """Close the battle environment and shut down an owned Showdown server."""
        try:
            if self.env is not None:
                self.env.close()
        finally:
            if self.server_context is not None:
                server_context, self.server_context = self.server_context, None
                server_context.close()

    def _default_battle_format(self, env_kwargs: dict) -> str:
        """Return the battle format a preset uses when no FormatConfig is supplied."""
        if self.environment_class is not PokemonLeagueEnv:
            return self.environment_class.BATTLE_FORMAT
        if env_kwargs.get("random_league"):
            return PokemonLeagueEnv.RANDOM_BATTLE_FORMAT
        generation = env_kwargs.get("generation")
        if generation is None:
            raise ValueError("the league opponent requires either generation or random_league=True")
        return get_pokemon_league(generation).battle_format
