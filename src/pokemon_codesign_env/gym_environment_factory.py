import warnings

import gymnasium
from gymnasium import spaces
import numpy as np

from pathlib import Path

from poke_env.environment import SinglesEnv

from pokemon_codesign_env.configs import TeamSearchConfig, FormatConfig, ObservationConfig
from pokemon_codesign_env.envs import MaxBasePowerPlayerEnv,PokemonLeagueEnv,RandomPlayerEnv,SimpleHeuristicsPlayerEnv,get_pokemon_league
from pokemon_codesign_env.observations import HumanVisibleBattleEncoder
from pokemon_codesign_env.spaces import TeamDesignSpace
from pokemon_codesign_env.team_designer import BatchedTeamDesigner, TeamDesigner
from pokemon_codesign_env.team_pool import validate_team_batch

# The presets differ by opponent policy, not by format restriction: each is a Custom Game of its
# generation played against a random opponent team. A tiered format would instead reject part of
# the design space through its banlist and its move-compatibility rules, and Monotype would reject
# any team whose Pokemon do not all share a type.
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

    metadata = {"render_modes": ["html"]}
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
        team_attempts: int = 1,
        render_mode: str | None = None,
        **env_kwargs,
    ):
        super().__init__()

        if render_mode not in (None, "html"):
            raise ValueError(f"unsupported render_mode {render_mode!r}; choose None or 'html'")
        self.render_mode = render_mode
        self._replay_html = None

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
        # A retry only redraws the attributes the designer resolves randomly, so it helps a
        # configuration that leaves moves or EVs random and does nothing for one that designs
        # them: the same team is rebuilt and rejected again. Raising an illegal design promptly
        # is more useful than retrying it, so retries are off unless asked for.
        self.team_designer = TeamDesigner(embodiment_config, max_attempts=team_attempts)

        # The wrapped battle environment, which owns the Showdown connection.
        self.observation_encoder = HumanVisibleBattleEncoder.from_showdown(format_config, observation_config=self.observation_config)
        self.env = self.environment_class(team=self.team_designer,observation_encoder=self.observation_encoder,format_config=format_config,showdown_path=showdown_path,capture_replay_html=render_mode == "html",**env_kwargs)

        # Bound and pad the action space. max_action_space_size exists in the codesign API for
        # environments where designing a different body changes how many actions there are. This
        # one's size comes from the generation and never from the team, which always has six switch
        # slots, so padding here only serves a policy spanning formats of different generations.
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

        # The design this environment starts from. Sampling it once rather than at every reset
        # gives a known state to return to, and makes a reset that is given no design repeatable.
        self.base_embodiment = self.embodiment_space.sample() if self.designs_an_embodiment else {}

        # Set when a reset is given an embodiment that cannot be battled with, so that step can
        # end the episode instead of acting in a battle that was never started.
        self._embodiment_rejected = False

        # Set by make() to the Showdown server this environment owns, so closing the environment
        # also shuts the server down.
        self.server_context = None

    @property
    def embodiment(self) -> dict:
        """The embodiment currently installed, read back from the team compiler."""
        return self.team_designer.get_optimizable_parameter_values()

    @property
    def designs_an_embodiment(self) -> bool:
        """Whether this configuration leaves anything for an embodiment designer to choose."""
        return bool(self.embodiment_space.pokemon_keys)

    def set_embodiment(self, embodiment) -> None:
        """Validate an embodiment and install it as the team to compile from.

        Accepts either the parameter dictionary or a point of `embodiment_space.flat_space`, so a
        vector environment can send a row of indices to a worker and let it decode there rather
        than shipping the design space between processes.
        """
        if not isinstance(embodiment, dict):
            embodiment = self.embodiment_space.from_index(embodiment)
        if not self.embodiment_space.contains(embodiment):
            raise ValueError("embodiment is not a member of embodiment_space")
        self.team_designer.set_optimizable_parameter_values(embodiment)

    def validate_embodiment(self, embodiment=None) -> bool:
        """Whether a design can actually be battled with.

        Membership of `embodiment_space` does not settle this. The space judges each parameter on
        its own, so it cannot see that two moves are individually legal for a species but not
        learnable together; only Showdown can. `validate_embodiments` answers the same question for
        a whole population at a fraction of the cost, and reports why each was rejected.

        The environment's own design is left untouched.
        """
        return self.validate_embodiments([self.base_embodiment if embodiment is None else embodiment])[0] is None

    def _compile_embodiment(self, embodiment: dict | None = None) -> str:
        """Compile the embodiment into a packed team that Pokemon Showdown has accepted.

        `embodiment_space` judges each parameter on its own, so it cannot see that two moves are
        individually legal for a species but not learnable together: Showdown rejects Arbok with
        both Body Slam and Disable, because no single Arbok could have been on both acquisition
        paths. Only Showdown can settle that, and asking it here means the reason reaches the
        caller. Left to poke-env, the same failure is raised inside its battle thread and reaches
        the caller as nothing more informative than a challenge timeout.
        """
        if embodiment is not None:
            self.set_embodiment(embodiment)
        return self.team_designer.yield_team()

    def validate_embodiments(self, embodiments) -> list[str | None]:
        """Report which of a batch of embodiments can be battled with.

        Returns one entry per embodiment: None when Showdown accepts it, otherwise the reason it
        was rejected. A population-based designer screens its candidates together, and one
        long-lived Showdown validator answers a whole batch in about the time a single
        `validate_embodiment` call spends starting a process.

        The environment's own embodiment is left untouched, so this can be used to pre-screen
        candidates without disturbing an episode in progress.
        """
        reasons: list[str | None] = [None] * len(embodiments)
        designs, positions = [], []
        for position, embodiment in enumerate(embodiments):
            try:
                values = embodiment if isinstance(embodiment, dict) else self.embodiment_space.from_index(embodiment)
                if not self.embodiment_space.contains(values):
                    raise ValueError("embodiment is not a member of embodiment_space")
                designs.append(values)
                positions.append(position)
            except ValueError as error:
                reasons[position] = str(error)

        if designs:
            # One reusable compiler for the batch, seeded per design so the check is repeatable.
            compiled = BatchedTeamDesigner.serialize_parameter_values(
                self.embodiment_config, designs, list(range(len(designs)))
            )
            problems = validate_team_batch(self.format_config.showdown_path, self.format_config.format_id, compiled)
            for position, team_problems in zip(positions, problems, strict=True):
                if team_problems is not None:
                    reasons[position] = "; ".join(team_problems)
        return reasons

    def _rejected_observation(self) -> dict:
        """A zero observation for a reset whose embodiment could not be battled with.

        No battle exists, so there is nothing to encode. Zero is a member of every part of the
        observation, and the action mask leaves a single action legal so that a masked policy
        still forms a distribution over it.
        """
        observation = {}
        for name, space in self.observation_space.spaces.items():
            if isinstance(space, spaces.Discrete):
                observation[name] = 0
            else:
                observation[name] = np.zeros(space.shape, dtype=space.dtype)
        observation["action_mask"][0] = 1
        return observation

    def _pad_observation(self, observation: dict) -> dict:
        """Widen the action mask to the padded action space, marking padded actions illegal."""
        mask = np.asarray(observation["action_mask"], dtype=np.int8)
        if mask.size == self.max_action_space_size:
            return observation
        padded = np.zeros(self.max_action_space_size, dtype=np.int8)
        padded[: mask.size] = mask
        return {**observation, "action_mask": padded}

    def reset(self, *, seed=None, options=None, embodiment: dict | None = None):
        """Start a battle with `embodiment` as the designed team.

        :param embodiment: the design to battle with. Omitting it keeps the design already
            installed, or falls back to `base_embodiment` on the first reset.
        :returns: the observation, the info, and `embodiment_rejected`, one flag per environment.
            A design that cannot be battled with is reported rather than raised, because what a
            rejected candidate is worth is the caller's decision. Stepping afterwards raises.
        """
        super().reset(seed=seed)
        self._replay_html = None
        if seed is not None:
            # Both sides of the battle are drawn from this seed: the designer resolves every random
            # and defaulted attribute of the designed team, and the opponent's team comes from
            # Showdown's generator, which takes a seed of ours. The battle's own rolls are the
            # server's and are not seeded here, so a seed fixes the teams rather than the outcome.
            self.team_designer.rng = np.random.default_rng(seed)
            self.embodiment_space.seed(seed)
            if hasattr(self.env, "seed_opponent_team"):
                self.env.seed_opponent_team(seed)

        # A vector environment cannot pass the embodiment as an argument, because the vector reset
        # it wraps takes none, so it arrives in the options instead and is removed before the rest
        # of the options travel on to the battle.
        if options and "embodiment" in options:
            options = dict(options)
            embodiment = options.pop("embodiment") if embodiment is None else embodiment
            options = options or None

        # Omitting the embodiment keeps the one already installed, so an automatic reset after a
        # finished episode battles on with the same design. The first reset has nothing to keep and
        # falls back to the base design, which is a known state rather than a fresh random one.
        if embodiment is None and self.designs_an_embodiment and not self.embodiment:
            embodiment = self.base_embodiment

        # Compiling here rather than letting poke-env call the designer at battle start keeps the
        # rejection in this thread, where it can be reported. The battle then uses exactly the team
        # that Showdown accepted.
        try:
            if embodiment is not None:
                self.set_embodiment(embodiment)
            packed_team = self._compile_embodiment()
        except ValueError as error:
            # A design an optimizer proposed is data, not a programming error: report it and let
            # whoever is searching decide what a rejected candidate is worth.
            self._embodiment_rejected = True
            self._started = True
            info = {"embodiment_rejected": True, "embodiment_rejection_reason": str(error)}
            return self._rejected_observation(), info, np.array([True])

        self._embodiment_rejected = False
        self.env.set_learner_team(packed_team)

        observation, info = self.env.reset(seed=seed, options=options)
        self._started = True
        return self._pad_observation(observation), {**info, "embodiment_rejected": False}, np.array([False])

    def step(self, action=None, embodiment: dict | None = None):
        """Take one battle action.

        :param embodiment: ignored with a warning. The team is fixed once a battle begins, so this
            environment does not support transition-level embodiment changes; change a design at
            reset instead.
        """
        if not self._started:
            raise RuntimeError("reset() must be called before step()")
        if self._embodiment_rejected:
            raise RuntimeError("the embodiment given to reset was rejected, so no battle was started and there is nothing to step; reset with a design this environment accepts")
        if embodiment is not None:
            warnings.warn("this environment fixes the team for the duration of a battle, so the embodiment given to step is ignored; pass it to reset instead",
                stacklevel=2)
        if action is None:
            raise ValueError("step requires an action")

        # Nothing else checks the action: this environment is deliberately not wrapped in
        # gymnasium's PassiveEnvChecker, which would otherwise reject the extended signatures.
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"action {action} is outside {self.action_space}")
        if action >= self.natural_action_space_size:
            # Only reachable when max_action_space_size padded the space beyond what this format
            # uses. Those actions are the later generations' mechanics and are masked illegal in
            # every observation, so choosing one means the mask was ignored.
            raise ValueError(
                f"action {action} lies in the padded part of the action space, which is masked "
                f"illegal in every observation; {self.format_config.format_name} uses actions "
                f"0-{self.natural_action_space_size - 1}"
            )

        # poke-env reads the action as a NumPy scalar when deciding between a plain move and one
        # that mega evolves or terastallizes.
        observation, reward, terminated, truncated, info = self.env.step(np.int64(action))
        if "replay_html" in info:
            self._replay_html = info["replay_html"]
        # Gymnasium requires a reset once an episode is over, which PokEnv enforces itself because
        # it is not wrapped in OrderEnforcing.
        self._started = not (terminated or truncated)
        return self._pad_observation(observation), reward, terminated, truncated, info

    def render(self):
        """Return the most recent completed battle's HTML until the caller resets."""
        return self._replay_html if self.render_mode == "html" else None

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
        return self.default_battle_format(self.opponent, env_kwargs)

    @staticmethod
    def default_battle_format(opponent: str, env_kwargs: dict) -> str:
        """Return the battle format a preset uses when no FormatConfig is supplied."""
        if opponent not in OPPONENTS:
            raise ValueError(f"opponent must be one of {sorted(OPPONENTS)}, got {opponent!r}")
        environment_class = OPPONENTS[opponent]
        if environment_class is not PokemonLeagueEnv:
            return environment_class.BATTLE_FORMAT
        if env_kwargs.get("random_league"):
            return PokemonLeagueEnv.RANDOM_BATTLE_FORMAT
        generation = env_kwargs.get("generation")
        if generation is None:
            raise ValueError("the league opponent requires either generation or random_league=True")
        return get_pokemon_league(generation).battle_format
