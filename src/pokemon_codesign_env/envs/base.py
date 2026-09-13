import asyncio
import itertools
import logging
import os
from collections import deque
from pathlib import Path

import numpy as np
from gymnasium import spaces
from poke_env.battle import AbstractBattle
from poke_env.data import GenData
from poke_env.environment import SingleAgentWrapper, SinglesEnv
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.teambuilder import Teambuilder

from pokemon_codesign_env.configs import FormatConfig, ObservationConfig
from pokemon_codesign_env.observations import HumanVisibleBattleEncoder
from pokemon_codesign_env.protocol import install_gen4_protocol_adapter
from pokemon_codesign_env.team_pool import generate_team_batch, validate_team_batch
from pokemon_codesign_env.showdown import _poke_env_can_represent


class _UnexpectedEffectWarningFilter(logging.Filter):
    """Suppress poke-env's repetitive unknown-effect compatibility warning."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        known_effect = any(f"'{effect}'" in message for effect in ("BEAT_UP", "MUD_SPORT", "WATER_SPORT"))
        return not (
            record.name == "poke-env" and known_effect and message.startswith("Unexpected effect '")
            and "Effect.UNKNOWN will be used instead." in message
        )


logging.getLogger("poke-env").addFilter(_UnexpectedEffectWarningFilter())


_ACCOUNT_INDICES = itertools.count()


def _unique_account() -> AccountConfiguration:
    """A Showdown login no other client alive on this machine can be using.

    Showdown caps a username at 18 characters, which this stays well inside.
    """
    return AccountConfiguration(f"pce{os.getpid():x}-{next(_ACCOUNT_INDICES):x}", None)


class ShowdownRandomTeam(Teambuilder):
    """Generate random teams with Showdown and validate them for another format.

    Showdown draws each team from its own generator, so the opponent would vary from run to run
    however the environment was seeded. Handing it a seed of ours instead makes the sequence of
    opponents reproducible, which a battle needs as much as the designed team does.
    """

    def __init__(self, source_format: str, target_format: str, showdown_path: Path, max_attempts: int = 25, seed=None, batch_size: int = 64):
        super().__init__()
        self.source_format = source_format
        self.target_format = target_format
        self.generation = GenData.from_format(target_format).gen
        self.showdown_path = showdown_path
        self.max_attempts = max_attempts
        self.batch_size = max(1, batch_size)
        self.rng = np.random.default_rng(seed)
        self._ready = deque()

    def seed(self, seed) -> None:
        """Draw every following opponent team from a stream starting at this seed."""
        self.rng = np.random.default_rng(seed)
        # Teams already drawn belong to the previous stream and must not outlive it.
        self._ready.clear()

    def _next_seed(self) -> str:
        """Showdown takes its seed as four 16-bit numbers."""
        return ",".join(str(int(value)) for value in self.rng.integers(0, 65536, size=4))

    def yield_team(self) -> str:
        if not self._ready:
            self._refill()
        return self._ready.popleft()

    def _refill(self) -> None:
        """Draw a batch of teams from the seed stream, keeping the ones the target format accepts.

        A team used to cost two Node processes, one to generate it and one to validate it, which
        was most of what resetting an environment spent. Both steps now run in a process that
        stays alive and handles a whole batch, so the cost is amortised over batch_size battles.
        Seeds are still drawn one per team in the same order and valid teams are handed out in
        that order, so a given seed produces exactly the sequence of opponents it did before.
        """
        last_error = None
        for _ in range(self.max_attempts):
            # A fresh seed per team, so a retry after a rejected team draws a different one rather
            # than reproducing the same rejection.
            seeds = [self._next_seed() for _ in range(self.batch_size)]
            try:
                candidates = generate_team_batch(self.showdown_path, self.source_format, seeds)
                problems = validate_team_batch(self.showdown_path, self.target_format, candidates)
            except (RuntimeError, ValueError) as error:
                last_error = error
                continue
            valid = [team for team, issue in zip(candidates, problems, strict=True) if issue is None
                and all(_poke_env_can_represent(mon.species or mon.nickname, self.generation) for mon in self.parse_packed_team(team))]
            if valid:
                self._ready.extend(valid)
                return
            last_error = ValueError("Pokemon Showdown rejected every generated team")
        raise ValueError(f"Could not generate a valid random opponent team: {last_error}") from last_error


class HumanVisibleSinglesEnv(SinglesEnv):
    """Singles environment using the configurable human-visible encoder."""

    def __init__(self, observation_encoder: HumanVisibleBattleEncoder, **kwargs):
        self.observation_encoder = observation_encoder
        super().__init__(**kwargs)
        if GenData.from_format(kwargs.get("battle_format", "gen8randombattle")).gen == 4:
            install_gen4_protocol_adapter(self.agent1)
            install_gen4_protocol_adapter(self.agent2)
        self.observation_spaces = {agent: observation_encoder.observation_space for agent in self.possible_agents}

    def calc_reward(self, battle: AbstractBattle) -> float:
        return self.reward_computing_helper(battle, fainted_value=2.0, hp_value=1.0, victory_value=10.0, number_of_pokemons=max(1, battle.team_size))

    def embed_battle(self, battle: AbstractBattle) -> dict[str, np.ndarray]:
        return self.observation_encoder.encode(battle)

    @staticmethod
    def order_to_action(order, battle, fake: bool = False, strict: bool = True) -> np.int64:
        """Convert opponent orders without poke-env's unbounded recursive fallback."""
        try:
            return SinglesEnv.order_to_action(order, battle, fake=fake, strict=True)
        except (AssertionError, ValueError):
            if strict:
                raise
            legal_actions = np.flatnonzero(SinglesEnv.get_action_mask(battle))
            return np.int64(legal_actions[0] if legal_actions.size else 0)

    def close(self, force: bool = True, wait: bool = True):
        super().close(force=force, wait=wait)
        for agent in (self.agent1, self.agent2):
            close_task = asyncio.run_coroutine_threadsafe(self._shutdown_client(agent.ps_client), agent.ps_client.loop)
            try:
                close_task.result(timeout=10)
            except Exception:
                close_task.cancel()

    @staticmethod
    async def _shutdown_client(ps_client) -> None:
        """Stop a client's listener and settle its in-flight message handlers.

        listen() dispatches each websocket message into its own task, so closing only the socket
        leaves those handlers pending. When the event loop is then torn down they are garbage
        collected mid-await, which poke-env reports as a wall of "Exception ignored in" traces at
        the end of a large run. Cancelling them and awaiting the result lets the loop close with
        nothing still running against it. Battles are already over by the time close is called, so
        no result is lost by draining them.
        """
        await ps_client._stop_listening()
        pending = [task for task in getattr(ps_client, "_active_tasks", ()) if not task.done()]
        if not pending:
            return
        # A cancelled handler logs its own CancelledError at CRITICAL, so a client being discarded
        # would still print one line per in-flight message. Its logs are of no further use once it
        # is shutting down, so they are turned off for the drain.
        ps_client.logger.disabled = True
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)


class BaselineOpponentEnv(SingleAgentWrapper):
    """Base for a Gymnasium environment preconfigured with a poke-env opponent."""

    BATTLE_FORMAT: str
    OPPONENT_TEAM_FORMAT: str
    OPPONENT_CLASS: type
    OBSERVATION_CONFIG = ObservationConfig()

    def __init__(
        self, team=None, battle_format: str | None = None, opponent_team_format: str | None = None,
        observation_config: ObservationConfig | None = None, observation_encoder: HumanVisibleBattleEncoder | None = None,
        format_config: FormatConfig | None = None, showdown_path: Path | None = None, opponent_team=None, opponent_kwargs: dict | None = None, **env_kwargs,
    ):
        if team is None:
            raise ValueError("team must contain the co-design agent's designed team")
        battle_format = battle_format or self.BATTLE_FORMAT
        self.opponent_team_format = opponent_team_format or self.OPPONENT_TEAM_FORMAT
        observation_config = observation_config or self.OBSERVATION_CONFIG
        self.format_config = format_config or FormatConfig.from_showdown(battle_format, showdown_path)
        self.observation_encoder = observation_encoder or HumanVisibleBattleEncoder.from_showdown(self.format_config, observation_config=observation_config)
        if opponent_team is None:
            opponent_team = self.make_random_opponent_team()
        self.opponent_team = opponent_team

        # Showdown rejects a login whose name another live client already holds. poke-env's
        # generated names are a random five character suffix, and 8192 environments need 16384 of
        # them, which collides with about 90% probability; its non-random alternative counts from
        # one inside each process, so 256 workers all propose the same names. A name only has to
        # be unique among the clients currently connected, which the owning pid and a counter
        # within that process guarantee outright.
        env_kwargs.setdefault("account_configuration1", _unique_account())
        env_kwargs.setdefault("account_configuration2", _unique_account())
        env_kwargs.setdefault("choose_on_teampreview", False)
        env_kwargs.setdefault("log_level", 40)
        env_kwargs.setdefault("open_timeout", None)
        env_kwargs.setdefault("strict", False)
        # poke-env pings every 20 seconds and closes the battle if a pong takes another 20, which
        # suits a public server reached over a network. These servers are local and owned by this
        # process, and past a few thousand environments both ends are oversubscribed: a shard is
        # single-threaded and a worker packs many battles onto one event loop. A pong then arrives
        # late routinely and tears down a healthy battle with "keepalive ping timeout". A server
        # that has genuinely died is still caught, by the vector environment's step timeout and
        # worker restart rather than by a keepalive.
        env_kwargs.setdefault("ping_interval", 60.0)
        env_kwargs.setdefault("ping_timeout", 300.0)
        battle_env = HumanVisibleSinglesEnv(observation_encoder=self.observation_encoder, battle_format=battle_format, team=team, **env_kwargs)
        battle_env.agent2.update_team(opponent_team)

        opponent_kwargs = dict(opponent_kwargs or {})
        opponent_kwargs.setdefault("battle_format", battle_format)
        opponent_kwargs.setdefault("start_listening", False)
        opponent = self.OPPONENT_CLASS(**opponent_kwargs)
        super().__init__(battle_env, opponent)

        state_space = self.observation_space["observation"]
        mask_size = self.observation_space["action_mask"].shape[0]
        self.observation_space = spaces.Dict({**state_space.spaces, "action_mask": spaces.MultiBinary(mask_size)})

    def make_random_opponent_team(self) -> ShowdownRandomTeam:
        return ShowdownRandomTeam(self.opponent_team_format, self.format_config.format_id, self.format_config.showdown_path)

    def seed_opponent_team(self, seed) -> None:
        """Make the generated opponent teams reproducible, when they are generated at all.

        A supplied opponent team, or one drawn from a pool, is left alone.
        """
        if isinstance(self.opponent_team, ShowdownRandomTeam):
            self.opponent_team.seed(seed)

    def set_learner_team(self, team: str) -> None:
        """Use a new learner embodiment from the next battle onward."""
        self.team = team
        self.env.agent1.update_team(team)

    @staticmethod
    def _structured(observation):
        action_mask = np.asarray(observation["action_mask"], dtype=np.int8).copy()
        if not action_mask.any():
            action_mask[0] = 1
        return {**observation["observation"], "action_mask": action_mask}

    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        return self._structured(observation), info

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        return self._structured(observation), reward, terminated, truncated, info
