import subprocess
from pathlib import Path

import numpy as np
from gymnasium import spaces
from poke_env.battle import AbstractBattle
from poke_env.environment import SingleAgentWrapper, SinglesEnv
from poke_env.teambuilder import Teambuilder

from pokemon_codesign_env.configs import FormatConfig, ObservationConfig
from pokemon_codesign_env.observations import HumanVisibleBattleEncoder
from pokemon_codesign_env.showdown import validate_team


class ShowdownRandomTeam(Teambuilder):
    """Generate random teams with Showdown and validate them for another format."""

    def __init__(self, source_format: str, target_format: str, showdown_path: Path, max_attempts: int = 25):
        super().__init__()
        self.source_format = source_format
        self.target_format = target_format
        self.showdown_path = showdown_path
        self.max_attempts = max_attempts

    def yield_team(self) -> str:
        last_error = None
        for _ in range(self.max_attempts):
            try:
                result = subprocess.run([str(self.showdown_path / "pokemon-showdown"), "generate-team", self.source_format], cwd=self.showdown_path, capture_output=True, check=False, text=True)
                if result.returncode:
                    message = result.stderr.strip() or result.stdout.strip()
                    raise RuntimeError(f"Pokemon Showdown could not generate a team: {message}")
                packed_team = result.stdout.strip()
                validate_team(self.target_format, packed_team, self.showdown_path)
                return packed_team
            except (RuntimeError, ValueError) as error:
                last_error = error
        raise ValueError(f"Could not generate a valid random opponent team: {last_error}") from last_error


class HumanVisibleSinglesEnv(SinglesEnv):
    """Singles environment using the configurable human-visible encoder."""

    def __init__(self, observation_encoder: HumanVisibleBattleEncoder, **kwargs):
        self.observation_encoder = observation_encoder
        super().__init__(**kwargs)
        self.observation_spaces = {agent: observation_encoder.observation_space for agent in self.possible_agents}

    def calc_reward(self, battle: AbstractBattle) -> float:
        return self.reward_computing_helper(battle, fainted_value=2.0, hp_value=1.0, victory_value=10.0, number_of_pokemons=max(1, battle.team_size))

    def embed_battle(self, battle: AbstractBattle) -> dict[str, np.ndarray]:
        return self.observation_encoder.encode(battle)


class BaselineOpponentEnv(SingleAgentWrapper):
    """Base for a Gymnasium environment preconfigured with a poke-env opponent."""

    BATTLE_FORMAT: str
    OPPONENT_TEAM_FORMAT: str
    OPPONENT_CLASS: type
    OBSERVATION_CONFIG = ObservationConfig()

    def __init__(self, team=None, battle_format: str | None = None, opponent_team_format: str | None = None, observation_config: ObservationConfig | None = None, observation_encoder: HumanVisibleBattleEncoder | None = None, showdown_path: Path | None = None, opponent_team=None, opponent_kwargs: dict | None = None, **env_kwargs):
        if team is None:
            raise ValueError("team must contain the co-design agent's designed team")
        battle_format = battle_format or self.BATTLE_FORMAT
        self.opponent_team_format = opponent_team_format or self.OPPONENT_TEAM_FORMAT
        observation_config = observation_config or self.OBSERVATION_CONFIG
        self.format_config = FormatConfig.from_showdown(battle_format, showdown_path)
        self.observation_encoder = observation_encoder or HumanVisibleBattleEncoder.from_showdown(self.format_config, observation_config=observation_config)
        if opponent_team is None:
            opponent_team = self.make_random_opponent_team()

        env_kwargs.setdefault("choose_on_teampreview", False)
        env_kwargs.setdefault("log_level", 40)
        env_kwargs.setdefault("open_timeout", None)
        env_kwargs.setdefault("strict", False)
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

    @staticmethod
    def _structured(observation):
        return {**observation["observation"], "action_mask": np.asarray(observation["action_mask"], dtype=np.int8)}

    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        return self._structured(observation), info

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        return self._structured(observation), reward, terminated, truncated, info
