"""Replay capture must survive both league and vector same-step resets."""

from types import SimpleNamespace

from gymnasium import spaces

from pokemon_codesign_env.envs.base import BaselineOpponentEnv
from pokemon_codesign_env.envs.pokemon_league_env import LeagueDifficulty, PokemonLeagueEnv
from pokemon_codesign_env.gym_environment_factory import PokEnv
from poke_env.environment import SingleAgentWrapper


def test_battle_captures_html_before_reset(monkeypatch):
    battle = SimpleNamespace(finished=True, _build_replay_html=lambda: "<!DOCTYPE html>replay")
    env = object.__new__(BaselineOpponentEnv)
    env.capture_replay_html = True
    env.env = SimpleNamespace(battle1=battle)
    monkeypatch.setattr(SingleAgentWrapper, "step", lambda self, action: ({}, 1.0, True, False, {}))
    monkeypatch.setattr(BaselineOpponentEnv, "_structured", staticmethod(lambda observation: observation))

    assert env.step(0)[4]["replay_html"] == "<!DOCTYPE html>replay"


def test_league_carries_replay_into_next_trainer():
    battle = SimpleNamespace(won=True)
    inner = SimpleNamespace(
        env=SimpleNamespace(battle1=battle),
        step=lambda action: ({}, 1.0, True, False, {"replay_html": "<!DOCTYPE html>replay"}),
        reset=lambda: ({}, {"next_battle": True}),
    )
    env = object.__new__(PokemonLeagueEnv)
    env._episode_active = True
    env._battle_env = inner
    env.league = SimpleNamespace(trainers=[SimpleNamespace(name="first"), SimpleNamespace(name="second")])
    env.current_trainer_index = 0
    env.difficulty = LeagueDifficulty.EASY
    env._base_team = "team"
    env._current_team = "team"
    env._prepare_battle = lambda team: None
    env._league_observation = lambda observation: observation
    env._league_info = lambda info, **updates: {**info, **updates}

    _, _, terminated, truncated, info = env.step(0)
    assert not terminated and not truncated
    assert info["replay_html"] == "<!DOCTYPE html>replay"
    assert info["next_battle"] is True


def test_pokenv_keeps_latest_replay_across_steps(monkeypatch):
    env = object.__new__(PokEnv)
    env.render_mode = "html"
    env._replay_html = "old replay"
    env._started = True
    env._embodiment_rejected = False
    env.action_space = spaces.Discrete(2)
    env.natural_action_space_size = 2
    env.env = SimpleNamespace(step=lambda action: ({}, 0.0, False, False, {}))
    monkeypatch.setattr(PokEnv, "_pad_observation", lambda self, observation: observation)

    env.step(0)
    assert env.render() == "old replay"
    env.env.step = lambda action: ({}, 1.0, True, False, {"replay_html": "new replay"})
    env.step(0)
    assert env.render() == "new replay"
