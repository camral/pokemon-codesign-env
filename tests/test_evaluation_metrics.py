from pokemon_codesign_env.evaluation import IsolatedEvaluationResult


def test_evaluation_result_summarizes_known_battle_statistics():
    result = IsolatedEvaluationResult(
        mean_return=2.0,
        episode_returns=(1.0, 3.0),
        timed_out_episodes=0,
        episode_lengths=(5, 7),
        wins=(True, False),
        turn_counts=(4, 6),
    )

    assert result.win_rate == 0.5
    assert result.mean_episode_length == 6
    assert result.mean_turn_count == 5
