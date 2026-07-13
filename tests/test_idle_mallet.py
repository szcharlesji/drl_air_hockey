import numpy as np
import pytest

from drl_air_hockey.agents.idle_mallet import IdleMalletAgent


class _FakeAgent:
    def __init__(self):
        self.env_info = {"joint_pos_ids": np.array([1, 3, 4])}
        self.preprocessors = [lambda obs: obs]
        self.draw_calls = 0
        self.episode_start_calls = 0
        self.reset_calls = 0

    def draw_action(self, obs):
        self.draw_calls += 1
        return np.full((2, 3), float(self.draw_calls))

    def episode_start(self):
        self.episode_start_calls += 1

    def reset(self):
        self.reset_calls += 1


def test_idle_mallet_holds_for_a_random_finite_pause_and_advances_policy(monkeypatch):
    agent = _FakeAgent()
    wrapper = IdleMalletAgent(
        agent, idle_probability=0.5, idle_min_steps=2, idle_max_steps=2
    )
    wrapper.episode_start()
    random_draws = iter((0.0, 1.0))  # start first pause, then resume
    monkeypatch.setattr(np.random, "random", lambda: next(random_draws))
    monkeypatch.setattr(np.random, "randint", lambda low, high: 2)

    first_obs = np.array([9.0, 1.0, 9.0, 2.0, 3.0])
    second_obs = np.array([9.0, 4.0, 9.0, 5.0, 6.0])
    third_obs = np.array([9.0, 7.0, 9.0, 8.0, 9.0])
    np.testing.assert_array_equal(
        wrapper.draw_action(first_obs), np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    )
    np.testing.assert_array_equal(
        wrapper.draw_action(second_obs), np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    )
    np.testing.assert_array_equal(
        wrapper.draw_action(third_obs), np.full((2, 3), 3.0)
    )
    assert not wrapper.is_idle
    assert wrapper.pause_stats == {
        "count": 1,
        "paused_steps": 2,
        "currently_paused": False,
    }
    assert agent.draw_calls == 3
    assert agent.episode_start_calls == 1


def test_zero_probability_is_a_passthrough_without_advancing_numpy_rng():
    agent = _FakeAgent()
    wrapper = IdleMalletAgent(agent, idle_probability=0.0)
    wrapper.episode_start()
    np.random.seed(123)
    expected_next_draw = np.random.random()
    np.random.seed(123)

    action = wrapper.draw_action(np.array([9.0, 1.0, 9.0, 2.0, 3.0]))
    assert np.random.random() == expected_next_draw
    np.testing.assert_array_equal(action, np.full((2, 3), 1.0))
    assert not wrapper.is_idle
    assert agent.draw_calls == 1


@pytest.mark.parametrize("probability", [-0.01, 1.01])
def test_idle_probability_must_be_a_probability(probability):
    with pytest.raises(ValueError, match="idle_probability"):
        IdleMalletAgent(_FakeAgent(), idle_probability=probability)


@pytest.mark.parametrize(
    "min_steps,max_steps", [(0, 1), (2, 1), (True, 2), (1.5, 2)]
)
def test_idle_pause_length_must_be_valid(min_steps, max_steps):
    with pytest.raises(ValueError, match="idle pause lengths"):
        IdleMalletAgent(
            _FakeAgent(), idle_min_steps=min_steps, idle_max_steps=max_steps
        )
