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
        return np.full((2, 3), 7.0)

    def episode_start(self):
        self.episode_start_calls += 1

    def reset(self):
        self.reset_calls += 1


def test_idle_mallet_holds_first_measured_joint_pose_for_full_episode():
    agent = _FakeAgent()
    wrapper = IdleMalletAgent(agent, idle_probability=1.0)
    wrapper.episode_start()

    first_obs = np.array([9.0, 1.0, 9.0, 2.0, 3.0])
    second_obs = np.array([9.0, 4.0, 9.0, 5.0, 6.0])
    np.testing.assert_array_equal(
        wrapper.draw_action(first_obs), np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    )
    np.testing.assert_array_equal(
        wrapper.draw_action(second_obs), np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    )
    assert wrapper.is_idle
    assert agent.draw_calls == 0
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
    np.testing.assert_array_equal(action, np.full((2, 3), 7.0))
    assert not wrapper.is_idle
    assert agent.draw_calls == 1


@pytest.mark.parametrize("probability", [-0.01, 1.01])
def test_idle_probability_must_be_a_probability(probability):
    with pytest.raises(ValueError, match="idle_probability"):
        IdleMalletAgent(_FakeAgent(), idle_probability=probability)
