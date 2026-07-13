"""Per-episode wrapper for occasionally stationary air-hockey mallets.

The tournament framework calls :meth:`episode_start` before the first
observation is available.  The idle decision is therefore deferred until the
first :meth:`draw_action`, where we can also capture the measured joint pose
to hold for the rest of that episode.  This is important for the data
collector: it seeds NumPy after the episode reset, so sampling here keeps the
decision deterministic per game.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


class IdleMalletAgent:
    """Wrap an Air Hockey Challenge agent with a per-episode idle option.

    When an episode is sampled as idle, the first measured joint pose is held
    with zero desired joint velocity.  Otherwise all calls are delegated to
    the wrapped agent unchanged.  The wrapper deliberately does not sample
    NumPy's RNG when ``idle_probability`` is exactly 0 or 1, preserving the
    wrapped policy's random-number stream in those common cases.
    """

    def __init__(self, agent: Any, idle_probability: float = 0.0):
        probability = float(idle_probability)
        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"idle_probability must be in [0, 1], got {idle_probability!r}"
            )
        if not hasattr(agent, "env_info"):
            raise TypeError("IdleMalletAgent requires an agent with env_info")

        self.agent = agent
        self.idle_probability = probability
        self._first_action = True
        self._is_idle = False
        self._held_joint_pos: Optional[np.ndarray] = None

    @property
    def preprocessors(self):
        """Use exactly the wrapped agent's observation preprocessors."""
        return self.agent.preprocessors

    @property
    def is_idle(self) -> bool:
        """Whether the current episode was sampled as stationary."""
        return self._is_idle

    def _start_episode(self) -> None:
        self._first_action = True
        self._is_idle = False
        self._held_joint_pos = None

    def reset(self) -> None:
        """Reset both wrapper state and the wrapped agent."""
        self.agent.reset()
        self._start_episode()

    def episode_start(self) -> None:
        """Forward the framework lifecycle event to the wrapped agent."""
        self.agent.episode_start()
        self._start_episode()

    def draw_action(self, obs: np.ndarray) -> np.ndarray:
        """Return the wrapped action, or a zero-velocity joint hold action."""
        if self._first_action:
            self._first_action = False
            if self.idle_probability == 0.0:
                self._is_idle = False
            elif self.idle_probability == 1.0:
                self._is_idle = True
            else:
                self._is_idle = bool(np.random.random() < self.idle_probability)

            if self._is_idle:
                joint_pos_ids = self.agent.env_info["joint_pos_ids"]
                self._held_joint_pos = np.asarray(obs[joint_pos_ids]).copy()

        if not self._is_idle:
            return self.agent.draw_action(obs)

        assert self._held_joint_pos is not None
        return np.vstack((self._held_joint_pos, np.zeros_like(self._held_joint_pos)))

    def __getattr__(self, name: str) -> Any:
        """Preserve the wrapped agent API for framework-specific attributes."""
        return getattr(self.agent, name)
