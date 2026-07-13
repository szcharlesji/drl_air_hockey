"""Intermittent, physically stationary pauses for air-hockey mallets.

The wrapper independently samples brief pauses for each player. During a
pause it holds the measured joint pose with zero desired velocity, while still
calling the wrapped policy every control step so recurrent state, waypoint
logic, and random-number streams keep advancing naturally.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


class IdleMalletAgent:
    """Wrap an agent with random, finite joint-position hold intervals.

    ``idle_probability`` is the chance to start a pause on an *unpaused*
    control step. A pause lasts uniformly from ``idle_min_steps`` through
    ``idle_max_steps`` inclusive. At 50 Hz, the default 5--25 steps is
    0.10--0.50 seconds.
    """

    def __init__(
        self,
        agent: Any,
        idle_probability: float = 0.0,
        idle_min_steps: int = 5,
        idle_max_steps: int = 25,
    ):
        probability = float(idle_probability)
        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"idle_probability must be in [0, 1], got {idle_probability!r}"
            )
        if (
            isinstance(idle_min_steps, bool)
            or isinstance(idle_max_steps, bool)
            or not isinstance(idle_min_steps, (int, np.integer))
            or not isinstance(idle_max_steps, (int, np.integer))
        ):
            raise ValueError("idle pause lengths must be positive integers")
        min_steps = int(idle_min_steps)
        max_steps = int(idle_max_steps)
        if min_steps <= 0 or max_steps <= 0 or min_steps > max_steps:
            raise ValueError(
                "idle pause lengths must satisfy 0 < idle_min_steps <= idle_max_steps"
            )
        if not hasattr(agent, "env_info"):
            raise TypeError("IdleMalletAgent requires an agent with env_info")

        self.agent = agent
        self.idle_probability = probability
        self.idle_min_steps = min_steps
        self.idle_max_steps = max_steps
        self._pause_remaining = 0
        self._held_joint_pos: Optional[np.ndarray] = None
        self._pause_count = 0
        self._paused_steps = 0

    @property
    def preprocessors(self):
        """Use exactly the wrapped agent's observation preprocessors."""
        return self.agent.preprocessors

    @property
    def is_idle(self) -> bool:
        """Whether this mallet is currently inside a sampled pause."""
        return self._pause_remaining > 0

    @property
    def pause_stats(self):
        """Per-episode pause accounting for collection metadata."""
        return {
            "count": self._pause_count,
            "paused_steps": self._paused_steps,
            "currently_paused": self.is_idle,
        }

    def _start_episode(self) -> None:
        self._pause_remaining = 0
        self._held_joint_pos = None
        self._pause_count = 0
        self._paused_steps = 0

    def reset(self) -> None:
        """Reset both wrapper state and the wrapped agent."""
        self.agent.reset()
        self._start_episode()

    def episode_start(self) -> None:
        """Forward the framework lifecycle event to the wrapped agent."""
        self.agent.episode_start()
        self._start_episode()

    def draw_action(self, obs: np.ndarray) -> np.ndarray:
        """Advance the policy, optionally replacing its action with a hold."""
        # Keep policy state coherent even while the externally applied action
        # is a hold. This matters for recurrent tournament policies and the
        # smooth-random agent's waypoint clock.
        policy_action = self.agent.draw_action(obs)

        if self._pause_remaining == 0 and self.idle_probability:
            if self.idle_probability == 1.0 or np.random.random() < self.idle_probability:
                joint_pos_ids = self.agent.env_info["joint_pos_ids"]
                self._held_joint_pos = np.asarray(obs[joint_pos_ids]).copy()
                self._pause_remaining = int(
                    np.random.randint(self.idle_min_steps, self.idle_max_steps + 1)
                )
                self._pause_count += 1

        if self._pause_remaining:
            assert self._held_joint_pos is not None
            self._pause_remaining -= 1
            self._paused_steps += 1
            return np.vstack((self._held_joint_pos, np.zeros_like(self._held_joint_pos)))
        return policy_action

    def __getattr__(self, name: str) -> Any:
        """Preserve the wrapped agent API for framework-specific attributes."""
        return getattr(self.agent, name)
