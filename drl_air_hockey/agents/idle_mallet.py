"""Intermittent, physically safe pauses for air-hockey mallets.

The wrapper independently samples brief pauses for each player. A pause holds
a policy-generated joint target with zero desired velocity.  This preserves a
valid, level end-effector target for the tournament's passive universal joint
and prevents a policy from planning far ahead while its action is suppressed.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


class IdleMalletAgent:
    """Wrap an agent with random, finite joint-position hold intervals.

    ``idle_probability`` is the chance to start a pause on an *unpaused*
    control step. A pause lasts uniformly from ``idle_min_steps`` through
    ``idle_max_steps`` inclusive. At 50 Hz, the default 5--25 steps is
    0.10--0.50 seconds. The wrapped policy is not advanced during a pause, so
    it resumes from the actual post-hold observation without a stale-command
    jump.
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
        self._held_action: Optional[np.ndarray] = None
        self._last_policy_action: Optional[np.ndarray] = None
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
        self._held_action = None
        self._last_policy_action = None
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

    def _consume_hold(self) -> np.ndarray:
        """Return one held action and account for one paused control step."""
        if self._pause_remaining:
            assert self._held_action is not None
            self._pause_remaining -= 1
            self._paused_steps += 1
            return self._held_action.copy()
        raise RuntimeError("attempted to consume an inactive mallet pause")

    def draw_action(self, obs: np.ndarray) -> np.ndarray:
        """Return a level-safe hold action, or advance the wrapped policy."""
        if self._pause_remaining:
            # Do not let the wrapped controller's internal trajectory run
            # ahead while its action is held. On resume it sees the actual
            # robot state and produces a continuous next command.
            return self._consume_hold()

        if self.idle_probability and (
            self.idle_probability == 1.0
            or np.random.random() < self.idle_probability
        ):
            # Hold the last *issued* policy target rather than the measured
            # joints. It is already IK-valid with the desired EE height and
            # level pose; zero desired velocity makes its endpoint stationary.
            # A first-ever pause has no issued target yet, so obtain one once.
            if self._last_policy_action is None:
                self._last_policy_action = np.asarray(
                    self.agent.draw_action(obs)
                ).copy()
            self._held_action = self._last_policy_action.copy()
            self._held_action[1] = 0.0
            self._pause_remaining = int(
                np.random.randint(self.idle_min_steps, self.idle_max_steps + 1)
            )
            self._pause_count += 1
            return self._consume_hold()

        policy_action = np.asarray(self.agent.draw_action(obs))
        self._last_policy_action = policy_action.copy()
        return policy_action

    def __getattr__(self, name: str) -> Any:
        """Preserve the wrapped agent API for framework-specific attributes."""
        return getattr(self.agent, name)
