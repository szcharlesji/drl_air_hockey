import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from collect_2023 import FixedLengthEventController  # noqa: E402


class _FakeTournament:
    def __init__(self, puck_pos, puck_vel=(0.0, 0.0, 0.0)):
        self.puck_pos = np.asarray(puck_pos, dtype=float)
        self.puck_vel = np.asarray(puck_vel, dtype=float)
        self.prev_side = 1
        self.timer = 0.0
        self.dt = 0.02
        self.faults = [0, 0]
        self.score = [0, 0]
        self.start_side = 0
        self.env_info = {"table": {"length": 2.0, "width": 1.0, "goal_width": 0.25}}

    def get_puck(self, _obs):
        return self.puck_pos, self.puck_vel

    def is_absorbing(self, _obs):
        return True


def _controller(puck_pos, puck_vel=(0.0, 0.0, 0.0)):
    base_env = _FakeTournament(puck_pos, puck_vel)
    return FixedLengthEventController(SimpleNamespace(base_env=base_env))


def test_edge_event_stays_visible_and_is_not_queued_for_hiding():
    # Beyond the end boundary but outside the goal mouth: a table-edge escape,
    # not an actual score.
    controller = _controller((1.02, 0.30), (0.1, 0.0, 0.0))
    controller.step_index = 4

    assert controller.is_absorbing(None) is False
    assert controller.events == [{"kind": "escape_or_invalid_speed", "step": 4}]
    assert controller.pending_goal is None
    controller.hide_pending_puck()
    assert not controller.puck_hidden

    # Remaining at the edge does not add repeated events or change visibility.
    controller.step_index = 5
    assert controller.is_absorbing(None) is False
    assert len(controller.events) == 1
    assert not controller.puck_hidden


def test_only_an_actual_goal_is_queued_for_hiding():
    controller = _controller((1.02, 0.0), (0.1, 0.0, 0.0))
    controller.step_index = 9

    assert controller.is_absorbing(None) is False
    assert controller.events == [{"kind": "goal_player_1", "step": 9}]
    assert controller.pending_goal == controller.events[0]
    assert not controller.puck_hidden
