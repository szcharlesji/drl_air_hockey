import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import collect_2023  # noqa: E402
from collect_2023 import (  # noqa: E402
    HardMalletLevelGuard,
    _clamp_near_limit_level_angles,
    _scale_mallet_mesh_radially,
    _solve_universal_level_angles,
)


def _rot_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array(((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c)))


def _rot_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array(((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)))


def test_universal_level_solver_inverts_the_xml_y_then_x_joint_order():
    expected = np.array((0.43, -0.27))
    down_in_link = _rot_y(expected[0]) @ _rot_x(expected[1]) @ np.array((0.0, 0.0, 1.0))

    actual = _solve_universal_level_angles(down_in_link)

    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_level_solution_is_clamped_to_the_physical_universal_joint_range():
    ranges = np.array(((-1.5708, 1.5708), (-1.5708, 1.5708)))
    near_limit = np.array((-1.571033, 0.2))

    result = _clamp_near_limit_level_angles(near_limit, ranges)

    np.testing.assert_allclose(result, np.array((-1.5708, 0.2)))
    np.testing.assert_allclose(
        _clamp_near_limit_level_angles(np.array((-2.0, 2.0)), ranges),
        np.array((-1.5708, 1.5708)),
    )


def test_enlarging_a_mallet_mesh_preserves_its_axial_extent():
    vertices = np.array(
        ((-1.0, -2.0, -3.0), (1.0, -2.0, 5.0), (-1.0, 2.0, 5.0), (1.0, 2.0, -3.0))
    )
    original = vertices.copy()
    center = vertices.mean(axis=0)
    axis = np.array((1.0, 0.0, 0.0))

    _scale_mallet_mesh_radially(vertices, 2.5, axis)

    old_offsets = original - center
    parallel = np.outer(old_offsets @ axis, axis)
    radial = old_offsets - parallel
    np.testing.assert_allclose(vertices - center, parallel + 2.5 * radial)


def _scripted_guard(monkeypatch, floor_script):
    """A HardMalletLevelGuard whose simulator is a scripted replay.

    Every fake lift perturbs the arm coordinates so pose snapshots are
    distinguishable, and ``_mallet_floor_zs`` replays one scripted floor
    measurement per call.
    """
    guard = HardMalletLevelGuard.__new__(HardMalletLevelGuard)
    guard.model = None
    guard.data = SimpleNamespace(qpos=np.zeros(18))
    guard._arm_qpos_addrs = np.arange(14).reshape(2, 7)
    guard._qpos_addrs = np.arange(14, 18)
    floors = iter(floor_script)
    guard._mallet_floor_zs = lambda: np.asarray(next(floors), dtype=float)
    lifts = []

    def fake_lift(extra_height):
        lifts.append(np.array(extra_height, dtype=float))
        guard.data.qpos[:14] += 1.0
        return np.zeros(2)

    guard._project_arm_height = fake_lift
    guard._set_level_angles = lambda refresh_kinematics: None
    monkeypatch.setattr(collect_2023.mujoco, "mj_fwdPosition", lambda *args: None)
    return guard, lifts


def test_floor_correction_skips_mallets_already_above_the_table(monkeypatch):
    guard, lifts = _scripted_guard(monkeypatch, [[2e-4, 1e-3]])

    guard._correct_floor_penetration()

    assert lifts == []
    np.testing.assert_array_equal(guard.data.qpos, np.zeros(18))


def test_floor_correction_converges_in_one_round_at_ordinary_poses(monkeypatch):
    guard, lifts = _scripted_guard(
        monkeypatch, [[-0.0158, -0.0158], [2e-4, 2e-4]]
    )

    guard._correct_floor_penetration()

    assert len(lifts) == 1
    np.testing.assert_allclose(lifts[0], [0.016, 0.016])
    np.testing.assert_array_equal(guard.data.qpos[:14], np.ones(14))


def test_divergent_lift_settles_on_the_best_pose_per_mallet(monkeypatch):
    # Lifting rescues mallet 2 but mallet 1 only worsens after its first
    # round: the loop must stop, restore mallet 1's best pose, and accept a
    # sub-visual residual instead of aborting the worker.
    guard, lifts = _scripted_guard(
        monkeypatch,
        [
            [-0.0158, -0.0158],
            [-2.28e-4, 2e-4],
            [-4e-4, 2e-4],
            [-2.28e-4, 2e-4],
        ],
    )

    guard._correct_floor_penetration()

    assert len(lifts) == 2
    # Mallet 2 reached clearance after round one, so only mallet 1's
    # cumulative lift may grow in round two.
    assert lifts[1][0] > lifts[0][0]
    assert lifts[1][1] == lifts[0][1]
    # Mallet 1's arm returns to its best (round-one) pose; mallet 2 keeps
    # its latest, equally good pose.
    np.testing.assert_array_equal(guard.data.qpos[0:7], np.ones(7))
    np.testing.assert_array_equal(guard.data.qpos[7:14], np.full(7, 2.0))


def test_visible_penetration_still_aborts_the_worker(monkeypatch):
    guard, _ = _scripted_guard(
        monkeypatch, [[-0.01, -0.01], [-0.01, -0.01]]
    )

    with pytest.raises(RuntimeError, match="sinks visibly"):
        guard._correct_floor_penetration()
