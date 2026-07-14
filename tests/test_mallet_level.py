import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from collect_2023 import (  # noqa: E402
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


def test_near_limit_level_solution_is_safely_clamped_but_large_tilt_is_rejected():
    ranges = np.array(((-1.5708, 1.5708), (-1.5708, 1.5708)))
    near_limit = np.array((-1.571033, 0.2))

    result = _clamp_near_limit_level_angles(near_limit, ranges)

    np.testing.assert_allclose(result, np.array((-1.5708, 0.2)))
    with pytest.raises(RuntimeError, match="max_excess"):
        _clamp_near_limit_level_angles(np.array((-1.572, 0.2)), ranges)


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
