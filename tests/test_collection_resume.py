import json
import sys
from pathlib import Path

import pytest


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import collect_sa_dataset as launcher  # noqa: E402
import convert_npz_to_sa_h5 as converter  # noqa: E402


def _complete_game(path, seed):
    path.mkdir(parents=True)
    (path / "episode_000.npz").touch()
    (path / "meta.json").write_text(
        json.dumps({"n_episodes": 1, "seed": seed}), encoding="utf-8"
    )


def _config(raw_dir, h5_dir, games=4, workers=2, collector_args=""):
    return f"""
version: 1
dataset:
  raw_dir: {raw_dir}
  h5_dir: {h5_dir}
collection:
  conda_env: test
  platform: cpu
  steps: 1000
  width: 128
  height: 128
  fps: 50
  puck_radius: 0.1
  mallet_radius: 0.1
  robot_visual_scale: 2.5
  {collector_args}
sources:
  - name: aggressive
    model1: tournament_aggressive
    model2: tournament_aggressive
    games: {games}
    workers: {workers}
conversion: {{}}
"""


def test_completed_games_ignore_interrupted_directories_and_advance_seed(tmp_path):
    source = tmp_path / "aggressive"
    incomplete = source / "collect-interrupted" / "game_000"
    incomplete.mkdir(parents=True)
    (incomplete / "episode_000.npz").touch()
    _complete_game(source / "collect-a" / "game_000", seed=4)
    _complete_game(source / "game_001", seed=9)

    completed = launcher._completed_games(source)

    assert [path.name for path, _ in completed] == ["game_000", "game_001"]
    assert launcher._next_seed({"seed_base": 0}, completed) == 10
    assert converter.discover_games(source) == [path for path, _ in completed]


def test_content_hash_allows_target_and_worker_changes(tmp_path):
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    raw_dir = tmp_path / "raw"
    h5_dir = tmp_path / "h5"
    first.write_text(_config(raw_dir, h5_dir, games=4, workers=2), encoding="utf-8")
    second.write_text(_config(raw_dir, h5_dir, games=10, workers=8), encoding="utf-8")

    first_config, _, _ = launcher.load_config(first)
    second_config, _, _ = launcher.load_config(second)

    assert launcher._collection_content_hash(first_config) == launcher._collection_content_hash(
        second_config
    )


def test_config_rejects_collector_args_that_override_dataset_fields(tmp_path):
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        _config(tmp_path / "raw", tmp_path / "h5", collector_args="collector_args: {games: 1}"),
        encoding="utf-8",
    )

    try:
        launcher.load_config(config_path)
    except launcher.ConfigError as exc:
        assert "overrides a required dataset setting" in str(exc)
    else:
        raise AssertionError("reserved collector_args key should be rejected")


def test_collection_lock_prevents_overlapping_seed_allocation(tmp_path):
    first = launcher.CollectionLock(tmp_path / "raw", "first")
    second = launcher.CollectionLock(tmp_path / "raw", "second")

    first.acquire()
    try:
        with pytest.raises(RuntimeError, match="collection lock already exists"):
            second.acquire()
    finally:
        first.release()

    second.acquire()
    second.release()
