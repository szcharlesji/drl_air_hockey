#!/usr/bin/env python3
"""Collect a YAML-described air-hockey dataset and convert it to SA H5.

The low-level collector deliberately remains a small, direct CLI:

    python scripts/collect_2023.py ...

This launcher is the reproducible dataset-level layer. It resumes each source
to its configured *target* number of completed games, chooses fresh seeds,
snapshots the exact YAML plus stable config hashes, and then rebuilds the H5
dataset from all completed raw games.

Typical use:

    conda run -n airhockey2023 python scripts/collect_sa_dataset.py \\
        --config configs/data/airhockey-v6.yaml

Use ``--dry-run`` to validate a config and print the commands without
creating directories or launching collection.
"""

import argparse
import datetime as dt
import hashlib
import json
import os
import shlex
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised only in missing envs
    raise SystemExit(
        "PyYAML is required for YAML dataset configs. Install it with "
        "`conda install -n airhockey2023 pyyaml` or `pip install PyYAML`."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "data" / "airhockey-v6.yaml"

# Core settings are expressed structurally in YAML and must not be overridden
# later through collector_args (argparse would otherwise silently use the last
# occurrence). The remaining flags are explicit, safe extensions.
RESERVED_COLLECTOR_OPTION_KEYS = frozenset({
    "platform", "workers", "model1", "model2", "games", "steps",
    "puck-radius", "mallet-radius", "robot-visual-scale", "width", "height",
    "fps", "seed", "out", "run-dir", "game-start",
})
ALLOWED_COLLECTOR_OPTION_KEYS = frozenset({
    "episode-mode", "orientation-marker-arm-length",
    "orientation-marker-stroke-width", "idle-prob1", "idle-prob2",
    "idle-min-steps", "idle-max-steps", "post-goal-policy",
    "mallet-level-lock",
    "keep-scoreboard", "shadows", "gpu",
})


class ConfigError(ValueError):
    """Raised when a dataset config is structurally invalid."""


def _as_mapping(value: Any, label: str) -> MutableMapping[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{label} must be a mapping")
    return value


def _require(mapping: Mapping[str, Any], key: str, label: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"missing required key: {label}.{key}")
    return mapping[key]


def _validate_collector_options(value: Any, label: str) -> None:
    """Allow only non-core collect_2023.py options in a YAML extension map."""
    options = _as_mapping(value, label)
    for key in options:
        if not isinstance(key, str) or not key or key.startswith("-"):
            raise ConfigError(f"{label} keys must be bare option names")
        normalised = key.replace("_", "-")
        if normalised in RESERVED_COLLECTOR_OPTION_KEYS:
            raise ConfigError(
                f"{label}.{key} overrides a required dataset setting; "
                "set it in the structured YAML field instead"
            )
        if normalised not in ALLOWED_COLLECTOR_OPTION_KEYS:
            raise ConfigError(f"{label}.{key} is not a supported collector extension")


def _positive_int(value: Any, label: str) -> int:
    # ``bool`` is an ``int`` subclass, but it is never a meaningful count.
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{label} must be a positive integer")
    return value


def _positive_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ConfigError(f"{label} must be a positive number")
    return float(value)


def _path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{label} must be a non-empty path string")
    return Path(os.path.expandvars(os.path.expanduser(value))).resolve()


def _normalise_sources(value: Any) -> List[MutableMapping[str, Any]]:
    """Accept a list of source maps, keeping source order part of the config."""
    if not isinstance(value, list) or not value:
        raise ConfigError("sources must be a non-empty list")
    sources = []
    names = set()
    for index, source in enumerate(value):
        source = _as_mapping(source, f"sources[{index}]")
        name = _require(source, "name", f"sources[{index}]")
        if not isinstance(name, str) or not name or "/" in name or "\\" in name:
            raise ConfigError(
                f"sources[{index}].name must be a non-empty simple directory name"
            )
        if name in names:
            raise ConfigError(f"duplicate source name: {name}")
        names.add(name)
        for key in ("model1", "model2"):
            model = _require(source, key, f"sources[{index}]")
            if not isinstance(model, str) or not model:
                raise ConfigError(f"sources[{index}].{key} must be a non-empty string")
        _positive_int(_require(source, "games", f"sources[{index}]"),
                      f"sources[{index}].games")
        _positive_int(_require(source, "workers", f"sources[{index}]"),
                      f"sources[{index}].workers")
        if "seed_base" in source:
            seed_base = source["seed_base"]
            if isinstance(seed_base, bool) or not isinstance(seed_base, int):
                raise ConfigError(f"sources[{index}].seed_base must be an integer")
        if "collector_args" in source:
            _validate_collector_options(
                source["collector_args"], f"sources[{index}].collector_args"
            )
        sources.append(source)
    return sources


def _validate_mixture(
    value: Any, sources: Iterable[Mapping[str, Any]], total_games: int
) -> None:
    """Validate optional declared source fractions against requested games."""
    if value is None:
        return
    mixture = _as_mapping(value, "mixture")
    fractions = _as_mapping(_require(mixture, "fractions", "mixture"), "mixture.fractions")
    source_by_name = {source["name"]: source for source in sources}
    if set(fractions) != set(source_by_name):
        raise ConfigError(
            "mixture.fractions keys must exactly match source names "
            f"({sorted(source_by_name)})"
        )
    expected_sum = 0.0
    for name, fraction in fractions.items():
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
            raise ConfigError(f"mixture.fractions.{name} must be a number")
        fraction = float(fraction)
        if fraction < 0.0 or fraction > 1.0:
            raise ConfigError(f"mixture.fractions.{name} must be in [0, 1]")
        expected_sum += fraction
        actual = source_by_name[name]["games"] / total_games
        # Exact ratios are preferred, but a one-game rounding error is valid
        # for small smoke-test configs.
        if abs(actual - fraction) > (1.0 / total_games + 1e-12):
            raise ConfigError(
                f"source {name!r} has {actual:.4%} of games, not declared "
                f"{fraction:.4%}"
            )
    if abs(expected_sum - 1.0) > 1e-9:
        raise ConfigError("mixture.fractions must sum to 1.0")


def load_config(config_path: Path) -> Tuple[MutableMapping[str, Any], str, str]:
    """Load, validate, and return (config, original YAML, canonical hash)."""
    if not config_path.is_file():
        raise ConfigError(f"config not found: {config_path}")
    raw_text = config_path.read_text(encoding="utf-8")
    try:
        config = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {config_path}: {exc}") from exc
    config = _as_mapping(config, "config")

    if config.get("version") != 1:
        raise ConfigError("config.version must be 1")
    dataset = _as_mapping(_require(config, "dataset", "config"), "dataset")
    collection = _as_mapping(_require(config, "collection", "config"), "collection")
    conversion = _as_mapping(_require(config, "conversion", "config"), "conversion")
    sources = _normalise_sources(_require(config, "sources", "config"))

    _path(_require(dataset, "raw_dir", "dataset"), "dataset.raw_dir")
    _path(_require(dataset, "h5_dir", "dataset"), "dataset.h5_dir")
    env = _require(collection, "conda_env", "collection")
    if not isinstance(env, str) or not env:
        raise ConfigError("collection.conda_env must be a non-empty string")
    platform = _require(collection, "platform", "collection")
    if platform not in ("cpu", "gpu"):
        raise ConfigError("collection.platform must be 'cpu' or 'gpu'")
    for key in ("steps", "width", "height", "fps"):
        _positive_int(_require(collection, key, "collection"), f"collection.{key}")
    for key in ("puck_radius", "mallet_radius", "robot_visual_scale"):
        _positive_number(_require(collection, key, "collection"), f"collection.{key}")
    if "collector_args" in collection:
        _validate_collector_options(collection["collector_args"], "collection.collector_args")

    for key, default in (("img_size", 128), ("shard_size", 256), ("min_len", 20),
                         ("procs", 16)):
        _positive_int(conversion.get(key, default), f"conversion.{key}")
    val_frac = conversion.get("val_frac", 0.05)
    if isinstance(val_frac, bool) or not isinstance(val_frac, (int, float)):
        raise ConfigError("conversion.val_frac must be a number in [0, 1]")
    if not 0.0 <= float(val_frac) <= 1.0:
        raise ConfigError("conversion.val_frac must be in [0, 1]")
    if "overwrite" in conversion and not isinstance(conversion["overwrite"], bool):
        raise ConfigError("conversion.overwrite must be a boolean")

    total_games = sum(source["games"] for source in sources)
    _validate_mixture(config.get("mixture"), sources, total_games)

    # A canonical representation lets metadata distinguish meaningful config
    # changes while ignoring YAML comments, key order, and whitespace.
    try:
        canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"config must be JSON-compatible: {exc}") from exc
    return config, raw_text, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _add_cli_options(command: List[str], options: Mapping[str, Any], label: str) -> None:
    """Append an explicit argparse-style option map to a command.

    Keys use Python/YAML style (``robot_visual_scale``) and map directly to
    collector CLI spelling (``--robot-visual-scale``). ``true`` emits a flag,
    ``false`` omits it, and a scalar emits ``--flag VALUE``. Repeated options
    can be expressed as a list.
    """
    for key, value in options.items():
        if not isinstance(key, str) or not key or key.startswith("-"):
            raise ConfigError(f"{label} keys must be bare option names")
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                command.append(flag)
            continue
        if value is None:
            raise ConfigError(f"{label}.{key} cannot be null")
        if isinstance(value, (dict, tuple)):
            raise ConfigError(f"{label}.{key} must be a scalar or list")
        if isinstance(value, list):
            if not value:
                raise ConfigError(f"{label}.{key} must not be an empty list")
            for item in value:
                if isinstance(item, (dict, list, tuple)) or item is None:
                    raise ConfigError(f"{label}.{key} list items must be scalars")
                command.extend((flag, str(item)))
            continue
        command.extend((flag, str(value)))


def _read_complete_game_metadata(game_dir: Path) -> Optional[Dict[str, Any]]:
    """Read a game only after its atomic collector completion marker exists."""
    meta_path = game_dir / "meta.json"
    if not meta_path.is_file() or not any(game_dir.glob("episode_*.npz")):
        return None
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            return None
        n_episodes = int(metadata.get("n_episodes", 0))
    except (AttributeError, OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    return metadata if n_episodes > 0 else None


def _completed_games(source_dir: Path) -> List[Tuple[Path, Dict[str, Any]]]:
    """Find only fully written direct and multiworker collection games."""
    candidates = sorted(source_dir.glob("collect-*/game_*")) + sorted(
        source_dir.glob("game_*")
    )
    completed = []
    for game_dir in candidates:
        if not game_dir.is_dir():
            continue
        metadata = _read_complete_game_metadata(game_dir)
        if metadata is not None:
            completed.append((game_dir, metadata))
    return completed


def _next_seed(source: Mapping[str, Any], completed: Iterable[Tuple[Path, Mapping[str, Any]]]) -> int:
    """Choose an unused seed even if an interrupted run left holes in game IDs."""
    seed_base = int(source.get("seed_base", 0))
    completed_seeds = [seed_base - 1]
    for _, metadata in completed:
        seed = metadata.get("seed")
        if isinstance(seed, int) and not isinstance(seed, bool):
            completed_seeds.append(seed)
    return max(completed_seeds) + 1


def _collection_content_hash(config: Mapping[str, Any]) -> str:
    """Fingerprint raw-data semantics while allowing target/worker changes."""
    collection = config["collection"]
    payload = {
        "schema": "airhockey_raw_content_v1",
        "collection": {
            key: collection[key]
            for key in (
                "platform", "steps", "width", "height", "fps", "puck_radius",
                "mallet_radius", "robot_visual_scale",
            )
        },
        "collection_collector_args": collection.get("collector_args", {}),
        "sources": [
            {
                "name": source["name"],
                "model1": source["model1"],
                "model2": source["model2"],
                "seed_base": source.get("seed_base", 0),
                "collector_args": source.get("collector_args", {}),
            }
            for source in config["sources"]
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _quoted(command: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _has_completed_games(raw_dir: Path) -> bool:
    """Check every source-like child, including data from a previous config."""
    if not raw_dir.is_dir():
        return False
    for child in raw_dir.iterdir():
        if child.name in {".collection", "logs"} or not child.is_dir():
            continue
        if _completed_games(child):
            return True
    return False


def _assert_compatible_existing_raw(
    raw_dir: Path, content_hash: str, allow_mixed_config: bool
) -> None:
    """Prevent silent mixing of differently rendered or controlled raw data."""
    if not _has_completed_games(raw_dir):
        return

    metadata_dir = raw_dir / ".collection"
    hashes = set()
    unreadable_metadata = False
    if metadata_dir.is_dir():
        for metadata_path in metadata_dir.glob("*.json"):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                unreadable_metadata = True
                continue
            stored_hash = metadata.get("collection_content_sha256")
            if isinstance(stored_hash, str):
                hashes.add(stored_hash)
            else:
                unreadable_metadata = True

    compatible = hashes == {content_hash} and not unreadable_metadata
    if compatible or allow_mixed_config:
        return

    if not hashes:
        reason = "completed raw games have no compatible YAML collection metadata"
    elif content_hash not in hashes:
        reason = "completed raw games were created with different collection settings"
    else:
        reason = "completed raw games have incomplete or mixed collection metadata"
    raise ConfigError(
        f"{reason} under {raw_dir}; use a separate raw_dir, or pass "
        "--allow-mixed-config only if you intentionally want to mix them"
    )


class CollectionLock:
    """A root-level lock preventing two launchers from allocating the same seeds."""

    def __init__(self, raw_dir: Path, config_hash: str):
        self.path = raw_dir / ".collection" / "active.lock"
        self.config_hash = config_hash
        self.token = uuid.uuid4().hex
        self.held = False

    def acquire(self, break_existing: bool = False) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if break_existing:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
        payload = {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "config_sha256": self.config_hash,
            "token": self.token,
        }
        try:
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError as exc:
            try:
                details = self.path.read_text(encoding="utf-8").strip()
            except OSError:
                details = ""
            suffix = f" ({details})" if details else ""
            raise RuntimeError(
                f"collection lock already exists: {self.path}{suffix}. "
                "Do not run two launchers on the same raw_dir; after confirming "
                "the lock is stale, rerun with --break-lock."
            ) from exc
        try:
            os.write(fd, (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8"))
        finally:
            os.close(fd)
        self.held = True

    def release(self) -> None:
        if self.held:
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
                if payload.get("token") == self.token:
                    self.path.unlink()
            except (FileNotFoundError, OSError, json.JSONDecodeError, AttributeError):
                pass
            self.held = False


def _write_snapshot(
    raw_dir: Path,
    h5_dir: Path,
    config_path: Path,
    raw_config: str,
    config_hash: str,
    collection_content_hash: str,
    sources: List[Mapping[str, Any]],
    source_runs: List[Mapping[str, Any]],
    stamp: str,
) -> Tuple[Path, Path, Dict[str, Any]]:
    """Write immutable input config and mutable status metadata for this run."""
    metadata_dir = raw_dir / ".collection"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{stamp}-{config_hash[:12]}"
    snapshot_path = metadata_dir / f"{prefix}.yaml"
    metadata_path = metadata_dir / f"{prefix}.json"
    snapshot_path.write_text(raw_config, encoding="utf-8")
    metadata = {
        "schema": "airhockey_collection_run_v1",
        "status": "collecting",
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config_path": str(config_path),
        "config_sha256": config_hash,
        "collection_content_sha256": collection_content_hash,
        "config_snapshot": str(snapshot_path),
        "raw_dir": str(raw_dir),
        "h5_dir": str(h5_dir),
        "sources": source_runs,
        "target_games": sum(int(source["games"]) for source in sources),
        "new_games_requested": sum(int(run["remaining_games"]) for run in source_runs),
    }
    _atomic_json(metadata_path, metadata)
    return snapshot_path, metadata_path, metadata


def _collector_command(
    collection: Mapping[str, Any], source: Mapping[str, Any], source_dir: Path,
    games: int, seed: int,
) -> List[str]:
    """Build the fully explicit child collector command for one source."""
    command = [
        "conda", "run", "-n", str(collection["conda_env"]), "python",
        str(REPO_ROOT / "scripts" / "collect_2023.py"),
        "--platform", str(collection["platform"]),
        "--workers", str(source["workers"]),
        "--model1", str(source["model1"]),
        "--model2", str(source["model2"]),
        "--games", str(games),
        "--steps", str(collection["steps"]),
        "--puck-radius", str(collection["puck_radius"]),
        "--mallet-radius", str(collection["mallet_radius"]),
        "--robot-visual-scale", str(collection["robot_visual_scale"]),
        "--width", str(collection["width"]),
        "--height", str(collection["height"]),
        "--fps", str(collection["fps"]),
        "--seed", str(seed),
        "--out", str(source_dir),
    ]
    _add_cli_options(command, collection.get("collector_args", {}), "collection.collector_args")
    _add_cli_options(command, source.get("collector_args", {}),
                     f"source {source['name']}.collector_args")
    return command


def _conversion_command(
    collection: Mapping[str, Any], conversion: Mapping[str, Any],
    sources: Iterable[Mapping[str, Any]], raw_dir: Path, h5_dir: Path,
) -> List[str]:
    command = [
        "conda", "run", "-n", str(collection["conda_env"]), "python",
        str(REPO_ROOT / "scripts" / "convert_npz_to_sa_h5.py"),
    ]
    for source in sources:
        command.extend(("--source", f"{source['name']}={raw_dir / source['name']}"))
    command.extend((
        "--out", str(h5_dir),
        "--img-size", str(conversion.get("img_size", 128)),
        "--shard-size", str(conversion.get("shard_size", 256)),
        "--val-frac", str(conversion.get("val_frac", 0.05)),
        "--min-len", str(conversion.get("min_len", 20)),
        "--procs", str(conversion.get("procs", 16)),
    ))
    if conversion.get("overwrite", True):
        command.append("--overwrite")
    return command


def _terminate_jobs(jobs: Iterable[Tuple[str, subprocess.Popen, Any]]) -> None:
    for _, process, _ in jobs:
        if process.poll() is None:
            process.terminate()
    for _, process, handle in jobs:
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        handle.close()


def _run_collectors(
    jobs: List[Tuple[str, List[str], Path]],
) -> None:
    """Start sources together; stop peers if any source fails."""
    running = []
    try:
        for name, command, log_path in jobs:
            log_handle = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT)
            running.append((name, process, log_handle))
        pending = list(running)
        while pending:
            failures = []
            for job in pending[:]:
                name, process, log_handle = job
                return_code = process.poll()
                if return_code is None:
                    continue
                pending.remove(job)
                log_handle.close()
                if return_code:
                    failures.append((name, return_code))
            if failures:
                # A source can fail hours before another one finishes. Poll
                # every child rather than waiting in source order, then stop
                # the expensive peers immediately.
                _terminate_jobs(pending)
                details = ", ".join(f"{name} (exit {code})" for name, code in failures)
                raise RuntimeError(f"collection failed: {details}")
            if pending:
                time.sleep(0.2)
    except BaseException:
        _terminate_jobs(running)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG,
        help=f"dataset YAML (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="validate and print commands without collecting or converting",
    )
    parser.add_argument(
        "--skip-convert", action="store_true",
        help="collect raw games only; do not rebuild the H5 dataset",
    )
    parser.add_argument(
        "--allow-mixed-config", action="store_true",
        help="allow adding to existing raw games with incompatible or missing YAML metadata",
    )
    parser.add_argument(
        "--break-lock", action="store_true",
        help="remove an existing raw-dir collection lock; use only after confirming it is stale",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    try:
        config, raw_config, config_hash = load_config(config_path)
    except ConfigError as exc:
        raise SystemExit(f"config error: {exc}") from exc

    dataset = config["dataset"]
    collection = config["collection"]
    conversion = config["conversion"]
    sources = config["sources"]
    raw_dir = _path(dataset["raw_dir"], "dataset.raw_dir")
    h5_dir = _path(dataset["h5_dir"], "dataset.h5_dir")
    content_hash = _collection_content_hash(config)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")

    lock = None
    try:
        if not args.dry_run:
            lock = CollectionLock(raw_dir, config_hash)
            try:
                lock.acquire(break_existing=args.break_lock)
            except RuntimeError as exc:
                raise SystemExit(str(exc)) from exc
        _assert_compatible_existing_raw(raw_dir, content_hash, args.allow_mixed_config)

        source_runs = []
        jobs = []
        for source in sources:
            source_dir = raw_dir / source["name"]
            completed = _completed_games(source_dir)
            completed_games = len(completed)
            target_games = int(source["games"])
            remaining_games = max(0, target_games - completed_games)
            seed = _next_seed(source, completed)
            log_path = raw_dir / "logs" / f"{source['name']}-{stamp}.log"
            command = None
            if remaining_games:
                command = _collector_command(
                    collection, source, source_dir, remaining_games, seed
                )
                jobs.append((source["name"], command, log_path))
            source_runs.append({
                "name": source["name"],
                "raw_dir": str(source_dir),
                "log": str(log_path) if remaining_games else None,
                "completed_games": completed_games,
                "target_games": target_games,
                "remaining_games": remaining_games,
                "seed": seed if remaining_games else None,
                "workers": int(source["workers"]),
                "command": command,
            })

        conversion_command = _conversion_command(collection, conversion, sources, raw_dir, h5_dir)
        print(
            f"[config] {config_path} (sha256 {config_hash}; "
            f"content {content_hash})"
        )
        for source_run in source_runs:
            if source_run["remaining_games"]:
                print(
                    f"[collect] {source_run['name']}: {source_run['remaining_games']} remaining "
                    f"of {source_run['target_games']} games x {collection['steps']} steps, "
                    f"{source_run['workers']} workers, seed {source_run['seed']} "
                    f"({source_run['completed_games']} complete)"
                )
                print(f"          {_quoted(source_run['command'])}")
            else:
                print(
                    f"[collect] {source_run['name']}: target already complete "
                    f"({source_run['completed_games']}/{source_run['target_games']} games)"
                )
        if not args.skip_convert:
            print(f"[convert] {_quoted(conversion_command)}")
        if args.dry_run:
            print("[dry-run] configuration is valid; no files or processes were created")
            return

        (raw_dir / "logs").mkdir(parents=True, exist_ok=True)
        _, metadata_path, metadata = _write_snapshot(
            raw_dir, h5_dir, config_path, raw_config, config_hash, content_hash,
            sources, source_runs, stamp,
        )
        try:
            if jobs:
                _run_collectors(jobs)
            metadata["status"] = "collected"
            metadata["collected_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
            _atomic_json(metadata_path, metadata)
            if args.skip_convert:
                print(f"[done] raw data collected under {raw_dir}")
                return
            print(f"[convert] rebuilding {h5_dir} from completed raw games")
            subprocess.run(conversion_command, check=True)
            metadata["status"] = "complete"
            metadata["converted_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
            _atomic_json(metadata_path, metadata)
            h5_dir.mkdir(parents=True, exist_ok=True)
            _atomic_json(h5_dir / "collection_config.json", metadata)
            print(f"[done] dataset at {h5_dir} (manifest: {h5_dir / 'h5_manifest.json'})")
        except BaseException as exc:
            metadata["status"] = "failed"
            metadata["failed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
            metadata["error"] = str(exc)
            _atomic_json(metadata_path, metadata)
            raise
    except ConfigError as exc:
        raise SystemExit(f"config error: {exc}") from exc
    finally:
        if lock is not None:
            lock.release()


if __name__ == "__main__":
    main()
