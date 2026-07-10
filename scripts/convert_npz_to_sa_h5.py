#!/usr/bin/env python3
"""Convert collect_2023.py npz episodes into SA's langtable_action_h5_v1 format.

Output layout (what X/sa's LanguageTableH5Dataset reads, dataset=languagetable):

    <out>/h5_manifest.json                       # loader never globs *.h5
    <out>/train/shard_00000.h5 ...
    <out>/valid/shard_00000.h5 ...
    # shard schema: /episodes/<key>/images  (T, S, S, 3) uint8
    #               /episodes/<key>/actions (T, 4)       float32

Alignment: SA expects actions[t] to be the control applied at frame t that
produces frame t+1. The npz convention is obs-first (action[k] led INTO
image[k], action[0] = zeros), so per episode we store
    images[t]  = npz image[t]        for t in [0, T)
    actions[t] = npz action[t + 1]   flattened to [a1x, a1y, a2x, a2y]
dropping the final frame, which has no outgoing action.

Frames are box-downsampled (exact integer factor, e.g. 256 -> 128). Each
--source is a directory containing collect-*/game_*/ (or game_*/ directly);
the train/valid split is done at GAME level per source so validation games
are entirely held out.

Example:
    python scripts/convert_npz_to_sa_h5.py \
        --source aggressive=/home_shared/grail_charles/data/raw_2023/aggressive \
        --source random=/home_shared/grail_charles/data/raw_2023/random \
        --out /home_shared/grail_charles/data/airhockey_sa_h5_v1 --overwrite
"""
import argparse
import json
import shutil
import time
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np

MANIFEST_FORMAT = "langtable_action_h5_v1"
ACTION_DIM = 4


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="NAME=DIR",
        help="named episode source; DIR holds collect-*/game_*/ or game_*/",
    )
    parser.add_argument("--out", required=True, help="H5 root to create")
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--shard-size", type=int, default=256, help="episodes per shard")
    parser.add_argument("--val-frac", type=float, default=0.05, help="held-out games per source")
    parser.add_argument("--min-len", type=int, default=20, help="skip shorter episodes (frames)")
    parser.add_argument("--procs", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true", help="rebuild an existing H5 root")
    return parser.parse_args()


def discover_games(source_dir):
    """Return the game dirs under a source, sorted by (run, game index)."""
    root = Path(source_dir)
    games = sorted(root.glob("collect-*/game_*")) + sorted(root.glob("game_*"))
    return [g for g in games if g.is_dir()]


def load_episode(task):
    """Read one npz -> (key, images uint8 (T,S,S,3), actions float32 (T,4))."""
    key, path, img_size, min_len = task
    with np.load(path) as d:
        images = d["image"]
        actions = d["action"]
    n_frames = images.shape[0] - 1  # drop the final frame (no outgoing action)
    if n_frames < min_len:
        return None
    images = images[:-1]
    actions = actions[1:].reshape(n_frames, -1).astype(np.float32)
    if actions.shape[1] != ACTION_DIM:
        raise ValueError(f"{path}: action dim {actions.shape[1]} != {ACTION_DIM}")
    h, w = images.shape[1:3]
    if h != img_size or w != img_size:
        if h % img_size or w % img_size:
            raise ValueError(f"{path}: {h}x{w} not an integer multiple of {img_size}")
        fh, fw = h // img_size, w // img_size
        images = (
            images.reshape(n_frames, img_size, fh, img_size, fw, 3)
            .astype(np.uint16)
            .sum(axis=(2, 4))
            // (fh * fw)
        ).astype(np.uint8)
    return key, images, actions


class ShardWriter:
    def __init__(self, out_root, split, shard_size):
        self.split_dir = Path(out_root) / split
        self.split_dir.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.shard_size = shard_size
        self.entries = []
        self._h5 = None

    def _shard_name(self):
        return f"shard_{len(self.entries) // self.shard_size:05d}.h5"

    def add(self, key, images, actions):
        if self._h5 is None or len(self.entries) % self.shard_size == 0:
            self.close()
            self._h5 = h5py.File(self.split_dir / self._shard_name(), "w")
            self._h5.attrs["complete"] = False
        group = self._h5.require_group("episodes").create_group(key)
        chunk = (min(len(images), 16),) + images.shape[1:]
        group.create_dataset(
            "images", data=images, chunks=chunk, compression="gzip", compression_opts=4
        )
        group.create_dataset("actions", data=actions)
        self.entries.append(
            {
                "shard": f"{self.split}/{self._h5.filename.rsplit('/', 1)[1]}",
                "key": key,
                "length": int(len(images)),
            }
        )

    def close(self):
        if self._h5 is not None:
            self._h5.attrs["complete"] = True
            self._h5.close()
            self._h5 = None


def main():
    args = parse_args()
    out = Path(args.out)
    manifest_path = out / "h5_manifest.json"
    if manifest_path.exists():
        if not args.overwrite:
            raise SystemExit(f"{manifest_path} exists; pass --overwrite to rebuild")
        for split in ("train", "valid", "test"):
            shutil.rmtree(out / split, ignore_errors=True)
        manifest_path.unlink()
    out.mkdir(parents=True, exist_ok=True)

    # NAME=DIR -> ordered per-split task lists (train first, then valid)
    tasks = {"train": [], "valid": []}
    source_stats = {}
    for spec in args.source:
        name, _, src_dir = spec.partition("=")
        if not src_dir:
            raise SystemExit(f"--source must be NAME=DIR, got: {spec}")
        games = discover_games(src_dir)
        if not games:
            raise SystemExit(f"no game_* dirs found under {src_dir}")
        n_val = 0 if len(games) < 2 else max(1, round(args.val_frac * len(games)))
        source_stats[name] = {"dir": str(src_dir), "games": len(games), "val_games": n_val}
        for gi, game in enumerate(games):
            split = "valid" if gi >= len(games) - n_val else "train"
            for ep_path in sorted(game.glob("episode_*.npz")):
                key = f"{name}_g{gi:04d}_{ep_path.stem}"
                tasks[split].append((key, str(ep_path), args.img_size, args.min_len))

    manifest_splits = {"train": [], "valid": [], "test": []}
    n_frames = 0
    n_skipped = 0
    start = time.time()
    with Pool(args.procs) as pool:
        for split in ("train", "valid"):
            writer = ShardWriter(out, split, args.shard_size)
            for result in pool.imap(load_episode, tasks[split]):
                if result is None:
                    n_skipped += 1
                    continue
                key, images, actions = result
                writer.add(key, images, actions)
                n_frames += len(images)
            writer.close()
            manifest_splits[split] = writer.entries
            print(f"{split}: {len(writer.entries)} episodes")

    manifest = {
        "format": MANIFEST_FORMAT,
        "complete": True,
        "action_dim": ACTION_DIM,
        "image_size": [args.img_size, args.img_size],
        "shard_size": args.shard_size,
        "splits": manifest_splits,
        "metadata": {
            "sources": source_stats,
            "total_frames": n_frames,
            "skipped_short_episodes": n_skipped,
            "action_layout": "[agent1_x, agent1_y, agent2_x, agent2_y] commanded mallet xy, world frame",
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"wrote {manifest_path}: {n_frames} frames, "
        f"{n_skipped} short episodes skipped, {time.time() - start:.0f}s"
    )


if __name__ == "__main__":
    main()
