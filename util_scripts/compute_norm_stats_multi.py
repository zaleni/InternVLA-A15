#!/usr/bin/env python

import argparse
import hashlib
import multiprocessing as mp
from pathlib import Path

import tqdm
import torch
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.dataset_schemas import get_schema
from lerobot.utils.constants import OBS_STATE, ACTION, HF_LEROBOT_HOME
from lerobot.datasets.utils import cast_stats_to_numpy, write_json


DEFAULT_DATASET_ROOT = Path("/data/datasets/internvla_data")
DEFAULT_STATS_ROOT = Path("/data/jjhao/huggingface/lerobot/stats")


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute (and aggregate) normalization statistics for LeRobot datasets",
    )

    p.add_argument(
        "--action_mode",
        type=str,
        choices=["abs", "delta"],
        required=True,
        help="Action mode used to compute statistics (abs or delta).",
    )
    p.add_argument(
        "--chunk_size",
        type=int,
        required=True,
        help="Chunk size used for delta action computation (episodes shorter than chunk_size are skipped).",
    )
    p.add_argument(
        "--repo_ids",
        type=str,
        nargs="+",
        required=True,
        help="One or more LeRobotDataset repo ids (must share the same robot_type and feature schema).",
    )
    p.add_argument(
        "--root",
        type=str,
        default=None,
        help=(
            "Optional local dataset root. If omitted, each repo is resolved "
            f"from {DEFAULT_DATASET_ROOT} first, then HF_LEROBOT_HOME."
        ),
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of worker processes (repo-level parallelism).",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_STATS_ROOT),
        help=(
            "Stats root directory. Output is written below "
            "<output_dir>/<action_mode>/<robot_type>/<stats_name>/stats.json. "
            f"Default: {DEFAULT_STATS_ROOT}"
        ),
    )
    p.add_argument(
        "--stats_name",
        type=str,
        default=None,
        help=(
            "Optional readable output directory name. "
            "If omitted, a stable repo-list hash is used."
        ),
    )

    return p.parse_args()


class RunningStats:
    """Running stats for vectors: keeps count, mean, mean_sq, min, max."""

    def __init__(self):
        self._count = 0
        self._mean = None
        self._mean_of_squares = None
        self._min = None
        self._max = None

    def update(self, batch: torch.Tensor) -> None:
        batch = batch.to(torch.float32)

        if batch.ndim == 1:
            batch = batch[:, None]
        if batch.ndim > 1:
            batch = batch.reshape(-1, batch.shape[-1])

        count = batch.shape[0]
        mean = batch.mean(dim=0)
        mean_sq = (batch ** 2).mean(dim=0)
        min_ = batch.min(dim=0).values
        max_ = batch.max(dim=0).values

        if self._count == 0:
            self._count = count
            self._mean = mean
            self._mean_of_squares = mean_sq
            self._min = min_
            self._max = max_
            return

        total = self._count + count
        w_old = self._count / total
        w_new = count / total

        self._mean = w_old * self._mean + w_new * mean
        self._mean_of_squares = w_old * self._mean_of_squares + w_new * mean_sq
        self._min = torch.minimum(self._min, min_)
        self._max = torch.maximum(self._max, max_)
        self._count = total

    def merge(self, other: "RunningStats") -> None:
        """Merge another RunningStats (exact for mean/mean_sq/min/max)."""
        if other._count == 0:
            return
        if self._count == 0:
            self._count = other._count
            self._mean = other._mean.clone()
            self._mean_of_squares = other._mean_of_squares.clone()
            self._min = other._min.clone()
            self._max = other._max.clone()
            return

        total = self._count + other._count
        w_self = self._count / total
        w_other = other._count / total

        self._mean = w_self * self._mean + w_other * other._mean
        self._mean_of_squares = w_self * self._mean_of_squares + w_other * other._mean_of_squares
        self._min = torch.minimum(self._min, other._min)
        self._max = torch.maximum(self._max, other._max)
        self._count = total

    def to_payload(self) -> dict:
        """Serialize to a JSON-friendly dict."""
        if self._count == 0:
            # Keep empty stats explicit
            return {
                "count": 0,
                "mean": None,
                "mean_sq": None,
                "min": None,
                "max": None,
            }
        return {
            "count": int(self._count),
            "mean": self._mean.detach().cpu().tolist(),
            "mean_sq": self._mean_of_squares.detach().cpu().tolist(),
            "min": self._min.detach().cpu().tolist(),
            "max": self._max.detach().cpu().tolist(),
        }

    @staticmethod
    def from_payload(p: dict) -> "RunningStats":
        rs = RunningStats()
        if p["count"] == 0:
            return rs
        rs._count = int(p["count"])
        rs._mean = torch.tensor(p["mean"], dtype=torch.float32)
        rs._mean_of_squares = torch.tensor(p["mean_sq"], dtype=torch.float32)
        rs._min = torch.tensor(p["min"], dtype=torch.float32)
        rs._max = torch.tensor(p["max"], dtype=torch.float32)
        return rs

    def get_statistics(self) -> dict:
        """Return mean, std, min, max, count."""
        if self._count == 0:
            raise ValueError("No data has been added yet.")
        var = self._mean_of_squares - self._mean ** 2
        std = torch.sqrt(torch.clamp(var, min=0.0))
        return {
            "min": self._min.tolist(),
            "max": self._max.tolist(),
            "mean": self._mean.tolist(),
            "std": std.tolist(),
            "count": [int(self._count)],
        }


def _compute_one_repo(
    repo_id: str,
    action_mode: str,
    chunk_size: int,
    repo_root: str | None,
) -> dict:
    """Worker: compute stats for one repo, return serializable payload."""
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    dataset = LeRobotDataset(repo_id, root=repo_root)
    robot_type = dataset.meta.robot_type

    schema = get_schema(robot_type)
    mask = schema.action_mask
    mapping = schema.feature_mapping

    keys = list(dataset.meta.features.keys())
    for k in dataset.meta.video_keys + dataset.meta.image_keys:
        if k in keys:
            keys.remove(k)

    # Capture schema for consistency checks
    shapes = {k: dataset.meta.features[k]["shape"] for k in keys}
    visual_keys = [
        k
        for k in dataset.meta.video_keys + dataset.meta.image_keys
        if k in dataset.meta.stats
    ]
    visual_shapes = {k: dataset.meta.features[k]["shape"] for k in visual_keys}
    visual_stats = {
        k: _normalize_visual_stats(dataset.meta.stats[k]) for k in visual_keys
    }

    stats = {k: RunningStats() for k in keys}
    total_frames = 0
    skipped_episodes = 0

    from_ids = np.asarray(dataset.meta.episodes["dataset_from_index"])
    to_ids = np.asarray(dataset.meta.episodes["dataset_to_index"])
    total_episodes = dataset.num_episodes

    for from_idx, to_idx in zip(from_ids, to_ids):
        ep_len = int(to_idx - from_idx)
        total_frames += ep_len

        if ep_len < chunk_size:
            skipped_episodes += 1
            continue

        curr_episode = dataset.hf_dataset.select(np.arange(from_idx, to_idx))

        # Non-action stats always update; action stats depend on mode
        for key in keys:
            if action_mode == "abs" or key not in mapping[ACTION]:
                val = torch.stack(curr_episode[key][:])
                stats[key].update(val)

        if action_mode == "delta":
            action = [torch.stack(curr_episode[k][:]) for k in mapping[ACTION]]
            action = [a if a.ndim > 1 else a[:, None] for a in action]
            action = torch.cat(action, dim=-1)

            state = [torch.stack(curr_episode[k][:]) for k in mapping[OBS_STATE]]
            state = [s if s.ndim > 1 else s[:, None] for s in state]
            state = torch.cat(state, dim=-1)

            truncated_state = state[0 : (ep_len - chunk_size + 1)]
            action_chunk = action.unfold(dimension=0, size=chunk_size, step=1).permute(0, 2, 1)
            delta_action = action_chunk - torch.where(mask, truncated_state, 0)[:, None]

            sid, eid = 0, 0
            for action_key in mapping[ACTION]:
                eid += dataset.meta.features[action_key]["shape"][0]
                stats[action_key].update(delta_action[..., sid:eid])
                sid = eid

    payload = {k: stats[k].to_payload() for k in keys}

    return {
        "repo_id": repo_id,
        "robot_type": robot_type,
        "keys": keys,
        "shapes": shapes,
        "payload": payload,
        "visual_keys": visual_keys,
        "visual_shapes": visual_shapes,
        "visual_stats": visual_stats,
        "total_frames": int(total_frames),
        "skipped_episodes": int(skipped_episodes),
        "total_episodes": int(total_episodes),
    }


def _make_group_name(repo_ids: list[str]) -> str:
    """Short stable name for a repo set."""
    joined = "|".join(repo_ids)
    h = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:10]
    return f"agg_{len(repo_ids)}repos_{h}"


def _validate_stats_name(stats_name: str) -> str:
    """Require a single safe directory component."""
    candidate = Path(stats_name)
    if (
        not stats_name
        or stats_name in {".", ".."}
        or candidate.is_absolute()
        or len(candidate.parts) != 1
    ):
        raise ValueError(
            f"--stats_name must be one non-empty directory name, got: {stats_name!r}"
        )
    return stats_name


def _resolve_repo_roots(repo_ids: list[str], root: str | None) -> list[str | None]:
    """Resolve an optional direct dataset root or a parent containing repo IDs."""
    if root is None:
        repo_roots = []
        for repo_id in repo_ids:
            preferred_root = DEFAULT_DATASET_ROOT / repo_id
            fallback_root = HF_LEROBOT_HOME / repo_id
            if (preferred_root / "meta" / "info.json").is_file():
                repo_roots.append(preferred_root)
            elif (fallback_root / "meta" / "info.json").is_file():
                repo_roots.append(fallback_root)
            else:
                raise FileNotFoundError(
                    f"Local dataset not found for {repo_id!r}. Expected either "
                    f"{preferred_root / 'meta' / 'info.json'} or "
                    f"{fallback_root / 'meta' / 'info.json'}."
                )
        return [str(repo_root) for repo_root in repo_roots]

    root_path = Path(root).expanduser()
    if (root_path / "meta" / "info.json").is_file():
        if len(repo_ids) != 1:
            raise ValueError(
                "--root points directly to one dataset, but multiple --repo_ids were provided."
            )
        repo_roots = [root_path]
    else:
        repo_roots = [root_path / repo_id for repo_id in repo_ids]

    for repo_id, repo_root in zip(repo_ids, repo_roots, strict=True):
        info_path = repo_root / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(
                f"Local dataset not found for {repo_id!r}: expected {info_path}"
            )
    return [str(repo_root) for repo_root in repo_roots]


def _normalize_visual_stats(visual_stats: dict) -> dict:
    """Convert numpy/torch entries to lists."""
    out = {}
    for k, v in visual_stats.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, torch.Tensor):
            out[k] = v.detach().cpu().numpy().tolist()
        else:
            out[k] = v
    return out


def _aggregate_visual_stats(results: list[dict]) -> dict:
    """Aggregate common camera stats using LeRobot's official weighted merge."""
    common_keys = set(results[0]["visual_keys"])
    for result in results[1:]:
        common_keys &= set(result["visual_keys"])

    bad_shape_keys = set()
    first_shapes = results[0]["visual_shapes"]
    for result in results[1:]:
        for key in common_keys:
            if result["visual_shapes"][key] != first_shapes[key]:
                bad_shape_keys.add(key)
    common_keys -= bad_shape_keys

    ordered_keys = [
        key for key in results[0]["visual_keys"] if key in common_keys
    ]
    dropped = {
        result["repo_id"]: [
            key for key in result["visual_keys"] if key not in common_keys
        ]
        for result in results
    }
    dropped = {repo_id: keys for repo_id, keys in dropped.items() if keys}
    if dropped:
        print("[WARN] Ignoring visual stats not common to all repos:")
        for repo_id, keys in dropped.items():
            print(f"  - {repo_id}: {keys}")
    if bad_shape_keys:
        print(
            "[WARN] Dropped visual stats due to shape mismatch across repos: "
            f"{sorted(bad_shape_keys)}"
        )
    if not ordered_keys:
        return {}

    if len(results) == 1:
        return {
            key: results[0]["visual_stats"][key] for key in ordered_keys
        }

    per_repo_stats = [
        cast_stats_to_numpy(
            {key: result["visual_stats"][key] for key in ordered_keys}
        )
        for result in results
    ]
    aggregated = aggregate_stats(per_repo_stats)
    return {
        key: _normalize_visual_stats(aggregated[key]) for key in ordered_keys
    }


def compute_norm_stats_multi(cfg):
    repo_ids = cfg.repo_ids
    action_mode = cfg.action_mode
    chunk_size = cfg.chunk_size
    group_name = (
        _validate_stats_name(cfg.stats_name)
        if cfg.stats_name is not None
        else _make_group_name(repo_ids)
    )
    repo_roots = _resolve_repo_roots(repo_ids, cfg.root)

    print(f"---------- aggregate stats for {len(repo_ids)} datasets ----------")
    for rid, repo_root in zip(repo_ids, repo_roots, strict=True):
        print(f"  - {rid}: {repo_root or HF_LEROBOT_HOME / rid}")
    print(f"stats_name: {group_name}")

    # Repo-level parallelism
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=cfg.num_workers) as pool:
        results = list(
            tqdm.tqdm(
                pool.starmap(
                    _compute_one_repo,
                    [
                        (rid, action_mode, chunk_size, repo_root)
                        for rid, repo_root in zip(repo_ids, repo_roots, strict=True)
                    ],
                ),
                total=len(repo_ids),
                desc="Computing per-repo stats",
            )
        )

    # Consistency checks
    robot_types = {r["robot_type"] for r in results}
    if len(robot_types) != 1:
        raise ValueError(f"repo_ids must share the same robot_type, got: {sorted(robot_types)}")
    robot_type = results[0]["robot_type"]

    # Take the intersection of keys across all repos (preserve order from results[0]).
    # Keys that only exist in some repos, or whose shapes differ across repos, are dropped
    # from the aggregated stats with a warning.
    common_keys = set(results[0]["keys"])
    for r in results[1:]:
        common_keys &= set(r["keys"])

    shapes0 = results[0]["shapes"]
    bad_shape_keys = set()
    for r in results[1:]:
        for k in common_keys:
            if r["shapes"][k] != shapes0[k]:
                bad_shape_keys.add(k)
    common_keys -= bad_shape_keys

    keys0 = [k for k in results[0]["keys"] if k in common_keys]

    dropped_keys = {}
    for r in results:
        extra = [k for k in r["keys"] if k not in common_keys]
        if extra:
            dropped_keys[r["repo_id"]] = extra
    if dropped_keys:
        print("[WARN] Ignoring keys not common to all repos (or with shape mismatch):")
        for rid, ks in dropped_keys.items():
            print(f"  - {rid}: {ks}")
    if bad_shape_keys:
        print(f"[WARN] Dropped due to shape mismatch across repos: {sorted(bad_shape_keys)}")
    if not keys0:
        raise ValueError("No common feature keys across the provided repos.")

    # Merge numeric stats
    global_stats = {k: RunningStats() for k in keys0}
    total_frames = 0
    total_episodes = 0
    skipped_episodes = 0

    for r in results:
        total_frames += r["total_frames"]
        total_episodes += r["total_episodes"]
        skipped_episodes += r["skipped_episodes"]
        for k in keys0:
            tmp = RunningStats.from_payload(r["payload"][k])
            global_stats[k].merge(tmp)

    output_dict = {k: global_stats[k].get_statistics() for k in keys0}
    output_dict.update(_aggregate_visual_stats(results))

    # Output path: keep the established local layout, e.g.
    # stats/delta/arx_acone/<stats_name>/stats.json.
    output_dir = (
        Path(cfg.output_dir).expanduser()
        / action_mode
        / robot_type
        / group_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dict, output_dir / "stats.json")

    print("---------- done ----------")
    print(f"robot_type: {robot_type}")
    print(f"action_mode: {action_mode}")
    print(f"chunk_size: {chunk_size}")
    print(f"stats_name: {group_name}")
    print(f"output: {output_dir / 'stats.json'}")
    print(f"total_frames (sum of episode lengths): {total_frames}")
    print(f"total_episodes: {total_episodes} (skipped: {skipped_episodes} episodes with len < chunk_size)")


if __name__ == "__main__":
    compute_norm_stats_multi(parse_args())
