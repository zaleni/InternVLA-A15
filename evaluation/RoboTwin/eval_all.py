#!/usr/bin/env python3
"""Run InternVLA-A1.5 RoboTwin tasks concurrently across visible GPUs."""

from __future__ import annotations

import argparse
import ast
import json
import os
import queue
import shlex
import signal
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
INFERENCE_SCRIPT = SCRIPT_DIR / "inference.py"
MAX_TASKS = 50


def parse_args() -> argparse.Namespace:
    visible = os.environ.get("GPU_IDS") or os.environ.get("CUDA_VISIBLE_DEVICES")
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run all RoboTwin tasks with one evaluator process per GPU.",
    )
    parser.add_argument("checkpoint", type=Path, help="Checkpoint pretrained_model directory.")
    parser.add_argument("output_root", type=Path, help="Unique output directory for this checkpoint.")
    parser.add_argument(
        "--robotwin-root",
        type=Path,
        default=Path(os.environ.get("ROBOTWIN_ROOT", REPO_ROOT / "third_party" / "RoboTwin")),
    )
    parser.add_argument("--gpu-ids", default=visible or "0,1,2,3,4,5,6,7")
    parser.add_argument("--max-jobs-per-gpu", type=int, default=int(os.environ.get("MAX_JOBS_PER_GPU", "1")))
    parser.add_argument("--task-config", default=os.environ.get("TASK_CONFIG", "demo_randomized"))
    parser.add_argument("--start-task-idx", type=int, default=int(os.environ.get("START_TASK_IDX", "0")))
    parser.add_argument("--task-count", type=int, default=int(os.environ.get("TASK_COUNT", str(MAX_TASKS))))
    parser.add_argument("--num-episodes", type=int, default=int(os.environ.get("NUM_EPISODES", "100")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "42")))
    parser.add_argument("--stats-key", default=os.environ.get("STATS_KEY", "aloha"))
    parser.add_argument("--resize-size", type=int, default=int(os.environ.get("RESIZE_SIZE", "224")))
    parser.add_argument("--action-mode", choices=("abs", "delta"), default=os.environ.get("ACTION_MODE", "abs"))
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default=os.environ.get("DTYPE", "bfloat16"))
    parser.add_argument("--infer-horizon", type=int, default=int(os.environ.get("INFER_HORIZON", "25")))
    parser.add_argument("--inference-backend", choices=("standard", "optimized"), default=os.environ.get("INFERENCE_BACKEND", "standard"))
    parser.add_argument("--instruction-type", default=os.environ.get("INSTRUCTION_TYPE", "unseen"))
    parser.add_argument("--fps", type=int, default=int(os.environ.get("FPS", "30")))
    parser.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "INFO"))
    parser.add_argument("--max-attempts", type=int, default=int(os.environ.get("MAX_ATTEMPTS", "1")))
    parser.add_argument("--retry-delay", type=float, default=float(os.environ.get("RETRY_DELAY", "10")))
    parser.add_argument("--stagger-seconds", type=float, default=float(os.environ.get("STAGGER_SECONDS", "5")))
    parser.add_argument("--force", action=argparse.BooleanOptionalAction, default=os.environ.get("FORCE_RERUN", "false").lower() == "true")
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=os.environ.get("DRY_RUN", "false").lower() == "true")
    return parser.parse_args()


def task_names() -> list[str]:
    module = ast.parse(INFERENCE_SCRIPT.read_text(encoding="utf-8"), filename=str(INFERENCE_SCRIPT))
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "TASK_NAMES":
                    names = ast.literal_eval(node.value)
                    if isinstance(names, list) and all(isinstance(name, str) for name in names):
                        return names
    raise RuntimeError(f"Could not read TASK_NAMES from {INFERENCE_SCRIPT}")


def gpu_ids(value: str) -> list[str]:
    result = [item.strip() for item in value.replace(" ", ",").split(",") if item.strip()]
    if not result or len(set(result)) != len(result):
        raise ValueError(f"GPU_IDS must contain unique non-empty ids, got {value!r}")
    return result


def task_dir(args: argparse.Namespace, name: str) -> Path:
    return args.output_root / "robotwin" / args.task_config / name


def task_log(args: argparse.Namespace, idx: int, name: str) -> Path:
    return args.output_root / "logs" / args.task_config / f"{idx:02d}_{name}.log"


def task_done(args: argparse.Namespace, name: str) -> bool:
    directory = task_dir(args, name)
    success = len(list(directory.glob("success_*.mp4")))
    failure = len(list(directory.glob("failure_*.mp4")))
    return success + failure == args.num_episodes


def write_summary(args: argparse.Namespace, names: list[str]) -> Path:
    rows: list[tuple[int, str, int, int]] = []
    for idx, name in enumerate(names):
        directory = task_dir(args, name)
        success = len(list(directory.glob("success_*.mp4")))
        failure = len(list(directory.glob("failure_*.mp4")))
        if success + failure:
            rows.append((idx, name, success, success + failure))

    total_success = sum(row[2] for row in rows)
    total_episodes = sum(row[3] for row in rows)
    task_rates = [row[2] / row[3] for row in rows]
    average_rate = sum(task_rates) / len(task_rates) if task_rates else None
    overall_rate = total_success / total_episodes if total_episodes else None

    def percent(value: float | None) -> str:
        return "N/A" if value is None else f"{value * 100:.2f}%"

    lines = [
        f"checkpoint: {args.checkpoint}",
        f"task_config: {args.task_config}",
        f"dtype/backend: {args.dtype}/{args.inference_backend}",
        f"seed/stats: {args.seed}/{args.stats_key}",
        f"action/horizon: {args.action_mode}/{args.infer_horizon}",
        f"episodes/task: {args.num_episodes}",
        f"instruction: {args.instruction_type}",
        f"completed_tasks: {len(rows)}/{len(names)}",
        f"average_task_success_rate: {percent(average_rate)}",
        f"overall_episode_success_rate: {percent(overall_rate)}",
        f"total_success: {total_success}/{total_episodes}",
        "",
        "per_task:",
    ]
    completed_by_idx = {row[0]: row for row in rows}
    for idx, name in enumerate(names):
        row = completed_by_idx.get(idx)
        if row is None:
            lines.append(f"{idx:02d} {name}: N/A (0/0)")
        else:
            _, task_name, success, episodes = row
            lines.append(f"{idx:02d} {task_name}: {success / episodes * 100:.2f}% ({success}/{episodes})")

    summary_path = args.output_root / "summary.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def build_command(args: argparse.Namespace, idx: int, name: str) -> list[str]:
    return [
        sys.executable,
        str(INFERENCE_SCRIPT),
        "--ckpt-path", str(args.checkpoint),
        "--video-dir", str(task_dir(args, name)),
        "--task-config", args.task_config,
        "--task-idx", str(idx),
        "--instruction-type", args.instruction_type,
        "--seed", str(args.seed),
        "--stats-key", args.stats_key,
        "--resize-size", str(args.resize_size),
        "--action-mode", args.action_mode,
        "--dtype", args.dtype,
        "--num-episodes", str(args.num_episodes),
        "--fps", str(args.fps),
        "--infer-horizon", str(args.infer_horizon),
        "--inference-backend", args.inference_backend,
        "--log-level", args.log_level,
    ]


def validate(args: argparse.Namespace, names: list[str], devices: list[str]) -> None:
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.robotwin_root = args.robotwin_root.expanduser().resolve()
    if len(names) != MAX_TASKS:
        raise ValueError(f"Expected {MAX_TASKS} RoboTwin tasks, found {len(names)}")
    if args.start_task_idx < 0 or args.task_count <= 0 or args.start_task_idx + args.task_count > len(names):
        raise ValueError("Requested task range is outside the 50 RoboTwin tasks")
    if args.num_episodes <= 0 or args.max_jobs_per_gpu <= 0 or args.max_attempts <= 0:
        raise ValueError("num episodes, jobs per GPU, and max attempts must be positive")
    if args.infer_horizon <= 0 or args.resize_size <= 0 or args.retry_delay < 0 or args.stagger_seconds < 0:
        raise ValueError("invalid inference or retry setting")
    if not devices:
        raise ValueError("At least one GPU is required")
    required_checkpoint = [args.checkpoint / name for name in ("config.json", "model.safetensors", "stats.json")]
    missing = [str(path) for path in required_checkpoint if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Checkpoint must be a pretrained_model directory; missing {missing}")
    required_robotwin = [
        args.robotwin_root / "envs",
        args.robotwin_root / "description" / "utils",
        args.robotwin_root / "task_config" / f"{args.task_config}.yml",
    ]
    missing = [str(path) for path in required_robotwin if not path.exists()]
    if missing:
        raise FileNotFoundError(f"RoboTwin checkout is incomplete; missing {missing}")


def main() -> int:
    args = parse_args()
    names = task_names()
    devices = gpu_ids(args.gpu_ids)
    validate(args, names, devices)
    selected = [(idx, names[idx]) for idx in range(args.start_task_idx, args.start_task_idx + args.task_count)]
    slots = [(device, slot) for device in devices for slot in range(args.max_jobs_per_gpu)]
    print(f"checkpoint: {args.checkpoint}", flush=True)
    print(f"output: {args.output_root}", flush=True)
    print(f"robotwin: {args.robotwin_root}", flush=True)
    print(f"tasks: {selected[0][0]}-{selected[-1][0]} ({len(selected)})", flush=True)
    print(f"GPUs/slots: {','.join(devices)} / {len(slots)}", flush=True)
    print(f"config/episodes/dtype/horizon: {args.task_config}/{args.num_episodes}/{args.dtype}/{args.infer_horizon}", flush=True)
    if args.dry_run:
        for pos, (idx, name) in enumerate(selected):
            device, slot = slots[pos % len(slots)]
            print(f"GPU {device}/slot {slot}: task {idx:02d} {name}", flush=True)
        return 0

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "logs" / args.task_config).mkdir(parents=True, exist_ok=True)
    (args.output_root / "commands" / args.task_config).mkdir(parents=True, exist_ok=True)
    run_config = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint),
        "robotwin_root": str(args.robotwin_root),
        "gpu_ids": devices,
        "max_jobs_per_gpu": args.max_jobs_per_gpu,
        "task_config": args.task_config,
        "num_episodes": args.num_episodes,
        "seed": args.seed,
        "action_mode": args.action_mode,
        "dtype": args.dtype,
        "infer_horizon": args.infer_horizon,
        "tasks": [{"idx": idx, "name": name} for idx, name in selected],
    }
    (args.output_root / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")

    pending: queue.Queue[tuple[int, str]] = queue.Queue()
    for idx, name in selected:
        if not args.force and task_done(args, name):
            print(f"[skip] {idx:02d} {name}", flush=True)
        else:
            pending.put((idx, name))
    if pending.empty():
        summary_path = write_summary(args, names)
        print(f"summary: {summary_path}", flush=True)
        print("All tasks are already complete.", flush=True)
        return 0

    stop = threading.Event()
    active: dict[int, subprocess.Popen[str]] = {}
    active_lock = threading.Lock()

    def stop_all(signum: int, _frame: object) -> None:
        print(f"Received signal {signum}; stopping evaluators", flush=True)
        stop.set()
        with active_lock:
            processes = list(active.values())
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

    signal.signal(signal.SIGINT, stop_all)
    signal.signal(signal.SIGTERM, stop_all)
    base_env = os.environ.copy()
    base_env["PYTHONPATH"] = os.pathsep.join(
        item for item in (str(REPO_ROOT / "src"), str(args.robotwin_root), base_env.get("PYTHONPATH", "")) if item
    )
    base_env["ROBOTWIN_ROOT"] = str(args.robotwin_root)
    base_env["PYTHONUNBUFFERED"] = "1"
    base_env["TOKENIZERS_PARALLELISM"] = "false"
    base_env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    base_env.setdefault("OMP_NUM_THREADS", "1")
    base_env.setdefault("MKL_NUM_THREADS", "1")

    def worker(worker_id: int, device: str, slot: int) -> None:
        while not stop.is_set():
            try:
                idx, name = pending.get_nowait()
            except queue.Empty:
                return
            command = build_command(args, idx, name)
            log_path = task_log(args, idx, name)
            command_path = args.output_root / "commands" / args.task_config / f"{idx:02d}_{name}.sh"
            command_path.write_text(
                f"CUDA_VISIBLE_DEVICES={shlex.quote(device)} {shlex.join(command)}\n", encoding="utf-8"
            )
            completed = False
            for attempt in range(1, args.max_attempts + 1):
                if stop.is_set():
                    break
                with log_path.open("w" if attempt == 1 else "a", encoding="utf-8") as log_file:
                    log_file.write(
                        f"\n=== attempt {attempt}/{args.max_attempts}, GPU={device}/slot={slot}, "
                        f"{datetime.now(timezone.utc).isoformat()} ===\n{shlex.join(command)}\n"
                    )
                    log_file.flush()
                    child_env = base_env.copy()
                    child_env["CUDA_VISIBLE_DEVICES"] = device
                    try:
                        process = subprocess.Popen(
                            command,
                            cwd=args.robotwin_root,
                            env=child_env,
                            stdout=log_file,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                            text=True,
                        )
                    except OSError as exc:
                        log_file.write(f"Failed to launch: {exc}\n")
                        continue
                    with active_lock:
                        active[worker_id] = process
                    exit_code = process.wait()
                    with active_lock:
                        active.pop(worker_id, None)
                if exit_code == 0 and task_done(args, name):
                    completed = True
                    print(f"[done] GPU={device} task={idx:02d} {name}", flush=True)
                    break
                if attempt < args.max_attempts and not stop.is_set():
                    print(f"[retry] GPU={device} task={idx:02d} {name} exit={exit_code}", flush=True)
                    stop.wait(args.retry_delay)
            if not completed and not stop.is_set():
                print(f"[failed] GPU={device} task={idx:02d} {name}; log={log_path}", flush=True)
            pending.task_done()

    threads = [threading.Thread(target=worker, args=(worker_id, device, slot), daemon=False)
               for worker_id, (device, slot) in enumerate(slots[:pending.qsize()])]
    for index, thread in enumerate(threads):
        thread.start()
        if index + 1 < len(threads) and args.stagger_seconds:
            stop.wait(args.stagger_seconds)
    for thread in threads:
        thread.join()
    summary_path = write_summary(args, names)
    print(f"summary: {summary_path}", flush=True)
    return 130 if stop.is_set() else (0 if pending.empty() else 1)


if __name__ == "__main__":
    raise SystemExit(main())
