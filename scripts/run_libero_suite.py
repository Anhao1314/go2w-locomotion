#!/usr/bin/env python3
"""并行跑 LIBERO 评测套件（多任务 × 多种子）。

耗时分解（profile_libero_eval.py，WSL 实测）：
    env.step 111 ms/步（物理子步为主，分辨率几乎无影响）
    策略前向 220 ms / 每 50 步
    → 单集 280 步 ≈ 34 s；100 集单进程 ≈ 57 分钟

因此加速靠**多进程**：每集是独立 env + 独立随机种子，天然可并行；
CPU 14 核 / 20 线程，物理步进是 CPU 受限。本脚本把 (task, seed) 分片，
每片一个子进程，最后合并结果。GPU 只跑策略前向，显存足够 3–4 进程。

用法（WSL，vla-env 的 python）：
    python scripts/run_libero_suite.py \
        --suite libero_spatial --tasks 0,1,2,3,4 \
        --seeds 1000-1019 --workers 4 \
        --out logs/vla/libero_suite_5tasks
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time
from typing import Any

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


def parse_int_list(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return out


def shard(items: list[int], n: int) -> list[list[int]]:
    """把 seeds 轮流分给 n 个分片（均衡每片的工作量）。"""
    return [items[i::n] for i in range(n)]


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--checkpoint", type=pathlib.Path,
                        default=PROJECT_ROOT / "logs/vla/libero_smolvla/checkpoints/060000/pretrained_model")
    parser.add_argument("--suite", type=str, default="libero_spatial")
    parser.add_argument("--tasks", type=str, default="0,1,2,3,4")
    parser.add_argument("--seeds", type=str, default="1000-1019")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--time-budget-s", type=float, default=14.0)
    parser.add_argument("--out", type=pathlib.Path,
                        default=PROJECT_ROOT / "logs/vla/libero_suite")
    args = parser.parse_args()

    tasks = parse_int_list(args.tasks)
    seeds = parse_int_list(args.seeds)
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    jobs: list[tuple[int, list[int], pathlib.Path]] = []
    for task in tasks:
        for k, part in enumerate(shard(seeds, args.workers)):
            if part:
                jobs.append((task, part, out_dir / f"task{task}_shard{k}"))

    print(f"[suite] tasks={tasks} seeds={len(seeds)} shards={len(jobs)} "
          f"workers={args.workers}", flush=True)

    started = time.time()
    # 真正的进程池：最多 workers 个存活进程。一次性起全部会把宿主内存打爆
    # （实测 20 个进程同时加载模型 → Errno 12 + OOM killer）。
    pending = list(jobs)
    running: list[tuple[subprocess.Popen[bytes], pathlib.Path, Any]] = []
    while pending or running:
        while pending and len(running) < args.workers:
            task, part, job_dir = pending.pop(0)
            cmd = [
                sys.executable, "-u", str(PROJECT_ROOT / "vla" / "evaluate_libero.py"),
                "--checkpoint", str(args.checkpoint),
                "--suite", args.suite,
                "--task-ids", str(task),
                "--seeds", ",".join(str(s) for s in part),
                "--fps", str(args.fps),
                "--time-budget-s", str(args.time_budget_s),
                "--out", str(job_dir),
            ]
            log = (log_dir / f"task{task}_{job_dir.name}.log").open("wb")
            running.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT),
                            job_dir, log))
        time.sleep(1.0)
        still = []
        for proc, job_dir, log in running:
            code = proc.poll()
            if code is None:
                still.append((proc, job_dir, log))
            else:
                log.close()
                if code != 0:
                    print(f"[suite] FAILED {job_dir.name} exit={code}", flush=True)
        running = still

    elapsed = time.time() - started

    per_task: dict[str, dict[str, Any]] = {}
    merged: list[dict] = []
    for task in tasks:
        rows = []
        for job_dir in sorted(out_dir.glob(f"task{task}_shard*")):
            summary_path = job_dir / "libero_eval_summary.json"
            if not summary_path.exists():
                continue
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            rows.extend(data["results"])
        if not rows:
            continue
        ok = sum(1 for r in rows if r["success"])
        per_task[str(task)] = {
            "episodes": len(rows),
            "successes": ok,
            "success_rate": ok / len(rows),
            "mean_steps": sum(r["steps"] for r in rows) / len(rows),
            "success_steps_mean": (sum(r["steps"] for r in rows if r["success"]) / ok) if ok else None,
        }
        merged.extend({"task": task, **r} for r in rows)

    total_ok = sum(1 for r in merged if r["success"])
    summary = {
        "suite": args.suite,
        "tasks": tasks,
        "seeds": seeds,
        "workers": args.workers,
        "fps": args.fps,
        "time_budget_s": args.time_budget_s,
        "episodes": len(merged),
        "successes": total_ok,
        "success_rate": total_ok / max(1, len(merged)),
        "per_task": per_task,
        "wall_time_s": round(elapsed, 1),
        "seconds_per_episode": round(elapsed / max(1, len(merged)), 1),
        "results": merged,
    }
    (out_dir / "suite_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({k: v for k, v in summary.items() if k != "results"},
                     indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
