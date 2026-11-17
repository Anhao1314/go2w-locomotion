#!/usr/bin/env python3
"""预计算 LIBERO 示范的逐帧技能标签（skill-conditioning 的数据侧）。

分段规则与 analyze_libero_skills.py 完全一致：夹爪"开→合→开"且首帧张开的
episode 视为模板匹配（上次实测 1170/1693 = 69%），
    approach = [0, t1)      t1 = 第一次闭合
    transport = [t1, t2)    t2 = 释放
    retreat  = [t2, end)
非模板 episode 全帧标 unknown（训练时保持原任务串，不强行猜）。

输出：logs/vla/skill_labels.json（紧凑格式，每 episode 记 t1/t2，逐帧标签由
sitecustomize 在加载时按帧号现算，27 万帧不落盘）。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import pyarrow.parquet as pq  # pyright: ignore[reportMissingImports]


def find_dataset_root(explicit: str | None) -> pathlib.Path:
    if explicit:
        return pathlib.Path(explicit)
    base = pathlib.Path("/root/.cache/huggingface/lerobot/hub/datasets--lerobot--libero")
    snaps = sorted((base / "snapshots").glob("*/meta/info.json"))
    if not snaps:
        raise SystemExit("no LIBERO dataset snapshot found under HF cache")
    return snaps[0].parent.parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None)
    parser.add_argument("--out", default="logs/vla/skill_labels.json")
    args = parser.parse_args()

    root = find_dataset_root(args.root)
    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        raise SystemExit(f"no parquet under {root/'data'}")

    episodes: dict[int, list[list[float]]] = {}
    for path in files:
        table = pq.read_table(path, columns=["episode_index", "action"])
        for ep, act in zip(table.column("episode_index").to_numpy(),
                           table.column("action").to_pylist()):
            episodes.setdefault(int(ep), []).append([float(v) for v in act])

    labels: dict[str, dict[str, object]] = {}
    n_template = 0
    for ep, acts in episodes.items():
        grip = [1.0 if a[6] > 0 else -1.0 for a in acts]
        changes = [i for i in range(1, len(grip)) if grip[i] != grip[i - 1]]
        if len(changes) == 2 and grip[0] < 0:
            t1, t2 = changes[0], changes[1]
            labels[str(ep)] = {"t1": t1, "t2": t2, "frames": len(acts)}
            n_template += 1
        else:
            labels[str(ep)] = {"t1": None, "t2": None, "frames": len(acts)}

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_root": str(root),
        "episodes": len(episodes),
        "template_episodes": n_template,
        "template_rate": round(n_template / max(1, len(episodes)), 4),
        "rule": "gripper open->close->open; approach=[0,t1) transport=[t1,t2) retreat=[t2,end); else unknown",
        "labels": labels,
    }
    out.write_text(json.dumps(payload), encoding="utf-8")
    print(f"[labels] episodes={len(episodes)} template={n_template} "
          f"({n_template / max(1, len(episodes)):.1%}) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
