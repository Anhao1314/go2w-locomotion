#!/usr/bin/env python3
"""LIBERO 示范数据的技能分段分析（分层 VLA 的前置验证）。

分层 VLA 的前提：演示能被切成可识别的技能段。LIBERO 的抓放任务在动作空间里
表现为夹爪二值信号（+1 闭合 / −1 张开）的两次跳变：

    接近 approach  → [夹爪闭合] → 搬运 transport → [夹爪张开] → 释放/撤退 retreat

本脚本只读 parquet 里的动作列（不解码视频），统计：
  * 每段时长分布与占比
  * 模板匹配率（是否都符合"一次闭合 + 一次张开"）
  * 各段的动作幅值（搬运段是否真比接近段更"忙"）

用法（WSL vla-env，离线）：
    HF_HUB_OFFLINE=1 python scripts/analyze_libero_skills.py \
        --root /root/.cache/huggingface/lerobot/hub/datasets--lerobot--libero/snapshots/<rev>
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
from typing import Any

import numpy as np


def load_actions(root: pathlib.Path) -> dict[int, np.ndarray]:
    """读所有 parquet 的 (episode_index, action)，按 episode 分组。"""
    import pyarrow.parquet as pq  # pyright: ignore[reportMissingImports]

    episodes: dict[int, list[list[float]]] = {}
    files = sorted((root / "data").rglob("*.parquet"))
    for path in files:
        table = pq.read_table(path, columns=["episode_index", "action"])
        ep = table.column("episode_index").to_numpy()
        act = table.column("action").to_pylist()
        for e, a in zip(ep, act):
            episodes.setdefault(int(e), []).append([float(v) for v in a])
    return {k: np.asarray(v, dtype=np.float64) for k, v in episodes.items()}


def segment(actions: np.ndarray) -> dict[str, Any]:
    """按夹爪符号跳变切段。返回 None 表示不符合模板。"""
    grip = np.sign(actions[:, 6])
    grip[grip == 0] = 1.0
    changes = np.flatnonzero(np.diff(grip) != 0) + 1
    seg = {
        "n_frames": int(len(actions)),
        "n_transitions": int(len(changes)),
        "transition_frames": changes.tolist(),
        "starts_closed": bool(grip[0] > 0),
        "ends_closed": bool(grip[-1] > 0),
    }
    if len(changes) >= 2:
        close_at, open_at = int(changes[0]), int(changes[1])
        seg.update({
            "approach_frames": close_at,
            "transport_frames": open_at - close_at,
            "retreat_frames": len(actions) - open_at,
        })
    return seg


def phase_activity(actions: np.ndarray, a: int, b: int) -> dict[str, float]:
    if b <= a:
        return {}
    block = actions[a:b]
    return {
        "trans_mag_mean": float(np.linalg.norm(block[:, :3], axis=1).mean()),
        "rot_mag_mean": float(np.linalg.norm(block[:, 3:6], axis=1).mean()),
        "grip_closed_frac": float((block[:, 6] > 0).mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--root", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path,
                        default=pathlib.Path("logs/vla/libero_skills.json"))
    args = parser.parse_args()

    episodes = load_actions(args.root)
    print(f"[skills] episodes={len(episodes)}")

    lengths, approaches, transports, retreats = [], [], [], []
    template_ok = 0
    acts = {"approach": [], "transport": [], "retreat": []}
    for idx, actions in episodes.items():
        seg = segment(actions)
        lengths.append(seg["n_frames"])
        if seg["n_transitions"] == 2 and not seg["starts_closed"]:
            template_ok += 1
            approaches.append(seg["approach_frames"])
            transports.append(seg["transport_frames"])
            retreats.append(seg["retreat_frames"])
            for name, (a, b) in (("approach", (0, seg["approach_frames"])),
                                 ("transport", (seg["approach_frames"],
                                                seg["approach_frames"] + seg["transport_frames"])),
                                 ("retreat", (seg["approach_frames"] + seg["transport_frames"],
                                              seg["n_frames"]))):
                acts[name].append(phase_activity(actions, a, b))

    def desc(values: list[float]) -> dict[str, float]:
        if not values:
            return {}
        return {
            "n": len(values),
            "mean": statistics.fmean(values),
            "p10": float(np.percentile(values, 10)),
            "p50": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90)),
        }

    result = {
        "episodes": len(episodes),
        "template_match_rate": template_ok / max(1, len(episodes)),
        "episode_frames": desc(lengths),
        "approach_frames": desc(approaches),
        "transport_frames": desc(transports),
        "retreat_frames": desc(retreats),
        "phase_share": {
            "approach": (statistics.fmean(approaches) / statistics.fmean(lengths)
                         if approaches and lengths else None),
            "transport": (statistics.fmean(transports) / statistics.fmean(lengths)
                          if transports and lengths else None),
            "retreat": (statistics.fmean(retreats) / statistics.fmean(lengths)
                        if retreats and lengths else None),
        },
        "phase_activity": {
            name: {
                key: statistics.fmean([a[key] for a in blocks if key in a])
                for key in ("trans_mag_mean", "rot_mag_mean", "grip_closed_frac")
            }
            for name, blocks in acts.items() if blocks
        },
    }

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
