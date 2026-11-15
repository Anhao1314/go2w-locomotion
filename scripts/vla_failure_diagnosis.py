#!/usr/bin/env python3
"""用已有数据做 VLA 失败归因（不跑新评测）。

数据来源（全部已入库）：
  reports/evidence/vla-2026-11-15/libero_suite_5tasks.json   基线 100 集
  reports/evidence/vla-2026-11-16/libero_e1_n25.json         E1 100 集（同一批 task-seed）
  logs/vla/libero_video_pack2/libero_eval_summary.json       4 集带逐步轨迹（2 成 2 败）

回答三个问题：
  A. 哪些 (task, seed) 是"稳定成功/稳定失败"，哪些在两次配置间翻转？
  B. 成功率没变时，完成时间有没有变（效率维度）？
  C. 带轨迹的 4 集里，失败是"没抓起来"还是"抓起来了没送到"？

用法：python scripts/vla_failure_diagnosis.py
"""

from __future__ import annotations

import json
import pathlib
import statistics
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
EVID = ROOT / "reports" / "evidence"


def load(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def main() -> int:
    base = load(EVID / "vla-2026-11-15" / "libero_suite_5tasks.json")
    e1 = load(EVID / "vla-2026-11-16" / "libero_e1_n25.json")
    pack = load(ROOT / "logs" / "vla" / "libero_video_pack2" / "libero_eval_summary.json")

    bmap = {(r["task"], r["seed"]): r for r in base["results"]}
    emap = {(r["task"], r["seed"]): r for r in e1["results"]}
    keys = sorted(set(bmap) & set(emap))

    print("=" * 70)
    print("A. 配对稳定性（同一 task-seed 在两个配置下的结局）")
    print("=" * 70)
    always_ok = [k for k in keys if bmap[k]["success"] and emap[k]["success"]]
    always_bad = [k for k in keys if not bmap[k]["success"] and not emap[k]["success"]]
    flip_ok = [k for k in keys if not bmap[k]["success"] and emap[k]["success"]]
    flip_bad = [k for k in keys if bmap[k]["success"] and not emap[k]["success"]]
    print(f"总是成功 {len(always_ok)} | 总是失败 {len(always_bad)} | "
          f"翻转 {len(flip_ok) + len(flip_bad)}（变好 {len(flip_ok)} / 变坏 {len(flip_bad)}）"
          f" | 合计 {len(keys)}")
    print("\n逐任务：")
    print(f"{'task':>5} {'总是成功':>8} {'总是失败':>8} {'翻转':>6}")
    per_task: dict[str, dict[str, int]] = {}
    for t in sorted({k[0] for k in keys}):
        tk = [k for k in keys if k[0] == t]
        ao = sum(1 for k in tk if bmap[k]["success"] and emap[k]["success"])
        ab = sum(1 for k in tk if not bmap[k]["success"] and not emap[k]["success"])
        fl = len(tk) - ao - ab
        per_task[str(t)] = {"always_ok": ao, "always_bad": ab, "flip": fl}
        print(f"{t:>5} {ao:>8} {ab:>8} {fl:>6}")

    print()
    print("=" * 70)
    print("B. 完成时间（只看成功 episode）")
    print("=" * 70)
    b_steps = [bmap[k]["steps"] for k in keys if bmap[k]["success"]]
    e_steps = [emap[k]["steps"] for k in keys if emap[k]["success"]]
    b_mean, e_mean = mean(b_steps), mean(e_steps)
    if b_mean is not None and e_mean is not None:
        print(f"基线：n={len(b_steps)} 均值 {b_mean:.1f} 步  中位 {statistics.median(b_steps):.0f}")
        print(f"E1  ：n={len(e_steps)} 均值 {e_mean:.1f} 步  中位 {statistics.median(e_steps):.0f}")
        print(f"差值：{e_mean - b_mean:+.1f} 步")

    print()
    print("=" * 70)
    print("C. 带轨迹的 4 集：失败卡在哪一步")
    print("=" * 70)
    rows = []
    for r in pack["results"]:
        trace = r.get("trace") or {}
        grip = [float(v) for v in (trace.get("gripper_action") or [])]
        z = [float(v) for v in (trace.get("state_z") or [])]
        closed = [i for i, v in enumerate(grip) if v > 0]
        first_close = closed[0] if closed else None
        lift = None
        if first_close is not None and z:
            tail = z[first_close:]
            if tail:
                lift = max(tail) - z[first_close]
        rows.append({
            "seed": r["seed"],
            "success": r["success"],
            "steps": r["steps"],
            "closed_steps": len(closed),
            "closed_frac": round(len(closed) / max(1, len(grip)), 2),
            "close_events": sum(1 for i in range(1, len(grip))
                                if (grip[i] > 0) != (grip[i - 1] > 0)),
            "first_close_step": first_close,
            "lift_after_close_m": None if lift is None else round(lift, 3),
            "ee_z_start": round(z[0], 3) if z else None,
            "ee_z_max": round(max(z), 3) if z else None,
        })
    print(f"{'seed':>6} {'结果':>6} {'步数':>5} {'闭合步数':>8} {'闭合占比':>8} "
          f"{'开合次数':>8} {'首次闭合':>8} {'闭合后抬升(m)':>13}")
    for r in rows:
        outcome = "成功" if r["success"] else "失败"
        print(f"{r['seed']:>6} {outcome:>6} {r['steps']:>5} {r['closed_steps']:>8} "
              f"{r['closed_frac']:>8} {r['close_events']:>8} "
              f"{str(r['first_close_step']):>8} {str(r['lift_after_close_m']):>13}")

    print("\n判读：")
    for r in rows:
        if r["success"]:
            continue
        lift = r["lift_after_close_m"]
        if lift is None:
            verdict = "从未闭合夹爪 → 接近/抓取阶段失败"
        elif lift < 0.01:
            verdict = "闭合后没有抬升 → 没抓住（空夹）"
        else:
            verdict = f"闭合后抬升 {lift:.3f} m → 抓住了，失败在搬运/放置"
        print(f"  seed {r['seed']}（{r['steps']} 步）：{verdict}")

    out = {
        "paired": {
            "n": len(keys), "always_success": len(always_ok),
            "always_fail": len(always_bad),
            "flip_improved": len(flip_ok), "flip_worsened": len(flip_bad),
            "per_task": per_task,
        },
        "time_to_success": {
            "base_mean_steps": mean(b_steps), "e1_mean_steps": mean(e_steps),
            "base_n": len(b_steps), "e1_n": len(e_steps),
        },
        "traced_episodes": rows,
    }
    path = ROOT / "reports" / "analysis" / "vla_failure_diagnosis.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n写出 {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
