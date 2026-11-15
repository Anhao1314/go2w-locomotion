#!/usr/bin/env python3
"""因子体检：判断哪些因子能支撑决策、哪些只是摆设。

五问：
  1. 覆盖    — 有多少行真的产出了这个因子
  2. 退化    — 方差为 0 的因子无法区分任何情况（组内退化尤其致命：说明该指标在该 run 上
               没有分辨率，例如所有 episode 步数完全相同）
  3. 冗余    — 两两相关 ≥0.95：区分"构造别名"（数学上完全由另一个决定）、
               "结果同源"（都由结局决定）、"真冗余"
  4. 判别力  — 只在**同一 run 内**估计效应量（Cohen's d）。跨 run 混算会把 run 间差异
               误读成因子作用（辛普森悖论）；组内样本不足时明确标"不可估"
  5. 杠杆    — 因子变差时对应哪个可干预项；没有杠杆的只能当诊断

输入：reports/analysis/factors_{run,rollout}_<date>.csv
输出：reports/analysis/audit_<date>.{md,json}

用法：python analysis/audit.py [--date 2026-11-15]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import statistics
import sys
from datetime import date as _date
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.factors import BY_NAME, catalog  # noqa: E402

ANALYSIS_DIR = ROOT / "reports" / "analysis"

# 由"结局"直接决定的因子：彼此高相关是结构性的，不属于冗余缺陷
OUTCOME_LIKE = {"task_success", "stab_fall", "task_stage_reached",
                "task_terminal_miss_m", "task_miss_ratio", "vla_closed_success_rate"}
# 构造别名：数学上完全由另一个因子决定
ALIAS_OF = {"task_time_s": "task_steps"}

# 因子 → 可干预的杠杆。空字符串 = 只能看不能改，不要拿它做决策依据。
LEVERS: dict[str, str] = {
    "track_vx_rmse": "速度跟踪奖励权重 / 指令滤波",
    "track_wz_rmse": "转向奖励权重 / 角速度限幅",
    "track_vx_bias": "执行器饱和检查 / 指令前馈",
    "stab_fall": "训练分布（推力/地形 DR）",
    "stab_recovery_time_s": "恢复奖励 / 步态切换时机",
    "stab_fall_after_push_s": "推力 DR 档位 / 恢复策略",
    "task_steps": "阶段命令时长 / 速度上限",
    "task_time_s": "同 task_steps（构造别名）",
    "task_terminal_miss_m": "停止容差 / 终点收敛奖励",
    "task_miss_ratio": "停止容差 / 终点收敛奖励",
    "task_stage_reached": "阶段机容差 / 阶段命令",
    "train_smp_per_s": "分辨率 / chunk 长度 / bf16（加 batch 无效）",
    "train_plateau_ratio": "停训阈值 / 换数据 / 换超参",
    "train_loss_decay_per_10k": "同上",
    "train_mem_gb": "批大小 / 精度 / 梯度检查点",
    "vla_ol_mse_ratio": "数据质量 / 损失加权",
    "vla_ol_mae_gripper_closed": "搬运段样本加权 / 缩短动作块",
    "vla_ol_vision_delta": "视觉编码器 / 相机配置",
    "vla_closed_success_rate": "结果指标（不做干预对象）",
    "vla_control_freq_ratio": "评测配置（必须显式指定 fps）",
    "gen_fixed_overhead_steps": "阶段切换逻辑 / 起停时长",
    "gen_per_metre_steps": "巡航速度命令",
    "gen_turn_cost_per_deg": "转向命令时长",
}


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def load_csv(path: pathlib.Path) -> tuple[list[str], list[dict[str, Any]]]:
    if not path.exists():
        return [], []
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        names = [n for n in (reader.fieldnames or []) if n not in ("run_id", "entity")]
        rows: list[dict[str, Any]] = []
        for raw in reader:
            row: dict[str, Any] = {"_run_id": raw.get("run_id")}
            for name in names:
                value = raw.get(name, "")
                row[name] = None if value in ("", None) else float(value)
            rows.append(row)
    return names, rows


def values(rows: list[dict[str, Any]], name: str) -> list[float]:
    return [float(r[name]) for r in rows if r.get(name) is not None]


def cohen_d(a: list[float], b: list[float]) -> float | None:
    if len(a) < 2 or len(b) < 2:
        return None
    va, vb = statistics.pvariance(a), statistics.pvariance(b)
    pooled = math.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
    if pooled < 1e-12:
        return None
    return (statistics.fmean(a) - statistics.fmean(b)) / pooled


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 10:
        return None
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx < 1e-12 or sy < 1e-12:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def within_run_degenerate(rows: list[dict[str, Any]], name: str,
                          min_n: int = 5) -> list[str]:
    groups: dict[str, list[float]] = {}
    for r in rows:
        if r.get(name) is None:
            continue
        groups.setdefault(str(r.get("_run_id")), []).append(float(r[name]))
    return sorted(rid for rid, vals in groups.items()
                  if len(vals) >= min_n and statistics.pstdev(vals) < 1e-12)


def pair_kind(a: str, b: str) -> str:
    if a in ALIAS_OF or b in ALIAS_OF:
        return "构造别名"
    if a in OUTCOME_LIKE and b in OUTCOME_LIKE:
        return "结果同源"
    return "真冗余"


def redundancy(rows: list[dict[str, Any]], names: list[str],
               threshold: float = 0.95) -> list[tuple[str, str, float, int, str]]:
    pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            both = [(r[a], r[b]) for r in rows
                    if r.get(a) is not None and r.get(b) is not None]
            if len(both) < 10:
                continue
            r = pearson([p[0] for p in both], [p[1] for p in both])
            if r is not None and abs(r) >= threshold:
                pairs.append((a, b, r, len(both), pair_kind(a, b)))
    return sorted(pairs, key=lambda p: -abs(p[2]))


# --------------------------------------------------------------------------- #
# 判别力：优先组内估计，避免跨 run 混杂
# --------------------------------------------------------------------------- #
def discrimination(rows: list[dict[str, Any]], names: list[str],
                   outcome: str) -> dict[str, dict[str, Any]]:
    by_run: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_run.setdefault(str(r.get("_run_id")), []).append(r)

    out: dict[str, dict[str, Any]] = {}
    for name in names:
        if name == outcome:
            continue
        within: list[float] = []
        used_runs: list[str] = []
        yes_all: list[float] = []
        no_all: list[float] = []
        for run_id, rs in by_run.items():
            yes = [float(r[name]) for r in rs
                   if r.get(name) is not None and r.get(outcome) == 1.0]
            no = [float(r[name]) for r in rs
                  if r.get(name) is not None and r.get(outcome) == 0.0]
            yes_all.extend(yes)
            no_all.extend(no)
            if len(yes) >= 2 and len(no) >= 2:
                d = cohen_d(yes, no)
                if d is not None:
                    within.append(d)
                    used_runs.append(run_id)
        out[name] = {
            "d_within": statistics.fmean(within) if within else None,
            "runs_used": used_runs,
            "d_pooled": cohen_d(yes_all, no_all),
            "n_outcome_1": len(yes_all),
            "n_outcome_0": len(no_all),
            "confounded": not within,
        }
    return out


# --------------------------------------------------------------------------- #
def audit(rows: list[dict[str, Any]], names: list[str]) -> dict[str, Any]:
    total = len(rows)
    per_factor: dict[str, Any] = {}
    for name in names:
        vals = values(rows, name)
        unique = len({round(v, 6) for v in vals})
        std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        per_factor[name] = {
            "coverage": len(vals) / total if total else 0.0,
            "n": len(vals),
            "std": std,
            "unique": unique,
            "constant": len(vals) > 1 and unique <= 1,
            "binary": unique == 2,
            "min": min(vals) if vals else None,
            "max": max(vals) if vals else None,
            "lever": LEVERS.get(name, ""),
            "registered": name in BY_NAME,
            "within_run_dead": within_run_degenerate(rows, name),
            "outcome_like": name in OUTCOME_LIKE,
        }
    return {
        "rows": total,
        "factors": per_factor,
        "redundant_pairs": redundancy(rows, names),
        "discrimination_success": discrimination(rows, names, "task_success"),
        "discrimination_fall": discrimination(rows, names, "stab_fall"),
    }


def verdict(name: str, info: dict[str, Any], redundant: set[str]) -> str:
    if info["coverage"] < 0.02:
        return "无数据"
    if info["outcome_like"]:
        return "结局指标"
    if info["constant"]:
        return "退化（常数）"
    if len(info["within_run_dead"]) >= 3 and not info["binary"]:
        return "组内无分辨率"
    if name in redundant:
        return "冗余（保留一个）"
    if not info["lever"]:
        return "诊断项"
    return "可用"


def _fmt(value: Any, nd: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{nd}f}"
    return str(value)


# --------------------------------------------------------------------------- #
def build_md(result: dict[str, Any], run_result: dict[str, Any]) -> str:
    lines: list[str] = []
    add = lines.append
    add("# 因子体检报告（自动生成）\n")
    add(f"- rollout 表 {result['rows']} 行；run 表 {run_result['rows']} 行")
    add("- 判定口径：覆盖 <2% 无数据；结局指标单列；常数=退化；组内退化 ≥3 个 run = "
        "组内无分辨率；|r|≥0.95 冗余；无杠杆=诊断项")
    add("- **判别力只在同一 run 内估计**；组内样本不足时标「不可估」，不给跨 run 的混杂数字\n")

    redundant_names = {n for pair in result["redundant_pairs"] for n in pair[:2]}

    add("## 一、rollout 级因子体检\n")
    add("| 因子 | 覆盖 | n | 标准差 | 唯一值 | 组内退化 run | 杠杆 | 判定 |")
    add("|---|---|---|---|---|---|---|---|")
    for name, info in result["factors"].items():
        add(f"| `{name}` | {info['coverage']:.0%} | {info['n']} | {info['std']:.4g} | "
            f"{info['unique']} | {len(info['within_run_dead'])} | "
            f"{info['lever'] or '—'} | {verdict(name, info, redundant_names)} |")
    add("")

    add("## 二、组内退化明细（指标没有分辨率）\n")
    any_dead = False
    for name, info in result["factors"].items():
        if info["within_run_dead"]:
            any_dead = True
            add(f"- `{name}`：{', '.join(info['within_run_dead'])}")
    if not any_dead:
        add("无。")
    add("")

    add("## 三、冗余因子对\n")
    if result["redundant_pairs"]:
        add("| 因子 A | 因子 B | 相关系数 | n | 性质 | 建议 |")
        add("|---|---|---|---|---|---|")
        for a, b, r, n, kind in result["redundant_pairs"]:
            advice = {"构造别名": "删掉其中一个",
                      "结果同源": "说明结局只有一维，需补独立结局指标",
                      "真冗余": "保留语义更直接的一个"}[kind]
            add(f"| `{a}` | `{b}` | {r:+.3f} | {n} | {kind} | {advice} |")
    else:
        add("未发现 |r| ≥ 0.95 的因子对。")
    add("")

    for title, key, outcome in (("四、成功判别力（outcome = task_success）",
                                 "discrimination_success", "task_success"),
                                ("五、摔倒判别力（outcome = stab_fall）",
                                 "discrimination_fall", "stab_fall")):
        add(f"## {title}\n")
        add("| 因子 | 组内 d | 可用 run | pooled d | n(1/0) | 备注 |")
        add("|---|---|---|---|---|---|")
        rows_shown = 0
        for name, info in result[key].items():
            if info["n_outcome_1"] == 0 or info["n_outcome_0"] == 0:
                continue
            rows_shown += 1
            note = "跨 run 混杂，不可用" if info["confounded"] else "组内估计"
            add(f"| `{name}` | {_fmt(info['d_within'])} | "
                f"{', '.join(info['runs_used']) if info['runs_used'] else '—'} | "
                f"{_fmt(info['d_pooled'])} | {info['n_outcome_1']}/{info['n_outcome_0']} | {note} |")
        if rows_shown == 0:
            add("| — | — | — | — | — | 没有同时含两组样本的因子 |")
        add("")

    add("## 六、run 级因子体检\n")
    add("| 因子 | 覆盖 | n | 标准差 | 唯一值 | 杠杆 | 判定 |")
    add("|---|---|---|---|---|---|---|")
    for name, info in run_result["factors"].items():
        add(f"| `{name}` | {info['coverage']:.0%} | {info['n']} | {info['std']:.4g} | "
            f"{info['unique']} | {info['lever'] or '—'} | "
            f"{verdict(name, info, set())} |")
    add("")

    cat = catalog()
    add("## 七、待补埋点（无法计算）\n")
    add("| 因子 | 需要的遥测 |")
    add("|---|---|")
    for group in cat["groups"].values():
        for item in group:
            if not item["available"]:
                add(f"| `{item['name']}` | {item['source']} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--date", default=_date.today().isoformat())
    args = parser.parse_args(argv)

    rollout_names, rollout_rows = load_csv(
        ANALYSIS_DIR / f"factors_rollout_{args.date}.csv")
    run_names, run_rows = load_csv(ANALYSIS_DIR / f"factors_run_{args.date}.csv")
    if not rollout_rows and not run_rows:
        print("[audit] 没找到因子表，先跑 python analysis/extract.py")
        return 1

    result = audit(rollout_rows, rollout_names)
    run_result = audit(run_rows, run_names) if run_rows else {"rows": 0, "factors": {}}
    (ANALYSIS_DIR / f"audit_{args.date}.json").write_text(
        json.dumps({"rollout": result, "run": run_result}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    (ANALYSIS_DIR / f"audit_{args.date}.md").write_text(
        build_md(result, run_result), encoding="utf-8")

    redundant_names = {n for pair in result["redundant_pairs"] for n in pair[:2]}
    buckets: dict[str, list[str]] = {}
    for name, info in result["factors"].items():
        buckets.setdefault(verdict(name, info, redundant_names), []).append(name)
    print(f"[audit] rollout={result['rows']} 行 / {len(result['factors'])} 因子")
    for label, names in sorted(buckets.items()):
        print(f"[audit] {label}: {names}")
    print("[audit] 冗余对: "
          f"{[(a, b, round(r, 3), k) for a, b, r, _, k in result['redundant_pairs']]}")
    print(f"[audit] 写出 {ANALYSIS_DIR}/audit_{args.date}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
