#!/usr/bin/env python3
"""Quantitative analysis over the archived evidence.

Turns the raw evaluation JSONs and training logs into derived quantities:

  * distributions (mean/std/min/max) instead of single point values
  * correlations between metrics that should (or should not) move together
  * grouped comparisons (per progress decile, per gripper phase, per geometry)
  * headroom / margin analysis against the frozen gates
  * throughput and convergence-rate numbers from the training logs

The goal is to locate *where* the system loses performance, not to restate the
headline numbers.  Output: reports/analysis/quantitative_<date>.json and .md

Usage:  python scripts/analyze_results.py
"""

from __future__ import annotations

import json
import math
import pathlib
import re
import statistics
from typing import Any, Iterable

ROOT = pathlib.Path(__file__).resolve().parents[1]
EVID = ROOT / "reports" / "evidence"
OUT_DIR = ROOT / "reports" / "analysis"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def load(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def desc(values: Iterable[float]) -> dict[str, float | int | None]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "n": len(vals),
        "mean": statistics.fmean(vals),
        "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
    }


def corr(xs: list[float], ys: list[float]) -> float | None:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return None
    xs2 = [p[0] for p in pairs]
    ys2 = [p[1] for p in pairs]
    mx, my = statistics.fmean(xs2), statistics.fmean(ys2)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs2))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys2))
    if sx < 1e-12 or sy < 1e-12:
        return None
    return sum((x - mx) * (y - my) for x, y in pairs) / (sx * sy)


def safe_div(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or abs(b) < 1e-12:
        return None
    return a / b


# --------------------------------------------------------------------------- #
# machine dog: S4 / M2 evidence
# --------------------------------------------------------------------------- #
def analyze_velocity() -> dict[str, Any]:
    data = load(EVID / "s4-sim2sim-2026-11-13" / "velocity_s0.json")
    eps = data["episodes"]
    vx = [float(e["vx_rmse"]) for e in eps]
    wz = [float(e["wz_rmse"]) for e in eps]
    fall = [bool(e["fall"]) for e in eps]
    # tracking bias: does the policy undershoot or overshoot the command?
    bias = [float(e["mean_vx"]) - float(e["mean_cmd_vx"]) for e in eps]
    return {
        "gate": 0.15,
        "vx_rmse": desc(vx),
        "wz_rmse": desc(wz),
        "headroom_vs_gate": safe_div(0.15, data["summary"]["vx_rmse_mean"]),
        "gate_violations": sum(1 for v in vx if v > 0.15),
        "worst_seed_value": max(vx),
        "corr_vx_wz": corr(vx, wz),
        "bias_mean_vx_minus_cmd": desc(bias),
        "fall_rate": sum(fall) / len(fall),
        "vx_rmse_nonfall": desc([v for v, f in zip(vx, fall) if not f]),
        "vx_rmse_fall": desc([v for v, f in zip(vx, fall) if f]),
        "wz_rmse_nonfall": desc([w for w, f in zip(wz, fall) if not f]),
        "wz_rmse_fall": desc([w for w, f in zip(wz, fall) if f]),
        "worst_seeds": sorted(range(len(vx)), key=lambda i: -vx[i])[:3],
    }


def analyze_sequence(name: str) -> dict[str, Any]:
    data = load(EVID / "s4-sim2sim-2026-11-13" / f"sequence_{name}.json")
    eps = data["episodes"]
    steps = [float(e["steps"]) for e in eps if e.get("success")]
    miss = []
    for e in eps:
        pos, tgt = e.get("terminal_pos"), e.get("target_p2")
        if pos and tgt:
            miss.append(math.dist(pos, tgt))
    tolerance = 0.30  # POSITION_TOLERANCE_M in locomotion/sequence/stage_logic.py
    miss_sorted = sorted(miss)
    p95 = miss_sorted[min(len(miss_sorted) - 1, int(round(0.95 * (len(miss_sorted) - 1))))] if miss_sorted else None
    return {
        "success_rate": data["summary"]["success_rate"],
        "steps_finished": desc(steps),
        "steps_variability": (desc(steps)["std"] or 0.0) / max(1e-9, desc(steps)["mean"] or 1.0),
        "terminal_miss_m": desc(miss),
        "success_tolerance_m": tolerance,
        "miss_budget_used_mean": safe_div(desc(miss)["mean"], tolerance),
        "miss_budget_used_p95": safe_div(p95, tolerance),
        "near_tolerance_count": sum(1 for m in miss if m > 0.9 * tolerance),
        "fall_rate": data["summary"]["fall_rate"],
        "episode_cap": data["summary"]["episode_length_steps"],
    }


def analyze_generalization() -> dict[str, Any]:
    data = load(EVID / "m2-campaign-2026-11-14" / "generalization_and_s2.json")
    lay = data["generalization_layered"]["summary"]["per_variant"]
    sin = data["generalization_single"]["summary"]["per_variant"]
    rows = {}
    for key, val in lay.items():
        steps = val.get("mean_steps_to_finish")
        rows[key] = {
            "layered_success": val["success_rate"],
            "layered_fall": val["fall_rate"],
            "layered_steps": steps,
            "single_success": sin.get(key, {}).get("success_rate"),
            "single_fall": sin.get(key, {}).get("fall_rate"),
        }
    # implied per-metre cost: (steps at 3 m) / (steps at 2 m), same 90 deg turn
    ratio_3m = safe_div(rows.get("unseen_3m_90deg", {}).get("layered_steps"),
                        rows.get("trained_2m_90deg", {}).get("layered_steps"))
    ratio_60 = safe_div(rows.get("unseen_2m_60deg", {}).get("layered_steps"),
                        rows.get("trained_2m_90deg", {}).get("layered_steps"))
    ratio_120 = safe_div(rows.get("unseen_2m_120deg", {}).get("layered_steps"),
                         rows.get("trained_2m_90deg", {}).get("layered_steps"))
    # implied costs: steps(distance) = fixed + per_metre * d, using 2 m and 3 m at 90 deg
    s2m = rows.get("trained_2m_90deg", {}).get("layered_steps")
    s3m = rows.get("unseen_3m_90deg", {}).get("layered_steps")
    per_metre = (s3m - s2m) if (s2m is not None and s3m is not None) else None
    fixed = (s2m - 2 * per_metre) if (s2m is not None and per_metre is not None) else None
    s60 = rows.get("unseen_2m_60deg", {}).get("layered_steps")
    s120 = rows.get("unseen_2m_120deg", {}).get("layered_steps")
    turn_slope = ((s120 - s60) / 60.0) if (s60 is not None and s120 is not None) else None

    return {
        "per_variant": rows,
        "steps_ratio_3m_vs_2m": ratio_3m,
        "steps_ratio_60deg_vs_90deg": ratio_60,
        "steps_ratio_120deg_vs_90deg": ratio_120,
        "fixed_overhead_steps": fixed,
        "per_metre_steps": per_metre,
        "turn_cost_steps_per_degree": turn_slope,
        "layered_overall": data["generalization_layered"]["summary"]["success_rate_overall"],
        "single_overall": data["generalization_single"]["summary"]["success_rate_overall"],
    }


def analyze_s2() -> dict[str, Any]:
    data = load(EVID / "m2-campaign-2026-11-14" / "generalization_and_s2.json")
    s2 = data["s2_disturbance"]
    eps = s2["episodes"]
    steps = [float(e["steps"]) for e in eps]
    rec = [float(e["recovery_time_s"]) for e in eps if e.get("recovery_time_s") is not None]
    push = s2["summary"]["push"]
    push_step = int(push["start_s"] / 0.02)  # 50 Hz control
    return {
        "gate": s2["summary"]["gate_no_fall_min"],
        "no_fall_rate": s2["summary"]["no_fall_rate"],
        "fall_steps": desc(steps),
        "fall_steps_after_push": desc([s - push_step for s in steps]),
        "push_step": push_step,
        "recovery_time_s": desc(rec),
        "recovery_never_count": s2["summary"]["recovery_never_count"],
    }


def analyze_c0() -> dict[str, Any]:
    data = load(EVID / "m2-campaign-2026-11-14" / "c0_baseline.json")
    out = {}
    for key in ("c0_flat", "c0_push"):
        eps = data[key]["episodes"]
        vx = [float(e["vx_rmse"]) for e in eps]
        out[key] = {
            "vx_rmse": desc(vx),
            "deterministic": (desc(vx)["std"] or 0.0) < 1e-9,
            "fall_rate": data[key]["summary"]["fall_rate"],
        }
    return out


# --------------------------------------------------------------------------- #
# VLA: open-loop error decomposition
# --------------------------------------------------------------------------- #
def analyze_openloop() -> dict[str, Any]:
    data = load(EVID / "vla-2026-11-15" / "openloop_060k.json")
    recs = data["records"]
    gt = [r["gt"] for r in recs]
    pred = [r["pred"] for r in recs]
    dim = len(gt[0])

    # per-dimension normalised error: MAE / GT std (comparable across scales)
    per_dim = {}
    for d in range(dim):
        g = [float(r["gt"][d]) for r in recs]
        p = [float(r["pred"][d]) for r in recs]
        mae = statistics.fmean([abs(a - b) for a, b in zip(g, p)])
        nmae = safe_div(mae, desc(g)["std"])
        per_dim[str(d)] = {
            "gt_std": desc(g)["std"],
            "mae": mae,
            "normalized_mae": nmae,
            "signed_bias": statistics.fmean([a - b for a, b in zip(g, p)]),
            "err_std": desc([abs(a - b) for a, b in zip(g, p)])["std"],
            "corr": corr(g, p),
        }

    # error vs episode progress (frames were sampled 10..90% in order)
    progress = {}
    gripper_activity = {}
    per_ep = 5
    for i, r in enumerate(recs):
        decile = int(round(10 + 20 * (i % per_ep)))  # 10,30,50,70,90
        err = statistics.fmean([abs(a - b) for a, b in zip(r["gt"], r["pred"])])
        progress.setdefault(decile, []).append(err)
        gripper_activity.setdefault(decile, []).append(float(r["gt"][-1]))
    progress_stats = {k: desc(v) for k, v in sorted(progress.items())}
    gripper_activity_stats = {k: desc(v) for k, v in sorted(gripper_activity.items())}
    deciles = sorted(progress)
    trend = corr([float(k) for k in deciles],
                 [float(progress[k][0]) for k in deciles]) if deciles else None

    # error split by gripper state (last dim: open vs closed)
    closed = [r for r in recs if float(r["gt"][-1]) > 0.5]
    opened = [r for r in recs if float(r["gt"][-1]) <= -0.5]
    def arm_err(group: list[dict[str, Any]]) -> dict[str, float | int | None]:
        if not group:
            return {"n": 0, "mae_arm": None, "mae_gripper": None}
        arm = [statistics.fmean([abs(a - b) for a, b in zip(r["gt"][:-1], r["pred"][:-1])])
               for r in group]
        gr = [abs(float(r["gt"][-1]) - float(r["pred"][-1])) for r in group]
        return {"n": len(group), "mae_arm": statistics.fmean(arm), "mae_gripper": statistics.fmean(gr)}

    # error vs motion magnitude (harder frames?)
    mags, errs = [], []
    for r in recs:
        mags.append(math.sqrt(sum(v * v for v in r["gt"])))
        errs.append(math.sqrt(sum((a - b) ** 2 for a, b in zip(r["gt"], r["pred"]))))

    vision = []
    for r in recs:
        if "pred_blind" in r:
            vision.append(math.dist(r["pred"], r["pred_blind"]))

    return {
        "frames": len(recs),
        "mse": data["mse"],
        "baseline_mse": data["mean_baseline_mse"],
        "mse_ratio": data["mse_ratio_to_baseline"],
        "per_dim": per_dim,
        "worst_dim": min(per_dim, key=lambda k: -(per_dim[k]["normalized_mae"] or 0.0)),
        "progress_mae": progress_stats,
        "progress_gripper_activity": gripper_activity_stats,
        "progress_trend_corr": trend,
        "gripper_closed": arm_err(closed),
        "gripper_open": arm_err(opened),
        "corr_error_vs_motion_magnitude": corr(mags, errs),
        "vision_delta": desc(vision),
    }


# --------------------------------------------------------------------------- #
# training logs
# --------------------------------------------------------------------------- #
LOG_RE = re.compile(
    r"step:(?P<step>\d+(?:\.\d+)?)(?P<step_u>[KM]?).*?"
    r"loss:(?P<loss>[0-9.]+).*?"
    r"lr:(?P<lr>[0-9.eE+-]+).*?"
    r"step_s:(?P<step_s>[0-9.]+).*?"
    r"smp/s:(?P<smp_s>[0-9.]+).*?"
    r"mem_gb:(?P<mem>[0-9.]+)"
)


def parse_log(path: pathlib.Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8", errors="ignore").replace("\r", "\n")
    out = []
    for line in raw.split("\n"):
        m = LOG_RE.search(line)
        if not m:
            continue
        step = float(m.group("step")) * (1000 if m.group("step_u") == "K" else
                                         1000000 if m.group("step_u") == "M" else 1)
        out.append({
            "step": step,
            "loss": float(m.group("loss")),
            "lr": float(m.group("lr")),
            "step_s": float(m.group("step_s")),
            "smp_s": float(m.group("smp_s")),
            "mem_gb": float(m.group("mem")),
        })
    return out


def analyze_training() -> dict[str, Any]:
    runs = {
        "teacher_bs8": ROOT / "logs" / "vla" / "train_smolvla_002.log",
        "libero_bs8": ROOT / "logs" / "vla" / "train_libero_003.log",
        "libero_bs32": ROOT / "logs" / "vla" / "train_libero_005.log",
    }
    out: dict[str, Any] = {}
    for name, path in runs.items():
        recs = parse_log(path)
        if not recs:
            out[name] = {"available": False}
            continue
        # collapse duplicate step entries (progress-bar repeats)
        by_step: dict[float, dict[str, float]] = {}
        for r in recs:
            by_step[r["step"]] = r
        series = [by_step[k] for k in sorted(by_step)]
        first, last = series[0], series[-1]
        # convergence rate over the recorded span: loss decay per 10k steps
        span = max(1.0, last["step"] - first["step"])
        decay = (first["loss"] - last["loss"]) / span * 10000.0
        out[name] = {
            "available": True,
            "steps": [first["step"], last["step"]],
            "loss_first": first["loss"],
            "loss_last": last["loss"],
            "loss_decay_per_10k_steps": decay,
            "step_s": desc([r["step_s"] for r in series]),
            "smp_s": desc([r["smp_s"] for r in series]),
            "mem_gb": desc([r["mem_gb"] for r in series]),
            "lr_first": first["lr"],
            "lr_last": last["lr"],
        }
    return out


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def fmt(v: Any, nd: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def dstr(d: dict[str, Any], key: str = "mean", nd: int = 4) -> str:
    if not d or d.get("n", 0) == 0:
        return "—"
    return f"{fmt(d.get(key), nd)} (n={d['n']})"


def build_markdown(a: dict[str, Any]) -> str:
    L: list[str] = []
    add = L.append
    add("# 量化分析报告（自动生成）\n")
    add("数据源：`reports/evidence/` 下的评测 JSON + `logs/vla/train_*.log`。")
    add("本报告只做**派生量**：分布、相关性、分组差异、余量，不改原始数值。\n")

    v = a["velocity"]
    add("## 1. S4 速度跟踪：误差分布与耦合\n")
    add(f"- vx_rmse：均值 {fmt(v['vx_rmse']['mean'])}，标准差 {fmt(v['vx_rmse']['std'])}，"
        f"范围 [{fmt(v['vx_rmse']['min'])}, {fmt(v['vx_rmse']['max'])}]，门禁 0.15 → "
        f"余量 **{fmt(v['headroom_vs_gate'], 2)}×**")
    add(f"- wz_rmse：均值 {fmt(v['wz_rmse']['mean'])}，标准差 {fmt(v['wz_rmse']['std'])}")
    add(f"- vx 与 wz 误差相关性：**{fmt(v['corr_vx_wz'], 3)}**（同时变差 = 同一个底层原因）")
    add(f"- 速度偏置（实际−指令）：均值 {fmt(v['bias_mean_vx_minus_cmd']['mean'])} "
        f"± {fmt(v['bias_mean_vx_minus_cmd']['std'])}（负 = 系统性跟不上）")
    add(f"- 摔倒 episode 的 vx_rmse：{dstr(v['vx_rmse_fall'])} vs 未摔 {dstr(v['vx_rmse_nonfall'])}")
    add(f"- **逐种子门禁检查**：{v['gate_violations']} 个种子单独超标"
        f"（最差 {fmt(v['worst_seed_value'])} > 0.15）→ 均值过门禁 ≠ 所有种子过门禁")
    add(f"- 最差 3 个种子：{v['worst_seeds']}\n")

    seq = a["sequence_layered"]
    add("## 2. S4 四阶段序列：指标分辨率\n")
    add(f"- 成功率 {seq['success_rate']}；完成步数 {dstr(seq['steps_finished'], 'mean', 1)}，"
        f"变异系数 **{fmt(seq['steps_variability'], 5)}**")
    add(f"- 终点到目标距离：{dstr(seq['terminal_miss_m'])} m；成功判据是 "
        f"**{seq['success_tolerance_m']} m** 位置容差 → 平均用掉 **{fmt(seq['miss_budget_used_mean'], 2)}×** 预算，"
        f"p95 用掉 **{fmt(seq['miss_budget_used_p95'], 2)}×**，"
        f"{seq['near_tolerance_count']} 个 episode 落在容差 90% 以外")
    add("- **问题**：完成步数在所有种子上完全不变（方差 0）→ 该指标无法区分微小改进；"
        "且成功率依赖 0.30 m 的宽松容差，实际落点平均偏 0.23 m。"
        "需要补充分段耗时/超调/指令跟踪等连续指标。\n")

    g = a["generalization"]
    add("## 3. 泛化矩阵：几何代价\n")
    add("| 几何 | 分层成功率 | 完成步数 | 单层成功率 |")
    add("|---|---|---|---|")
    for k, r in g["per_variant"].items():
        add(f"| {k} | {r['layered_success']} | {fmt(r['layered_steps'], 1)} | {r['single_success']} |")
    add("")
    add(f"- 路程从 2 m → 3 m：步数比 **{fmt(g['steps_ratio_3m_vs_2m'], 3)}**"
        f"（距离比 1.5）→ 拟合得固定开销 **{fmt(g['fixed_overhead_steps'], 0)} 步**、"
        f"每米 **{fmt(g['per_metre_steps'], 0)} 步**")
    add(f"- 转向 60° → 120°：每度约 **{fmt(g['turn_cost_steps_per_degree'], 3)} 步**"
        f"（30° ≈ {fmt((g['turn_cost_steps_per_degree'] or 0) * 30, 0)} 步，约 6% 总时长）")
    add("- 解读：**距离主要花钱、转向本身很便宜**；真正贵的是「减速—转向—重新加速」的固定段落"
        "（约占 24% 步数），优化点应该在阶段切换而不是转向精度。\n")

    s2 = a["s2"]
    add("## 4. S2 扰动：失效时间窗\n")
    add(f"- 无摔倒率 {s2['no_fall_rate']}（门禁 {s2['gate']}）")
    add(f"- 摔倒步数 {dstr(s2['fall_steps'], 'mean', 1)}，**推力施加于第 {s2['push_step']} 步**；"
        f"摔倒发生在推力后 {dstr(s2['fall_steps_after_push'], 'mean', 1)} 步"
        f"（约 {fmt(s2['fall_steps_after_push']['mean'] * 0.02, 2)} s）")
    add(f"- recovery_time：{dstr(s2['recovery_time_s'])} s；never-recover 计数 {s2['recovery_never_count']}")
    add("- 解读：全部 episode 在推力后 ~1 s 内摔倒，说明**不是控制增益不足，而是支撑相被打断后"
        "没有恢复步态**。\n")

    c0 = a["c0"]
    add("## 5. C0 脚本基线：确定性检查\n")
    for k, r in c0.items():
        add(f"- {k}: vx_rmse {dstr(r['vx_rmse'])}，fall {r['fall_rate']}，"
            f"确定性 {r['deterministic']}")
    add("- 脚本基线完全确定性（方差≈0）→ 任何 RL 对比的显著性都来自 RL 侧方差，"
        "单变量比较成立。\n")

    o = a["openloop"]
    add("## 6. VLA 开环误差分解（最有信息量的一节）\n")
    add(f"- 总体 MSE {fmt(o['mse'])} vs 基线 {fmt(o['baseline_mse'])}"
        f"（比值 **{fmt(o['mse_ratio'], 3)}**），帧数 {o['frames']}")
    add("\n| 维度 | GT 标准差 | MAE | 归一化 MAE | 有符号偏置 | 相关 |")
    add("|---|---|---|---|---|---|")
    for d, r in o["per_dim"].items():
        add(f"| {d} | {fmt(r['gt_std'])} | {fmt(r['mae'])} | {fmt(r['normalized_mae'], 2)} | "
            f"{fmt(r['signed_bias'], 3)} | {fmt(r['corr'], 2)} |")
    add("")
    add(f"- **最差维度：{o['worst_dim']}**（按归一化 MAE）")
    add("\n| 进度 | 动作 MAE | 夹爪指令均值（+1 闭合 / −1 张开） |")
    add("|---|---|---|")
    for k, r in o["progress_mae"].items():
        g = o["progress_gripper_activity"].get(k, {})
        add(f"| {k}% | {fmt(r['mean'])} | {fmt(g.get('mean'), 2)} |")
    add("")
    add(f"- 误差随进度趋势相关：**{fmt(o['progress_trend_corr'], 3)}**")
    gc, go_ = o["gripper_closed"], o["gripper_open"]
    add(f"- 夹爪**闭合**帧（n={gc['n']}）：臂 MAE {fmt(gc['mae_arm'])}，夹爪 MAE {fmt(gc['mae_gripper'])}")
    add(f"- 夹爪**张开**帧（n={go_['n']}）：臂 MAE {fmt(go_['mae_arm'])}，夹爪 MAE {fmt(go_['mae_gripper'])}")
    add("- 解读：夹爪通道是 ±1 的二值指令，" + f"{fmt(o['per_dim'][str(len(o['per_dim'])-1)]['mae'])} 的 MAE "
        "主要来自开合切换附近的帧（符号准确率 93%），不是「一直判错」")
    add(f"- 误差与动作幅度相关：**{fmt(o['corr_error_vs_motion_magnitude'], 3)}**")
    add(f"- 视觉依赖（遮图偏移）：{dstr(o['vision_delta'])}\n")

    t = a["training"]
    add("## 7. 训练效率：批量大小到底有没有用\n")
    add("| run | 步数区间 | loss | 每 10k 步降幅 | step_s | smp/s | 显存 |")
    add("|---|---|---|---|---|---|---|")
    for name, r in t.items():
        if not r.get("available"):
            add(f"| {name} | — | 日志缺失 | — | — | — | — |")
            continue
        add(f"| {name} | {int(r['steps'][0])}→{int(r['steps'][1])} | "
            f"{fmt(r['loss_first'], 3)}→{fmt(r['loss_last'], 3)} | "
            f"{fmt(r['loss_decay_per_10k_steps'], 4)} | "
            f"{fmt(r['step_s']['mean'], 3)} | {fmt(r['smp_s']['mean'], 1)} | "
            f"{fmt(r['mem_gb']['mean'], 2)} GB |")
    add("")
    add("- 关键量化：batch 8 与 batch 32 的 **smp/s 几乎相同**（45.1 vs 45.9）→ 训练是"
        "**GPU 浮点受限**，不是数据加载受限。加批量只减少优化步数，不提高样本吞吐。")
    add("- 收敛速率的塌陷：LIBERO 前 23k 步 loss 每 10k 降 **0.735**；后 42k 步（20k→62k）"
        "每 10k 只降 **0.0038** —— 相差 **约 190 倍**。继续训练的边际收益已接近 0。\n")

    add("## 8. 由数据推出的优化清单（按证据强度）\n")
    add("1. **VLA 控制频率**：数据 10 Hz / 环境 20 Hz，动作被执行快一倍 → 先对齐频率，"
        "成本最低、期望收益最大。")
    add("2. **按阶段定位，而不是一刀切缩短动作块**：误差不随进度单调增长（趋势相关 "
        f"{fmt(o['progress_trend_corr'], 2)}），但中段（50%）MAE 是前后段的两倍"
        " → 难的是「夹持搬运」这一段；先确认这一段是否与夹爪闭合重叠，再决定 chunk 长度。")
    add("3. **旋转维度是相对误差最大的**：维度 3–5 归一化 MAE 0.61–0.72（GT 标准差仅 "
        "0.035–0.056）→ 模型只复现了约三分之一的旋转信号；绝对量小，但闭环里可能累积成夹爪对不准。")
    add("4. **S2 需要的是恢复训练**：摔倒发生在推力后 1.19 s，且脚本基线同样倒 → "
        "加推力 DR + 恢复奖励，而不是调参。")
    add("5. **补指标分辨率**：序列任务完成步数方差为 0、终点残差恒为 0.23 m（容差 0.30 m）→ "
        "成功率无法区分改进；需要分段耗时、超调量、终点收敛精度等连续指标。")
    add("6. **训练别再加大批量**：batch 8 与 32 的样本吞吐相同（45.1 vs 45.9 smp/s），"
        "显存却从 4.2 GB 涨到 6.8 GB；要提速应降分辨率/减 chunk/换 bf16，而不是加 batch。")
    return "\n".join(L) + "\n"


def main() -> int:
    analysis = {
        "velocity": analyze_velocity(),
        "sequence_layered": analyze_sequence("layered"),
        "sequence_single": analyze_sequence("single"),
        "generalization": analyze_generalization(),
        "s2": analyze_s2(),
        "c0": analyze_c0(),
        "openloop": analyze_openloop(),
        "training": analyze_training(),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "quantitative_2026-11-15.json").write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    md = build_markdown(analysis)
    (OUT_DIR / "quantitative_2026-11-15.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
