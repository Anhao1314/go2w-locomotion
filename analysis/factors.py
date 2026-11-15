"""因子库：把评测与训练的原始输出折算成统一的量化因子。

设计原则
--------
1. **一个因子一个定义**。名称、层级、单位、方向（越大越好/越小越好）、阈值、数据来源
   全部写死在 :data:`FACTORS` 里；计算函数只允许按注册表产出，避免同一个指标在不同
   脚本里各算一遍。
2. **区分「已可采集」与「需新增埋点」**。`available=False` 的因子不是待办装饰，而是
   对遥测 schema 的缺口说明——它们定义了下一轮实验应该记录什么。
3. **层级分明**：`rollout`（单次 episode）→ `run`（一组 episode 的聚合）→ `frame`
   （VLA 开环单帧）→ `train_step`（训练日志行）→ `semantic`（变体/子群）。

用法::

    from analysis.factors import FACTORS, rollout_factors, summarize, run_factors
    rec = {"run_id": "s4_velocity_s0", "entity": "seed0", "vx_rmse": 0.07, ...}
    rollout_factors(rec)          # {'track_vx_rmse': 0.07, ...}
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Iterable

# --------------------------------------------------------------------------- #
# 注册表
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Factor:
    """一个可计算因子的完整定义。"""

    name: str
    group: str
    level: str  # rollout | run | frame | train_step | semantic
    unit: str
    direction: str  # lower | higher | neutral
    definition: str
    source: str
    gate: float | None = None
    available: bool = True


FACTORS: tuple[Factor, ...] = (
    # ---- 速度跟踪 -------------------------------------------------------- #
    Factor("track_vx_rmse", "tracking", "rollout", "m/s", "lower",
           "前向速度跟踪均方根误差（相对指令）", "evaluation.json: episodes[].vx_rmse", 0.15),
    Factor("track_wz_rmse", "tracking", "rollout", "rad/s", "lower",
           "偏航角速度跟踪均方根误差", "evaluation.json: episodes[].wz_rmse"),
    Factor("track_vx_bias", "tracking", "rollout", "m/s", "neutral",
           "实际平均速度减指令速度（有符号；负=系统性跟不上）",
           "evaluation.json: mean_vx - mean_cmd_vx"),
    Factor("track_vx_rmse_cv", "tracking", "run", "1", "lower",
           "vx_rmse 的种子间变异系数（标准差/均值）", "派生"),
    Factor("track_wz_rmse_cv", "tracking", "run", "1", "lower",
           "wz_rmse 的种子间变异系数", "派生"),
    Factor("track_axis_ratio_wz_vx", "tracking", "run", "1", "lower",
           "wz_rmse / vx_rmse：转向相对直行的跟踪难度倍数", "派生"),
    # ---- 稳定性 ---------------------------------------------------------- #
    Factor("stab_fall", "stability", "rollout", "0/1", "lower",
           "该 episode 是否摔倒", "evaluation.json: episodes[].fall"),
    Factor("stab_fall_rate", "stability", "run", "1", "lower",
           "摔倒比例（门禁 ≤0.10）", "派生", 0.10),
    Factor("stab_no_fall_rate", "stability", "run", "1", "higher",
           "无摔倒比例（S2 门禁 ≥0.90）", "派生", 0.90),
    Factor("stab_recovery_time_s", "stability", "rollout", "s", "lower",
           "扰动后恢复到直立的时间（未恢复记 None）",
           "evaluation.json: episodes[].recovery_time_s"),
    Factor("stab_fall_after_push_s", "stability", "rollout", "s", "lower",
           "推力施加后到摔倒的延迟（失效时间窗）", "派生：steps 差 × control_dt"),
    Factor("stab_max_tilt_deg", "stability", "rollout", "deg", "lower",
           "机身最大倾角（需要逐步姿态日志）", "未采集：需记录 root quaternion", None, False),
    Factor("stab_slip_ratio", "stability", "rollout", "1", "lower",
           "接触滑动步数占比（需要逐步接触状态）", "未采集：需记录 contact", None, False),
    # ---- 任务完成 -------------------------------------------------------- #
    Factor("task_success", "task", "rollout", "0/1", "higher",
           "该 episode 是否判定成功", "evaluation.json: episodes[].success"),
    Factor("task_success_rate", "task", "run", "1", "higher",
           "成功率（预注册门禁 ≥0.90）", "派生", 0.90),
    Factor("task_steps", "task", "rollout", "steps", "lower",
           "完成所需控制步数", "evaluation.json: episodes[].steps"),
    Factor("task_time_s", "task", "rollout", "s", "lower",
           "完成所需时间 = 步数 × 控制周期", "派生"),
    Factor("task_terminal_miss_m", "task", "rollout", "m", "lower",
           "终点到目标的平面距离", "evaluation.json: |terminal_pos - target_p2|"),
    Factor("task_miss_ratio", "task", "rollout", "1", "lower",
           "终点残差 / 位置容差（成功预算的消耗比例）", "派生：miss / 0.30"),
    Factor("task_stage_reached", "task", "rollout", "stage", "higher",
           "达到的最大阶段（0-4）", "evaluation.json: episodes[].max_stage"),
    Factor("task_steps_cv", "task", "run", "1", "lower",
           "完成步数的变异系数（指标分辨率；0 = 测不出改进）", "派生"),
    Factor("task_path_efficiency", "task", "rollout", "1", "higher",
           "直线距离 / 实际路径长度（需要轨迹日志）", "未采集：需记录逐步位置", None, False),
    Factor("task_stage_time_s", "task", "rollout", "s", "lower",
           "各阶段耗时（需要阶段切换时间戳）", "未采集：需记录 stage transition", None, False),
    # ---- 泛化 ------------------------------------------------------------ #
    Factor("gen_success_variant", "generalization", "semantic", "1", "higher",
           "某几何变体上的成功率", "generalization: per_variant"),
    Factor("gen_steps_variant", "generalization", "semantic", "steps", "lower",
           "某几何变体上的平均完成步数", "generalization: per_variant"),
    Factor("gen_fixed_overhead_steps", "generalization", "run", "steps", "lower",
           "拟合的任务固定开销（起停 + 转向段），steps = a + b·distance", "派生"),
    Factor("gen_per_metre_steps", "generalization", "run", "steps/m", "lower",
           "每多 1 米路程增加的步数（拟合斜率）", "派生"),
    Factor("gen_turn_cost_per_deg", "generalization", "run", "steps/deg", "lower",
           "每多 1 度转向增加的步数", "派生"),
    # ---- VLA 开环 -------------------------------------------------------- #
    Factor("vla_ol_mse", "vla_openloop", "run", "action²", "lower",
           "开环动作预测 MSE", "openloop_summary.json: mse"),
    Factor("vla_ol_mse_ratio", "vla_openloop", "run", "1", "lower",
           "MSE / 平均动作基线 MSE（<1 表示优于常数预测）", "派生"),
    Factor("vla_ol_mae_dim", "vla_openloop", "semantic", "action", "lower",
           "逐维 MAE（7 维：3 位移 + 3 旋转 + 1 夹爪）", "openloop records"),
    Factor("vla_ol_nmae_dim", "vla_openloop", "semantic", "1", "lower",
           "逐维归一化 MAE = MAE / GT 标准差（跨量纲可比）", "派生"),
    Factor("vla_ol_bias_dim", "vla_openloop", "semantic", "action", "neutral",
           "逐维有符号偏置（区分系统性偏差与噪声）", "派生"),
    Factor("vla_ol_corr_dim", "vla_openloop", "semantic", "1", "higher",
           "逐维预测-真值相关系数", "派生"),
    Factor("vla_ol_mae_progress", "vla_openloop", "semantic", "action", "lower",
           "按 episode 进度分箱的动作 MAE", "派生"),
    Factor("vla_ol_mae_gripper_closed", "vla_openloop", "semantic", "action", "lower",
           "夹爪闭合（搬运）段的臂部 MAE", "派生"),
    Factor("vla_ol_mae_gripper_open", "vla_openloop", "semantic", "action", "lower",
           "夹爪张开（接近/释放）段的臂部 MAE", "派生"),
    Factor("vla_ol_gripper_sign_acc", "vla_openloop", "run", "1", "higher",
           "夹爪开合符号判断准确率", "派生"),
    Factor("vla_ol_vision_delta", "vla_openloop", "run", "action", "higher",
           "遮住图像后的动作偏移 L2（视觉依赖强度，太低=不看图）", "派生"),
    Factor("vla_closed_success_rate", "vla_closed", "run", "1", "higher",
           "闭环评测成功率（参考区间：LIBERO 文献 66–87%）",
           "libero_eval_summary.json: success_rate"),
    Factor("vla_control_freq_ratio", "vla_closed", "run", "1", "neutral",
           "环境控制频率 / 数据集元数据帧率；A/B 实测 2.0 有效（7/10）而 1.0 全败（0/10）"
           "→ 元数据 fps 不是控制契约",
           "analysis/protocol.py: LIBERO_ENV_FPS / LIBERO_DATASET_FPS"),
    # ---- 训练效率 -------------------------------------------------------- #
    Factor("train_smp_per_s", "training", "train_step", "samples/s", "higher",
           "样本吞吐", "train log: smp/s"),
    Factor("train_step_s", "training", "train_step", "s", "lower",
           "每优化步墙钟时间", "train log: step_s"),
    Factor("train_mem_gb", "training", "train_step", "GB", "lower",
           "训练显存占用", "train log: mem_gb"),
    Factor("train_loss_decay_per_10k", "training", "run", "loss/10k steps", "higher",
           "每 1 万步的 loss 降幅（收敛速率）", "派生"),
    Factor("train_plateau_ratio", "training", "run", "1", "lower",
           "后段收敛速率 / 前段收敛速率（越小越接近平台）", "派生"),
    Factor("train_grad_norm", "training", "train_step", "1", "neutral",
           "梯度范数", "train log: grdn"),
    Factor("train_lr_end", "training", "run", "1", "neutral",
           "训练结束时的学习率（判断退火是否走完）", "train log: lr"),
)

BY_NAME: dict[str, Factor] = {f.name: f for f in FACTORS}


# --------------------------------------------------------------------------- #
# 统计工具（纯标准库，避免给 CI 引 pandas）
# --------------------------------------------------------------------------- #
def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = q * (len(xs) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return xs[int(pos)]
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def summarize(values: Iterable[float | None]) -> dict[str, float | int | None]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None,
                "p10": None, "p90": None, "cv": None, "ci95": None}
    mean = statistics.fmean(vals)
    std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return {
        "n": len(vals),
        "mean": mean,
        "std": std,
        "min": min(vals),
        "max": max(vals),
        "p10": percentile(vals, 0.10),
        "p90": percentile(vals, 0.90),
        "cv": (std / abs(mean)) if abs(mean) > 1e-12 else None,
        "ci95": 1.96 * std / math.sqrt(len(vals)) if len(vals) > 1 else 0.0,
    }


def gate_check(name: str, value: float | None) -> dict[str, Any]:
    """按注册表里的方向与阈值判定，并给出余量倍数。"""
    factor = BY_NAME.get(name)
    if factor is None or factor.gate is None or value is None:
        return {"gate": None, "pass": None, "margin": None}
    gate = factor.gate
    if factor.direction == "lower":
        ok = value <= gate
        margin = gate / value if abs(value) > 1e-12 else None
    elif factor.direction == "higher":
        ok = value >= gate
        margin = value / gate if abs(gate) > 1e-12 else None
    else:
        ok = None
        margin = None
    return {"gate": gate, "pass": ok, "margin": margin}


# --------------------------------------------------------------------------- #
# rollout 级因子
# --------------------------------------------------------------------------- #
def rollout_factors(rec: dict[str, Any]) -> dict[str, float | None]:
    """把一条标准化的 rollout 记录折算成因子字典。

    记录字段（缺省即 None）: vx_rmse, wz_rmse, vx_bias, success, fall, steps,
    control_dt, terminal_miss, tolerance, stage_reached, recovery_time_s,
    fail_after_push_steps
    """
    dt = rec.get("control_dt") or 0.02
    steps = rec.get("steps")
    miss = rec.get("terminal_miss")
    tol = rec.get("tolerance")
    fail_delay = rec.get("fail_after_push_steps")
    # 完成类因子只在成功 episode 上有意义：失败时的 steps 是"摔倒用时"，
    # 残差是"根本不在目标附近"，混进均值会污染 run 级结论。
    succeeded = rec.get("success")
    complete = None if succeeded is False else steps
    miss_for_ratio = miss if succeeded is not False else None
    return {
        "track_vx_rmse": rec.get("vx_rmse"),
        "track_wz_rmse": rec.get("wz_rmse"),
        "track_vx_bias": rec.get("vx_bias"),
        "stab_fall": (1.0 if rec["fall"] else 0.0) if rec.get("fall") is not None else None,
        "stab_recovery_time_s": rec.get("recovery_time_s"),
        "stab_fall_after_push_s": (fail_delay * dt) if fail_delay is not None else None,
        "task_success": (1.0 if rec["success"] else 0.0) if rec.get("success") is not None else None,
        "task_steps": complete,
        "task_time_s": (complete * dt) if complete is not None else None,
        "task_terminal_miss_m": miss,
        "task_miss_ratio": (miss_for_ratio / tol) if (miss_for_ratio is not None and tol) else None,
        "task_stage_reached": rec.get("stage_reached"),
    }


# --------------------------------------------------------------------------- #
# run 级因子
# --------------------------------------------------------------------------- #
def run_factors(
    run_id: str,
    rollouts: list[dict[str, Any]],
    control_dt: float = 0.02,
    tolerance: float | None = None,
) -> dict[str, Any]:
    """一组 rollout 聚合成 run 级因子。

    返回 ``{"run_id": ..., "n": ..., "factors": {name: value}, "detail": {...}}``。
    """
    rows = [rollout_factors({**r, "control_dt": r.get("control_dt", control_dt),
                             "tolerance": r.get("tolerance", tolerance)})
            for r in rollouts]

    def col(name: str) -> list[float | None]:
        return [row.get(name) for row in rows]

    out: dict[str, Any] = {"run_id": run_id, "n": len(rows), "factors": {}, "detail": {}}

    def put(name: str, fn) -> None:
        value = fn()
        if value is not None:
            out["factors"][name] = value

    put("track_vx_rmse", lambda: summarize(col("track_vx_rmse"))["mean"])
    put("track_wz_rmse", lambda: summarize(col("track_wz_rmse"))["mean"])
    put("track_vx_bias", lambda: summarize(col("track_vx_bias"))["mean"])
    put("track_vx_rmse_cv", lambda: summarize(col("track_vx_rmse"))["cv"])
    put("track_wz_rmse_cv", lambda: summarize(col("track_wz_rmse"))["cv"])

    vx_mean = out["factors"].get("track_vx_rmse")
    wz_mean = out["factors"].get("track_wz_rmse")
    if vx_mean and wz_mean:
        out["factors"]["track_axis_ratio_wz_vx"] = wz_mean / vx_mean

    put("stab_fall_rate", lambda: summarize(col("stab_fall"))["mean"])
    fall_rate = out["factors"].get("stab_fall_rate")
    if fall_rate is not None:
        out["factors"]["stab_no_fall_rate"] = 1.0 - fall_rate
    put("stab_recovery_time_s", lambda: summarize(col("stab_recovery_time_s"))["mean"])
    put("stab_fall_after_push_s", lambda: summarize(col("stab_fall_after_push_s"))["mean"])

    put("task_success_rate", lambda: summarize(col("task_success"))["mean"])
    put("task_steps", lambda: summarize(col("task_steps"))["mean"])
    put("task_steps_cv", lambda: summarize(col("task_steps"))["cv"])
    put("task_time_s", lambda: summarize(col("task_time_s"))["mean"])
    put("task_terminal_miss_m", lambda: summarize(col("task_terminal_miss_m"))["mean"])
    put("task_miss_ratio", lambda: summarize(col("task_miss_ratio"))["mean"])
    put("task_stage_reached", lambda: summarize(col("task_stage_reached"))["mean"])

    # 保留完整分布，便于后续做置信区间与最坏情况分析
    out["detail"] = {
        name: summarize(col(name))
        for name in ("track_vx_rmse", "track_wz_rmse", "track_vx_bias",
                     "task_steps", "task_terminal_miss_m", "stab_recovery_time_s")
        if any(v is not None for v in col(name))
    }
    return out


# --------------------------------------------------------------------------- #
# 因子目录（供文档生成）
# --------------------------------------------------------------------------- #
def catalog() -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for f in FACTORS:
        groups.setdefault(f.group, []).append({
            "name": f.name,
            "level": f.level,
            "unit": f.unit,
            "direction": f.direction,
            "definition": f.definition,
            "source": f.source,
            "gate": f.gate,
            "available": f.available,
        })
    return {
        "total": len(FACTORS),
        "available": sum(1 for f in FACTORS if f.available),
        "missing": sum(1 for f in FACTORS if not f.available),
        "groups": groups,
    }
