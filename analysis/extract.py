#!/usr/bin/env python3
"""从证据目录生成因子表，并（重新）生成因子库文档。

输入：``reports/evidence/**/*.json``（评测原始输出）+ ``logs/vla/train_*.log``
输出：
  * ``reports/analysis/factors_run_<date>.csv``        — run 级因子（一行一个 run）
  * ``reports/analysis/factors_rollout_<date>.csv``    — rollout 级因子（一行一个 episode）
  * ``reports/analysis/factors_<date>.json``           — 全量（含分布与阈值判定）
  * ``docs/FACTORS.md``                                — 因子目录（自动生成）

用法::

    python analysis/extract.py                # 生成到默认路径
    python analysis/extract.py --date 2026-11-16
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys
from datetime import date as _date
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.factors import (  # noqa: E402
    BY_NAME,
    FACTORS,
    catalog,
    gate_check,
    percentile,
    run_factors,
    summarize,
)
from analysis.protocol import (  # noqa: E402
    CONTROL_DT_S,
    LIBERO_DATASET_FPS,
    LIBERO_ENV_FPS,
    POSITION_TOLERANCE_M,
)
from analysis.training import training_factors  # noqa: E402

EVID = ROOT / "reports" / "evidence"
OUT_DIR = ROOT / "reports" / "analysis"
DATE = _date.today().isoformat()
CONTROL_DT = CONTROL_DT_S
POS_TOLERANCE_M = POSITION_TOLERANCE_M


def load(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# 各实验的 rollout 抽取
# --------------------------------------------------------------------------- #
def velocity_rollouts() -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    data = load(EVID / "s4-sim2sim-2026-11-13" / "velocity_s0.json")
    rows = []
    for e in data["episodes"]:
        rows.append({
            "entity": f"seed{e['seed']}",
            "vx_rmse": e.get("vx_rmse"),
            "wz_rmse": e.get("wz_rmse"),
            "vx_bias": (e.get("mean_vx") - e.get("mean_cmd_vx"))
            if e.get("mean_vx") is not None and e.get("mean_cmd_vx") is not None else None,
            "fall": e.get("fall"),
        })
    return "s4_velocity_s0", rows, {"engine": "mujoco", "task": data["summary"].get("task")}


def sequence_rollouts(name: str) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    data = load(EVID / "s4-sim2sim-2026-11-13" / f"sequence_{name}.json")
    rows = []
    for e in data["episodes"]:
        pos, tgt = e.get("terminal_pos"), e.get("target_p2")
        miss = math.dist(pos, tgt) if (pos and tgt) else None
        rows.append({
            "entity": f"seed{e['seed']}",
            "success": e.get("success"),
            "fall": e.get("fall"),
            "steps": e.get("steps"),
            "terminal_miss": miss,
            "tolerance": POS_TOLERANCE_M,
            "stage_reached": e.get("max_stage"),
        })
    return f"s4_sequence_{name}", rows, {"engine": "mujoco", "task": data["summary"].get("task")}


def generalization_runs() -> list[tuple[str, list[dict[str, Any]], dict[str, Any]]]:
    data = load(EVID / "m2-campaign-2026-11-14" / "generalization_and_s2.json")
    out = []
    for group in ("layered", "single"):
        eps = data[f"generalization_{group}"]["episodes"]
        variants = sorted({e["variant"] for e in eps})
        for variant in variants:
            rows = [{
                "entity": f"seed{e['seed']}",
                "success": e.get("success"),
                "fall": e.get("fall"),
                "steps": e.get("steps"),
                "stage_reached": e.get("max_stage"),
                "variant": variant,
            } for e in eps if e["variant"] == variant]
            out.append((f"gen_{group}_{variant}", rows, {"variant": variant, "group": group}))
    return out


def s2_run() -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    data = load(EVID / "m2-campaign-2026-11-14" / "generalization_and_s2.json")
    s2 = data["s2_disturbance"]
    push_step = int(round(s2["summary"]["push"]["start_s"] / CONTROL_DT))
    rows = []
    for e in s2["episodes"]:
        fell = bool(e.get("fall"))
        rows.append({
            "entity": f"seed{e['seed']}",
            "fall": fell,
            # 注意：S2 的 steps 是"摔倒用时"而不是"完成用时"，因此不映射到 task_steps，
            # 只用于折算 stab_fall_after_push_s。
            "recovery_time_s": e.get("recovery_time_s"),
            "fail_after_push_steps": (e["steps"] - push_step) if fell else None,
        })
    return "s2_disturbance_push100N", rows, {"push_step": push_step,
                                            "force_n": s2["summary"]["push"]["force_n"]}


def friction_runs() -> list[tuple[str, list[dict[str, Any]], dict[str, Any]]]:
    """S2 摩擦阶跃轴：第 5 秒把足部摩擦降到给定倍率。"""
    out = []
    for tag in ("100", "07", "05", "03"):
        path = EVID / "m2-campaign-2026-11-16" / f"friction_{tag}.json"
        if not path.exists():
            continue
        data = load(path)
        rows = [{"entity": f"seed{e['seed']}", "vx_rmse": e.get("vx_rmse"),
                 "wz_rmse": e.get("wz_rmse"), "fall": e.get("fall")}
                for e in data["episodes"]]
        out.append((f"s2_friction_{tag}", rows, {"friction_tag": tag}))
    return out


def tolerance_runs() -> list[tuple[str, list[dict[str, Any]], dict[str, Any]]]:
    """S4 序列精度轴：位置容差 0.30 → 0.05 m。"""
    out = []
    for tag in ("030", "025", "020", "015", "010", "005"):
        path = EVID / "m2-campaign-2026-11-16" / f"tolerance_{tag}.json"
        if not path.exists():
            continue
        data = load(path)
        rows = [{"entity": f"seed{e['seed']}", "success": e.get("success"),
                 "fall": e.get("fall"), "steps": e.get("steps"),
                 "terminal_miss": e.get("terminal_miss_m"),
                 "tolerance": e.get("position_tolerance_m")}
                for e in data["episodes"]]
        out.append((f"s4_sequence_tol_{tag}", rows,
                    {"position_tolerance_m": data["summary"]["position_tolerance_m"]}))
    return out


def attribution_runs() -> list[tuple[str, list[dict[str, Any]], dict[str, Any]]]:
    """S4 归因消融：Oracle（规则直连 S0）与 Zero 对照。"""
    out = []
    for tag in ("oracle", "zero"):
        path = EVID / "s4-attribution-2026-11-16" / f"sequence_{tag}.json"
        if not path.exists():
            continue
        data = load(path)
        rows = [{"entity": f"seed{e['seed']}", "success": e.get("success"),
                 "fall": e.get("fall"), "steps": e.get("steps")}
                for e in data["episodes"]]
        out.append((f"s4_sequence_{tag}", rows, {"policy": tag}))
    gen = EVID / "s4-attribution-2026-11-16" / "generalization_oracle.json"
    if gen.exists():
        data = load(gen)
        rows = [{"entity": f"{e['variant']}_seed{e['seed']}", "success": e.get("success"),
                 "fall": e.get("fall"), "steps": e.get("steps"),
                 "stage_reached": e.get("max_stage")}
                for e in data["episodes"]]
        out.append(("s4_gen_oracle", rows, {"policy": "oracle"}))
    return out


def payload_runs() -> list[tuple[str, list[dict[str, Any]], dict[str, Any]]]:
    """S2 载荷轴：0/1/2/3 kg 各一个 run（同一批种子）。"""
    out = []
    for kg in (0, 1, 2, 3):
        path = EVID / "m2-campaign-2026-11-16" / f"payload_{kg}kg.json"
        if not path.exists():
            continue
        data = load(path)
        rows = [{"entity": f"seed{e['seed']}", "vx_rmse": e.get("vx_rmse"),
                 "wz_rmse": e.get("wz_rmse"), "fall": e.get("fall")}
                for e in data["episodes"]]
        out.append((f"s2_payload_{kg}kg", rows, {"payload_kg": kg}))
    return out


def c0_runs() -> list[tuple[str, list[dict[str, Any]], dict[str, Any]]]:
    data = load(EVID / "m2-campaign-2026-11-14" / "c0_baseline.json")
    out = []
    for key in ("c0_flat", "c0_push"):
        rows = [{
            "entity": f"seed{e['seed']}",
            "vx_rmse": e.get("vx_rmse"),
            "fall": e.get("fall"),
        } for e in data[key]["episodes"]]
        out.append((f"baseline_{key}", rows, {"method": data[key]["summary"].get("method")}))
    return out


def vla_closed_runs() -> list[tuple[str, list[dict[str, Any]], dict[str, Any]]]:
    """LIBERO 闭环评测。

    * fps10 —— 未加策略种子，作为结构性对照（全部超时）
    * fps20 unseeded —— 单次 10 集抽样，留档用来说明测量波动
    * fps20 seeded —— 两批各 10 集合并（20 集），**权威数字**
    """
    out = []
    seeded = []
    for tag in ("a", "b"):
        path = EVID / "vla-2026-11-15" / f"libero_seeded_{tag}.json"
        if path.exists():
            data = load(path)
            seeded.extend({
                "entity": f"seed{r['seed']}",
                "success": r["success"],
                "steps": r["steps"],
            } for r in data["results"])
    e1_path = EVID / "vla-2026-11-16" / "libero_e1_n25.json"
    if e1_path.exists():
        data = load(e1_path)
        rows = [{"entity": f"task{r['task']}_seed{r['seed']}",
                 "success": r["success"], "steps": r["steps"]}
                for r in data["results"]]
        out.append(("vla_closed_e1_n25", rows, {
            "fps": 20, "control_dt": 1.0 / 20, "n_action_steps": 25,
            "paired_p": data.get("paired_vs_baseline", {}).get("mcnemar_p"),
        }))

    suite_path = EVID / "vla-2026-11-15" / "libero_suite_5tasks.json"
    if suite_path.exists():
        data = load(suite_path)
        rows = [{"entity": f"task{r['task']}_seed{r['seed']}",
                 "success": r["success"], "steps": r["steps"]}
                for r in data["results"]]
        out.append(("vla_closed_suite_5tasks", rows, {
            "fps": 20, "control_dt": 1.0 / 20, "suite": data.get("suite"),
            "tasks": data.get("tasks"), "seeds": data.get("seeds"),
            "wilson95": data.get("wilson95"), "policy_seeded": True,
        }))

    if seeded:
        out.append(("vla_closed_fps20_seeded_n20", seeded, {
            "fps": 20, "control_dt": 1.0 / 20, "suite": "libero_spatial",
            "task_ids": [0], "policy_seeded": True,
            "batch_rates": [0.7, 0.3],
        }))
    for fps in (10, 20):
        path = EVID / "vla-2026-11-15" / f"libero_ab_fps{fps}_n10.json"
        if not path.exists():
            continue
        data = load(path)
        rows = [{
            "entity": f"ep{r['episode']}_seed{r['seed']}",
            "success": r["success"],
            "steps": r["steps"],
        } for r in data["results"]]
        out.append((f"vla_closed_fps{fps}_unseeded", rows, {
            "fps": fps,
            "control_dt": 1.0 / fps,
            "suite": data.get("suite"),
            "task_ids": data.get("task_ids"),
            "time_budget_s": data.get("time_budget_s"),
            "checkpoint": pathlib.Path(data.get("checkpoint", "")).name,
        }))
    return out


# --------------------------------------------------------------------------- #
# VLA 开环因子（帧级 → 语义级）
# --------------------------------------------------------------------------- #
def openloop_factors() -> dict[str, Any]:
    data = load(EVID / "vla-2026-11-15" / "openloop_060k.json")
    recs = data["records"]
    dim = len(recs[0]["gt"])

    per_dim = {}
    for d in range(dim):
        g = [float(r["gt"][d]) for r in recs]
        p = [float(r["pred"][d]) for r in recs]
        errs = [abs(a - b) for a, b in zip(g, p)]
        mae = sum(errs) / len(errs)
        gt_std = summarize(g)["std"] or 0.0
        per_dim[str(d)] = {
            "vla_ol_mae_dim": mae,
            "vla_ol_nmae_dim": (mae / gt_std) if gt_std > 1e-12 else None,
            "vla_ol_bias_dim": sum(a - b for a, b in zip(g, p)) / len(g),
            "vla_ol_corr_dim": _corr(g, p),
        }

    progress: dict[int, list[float]] = {}
    for i, r in enumerate(recs):
        decile = int(round(10 + 20 * (i % 5)))
        progress.setdefault(decile, []).append(
            sum(abs(a - b) for a, b in zip(r["gt"], r["pred"])) / dim)

    closed = [r for r in recs if float(r["gt"][-1]) > 0.5]
    opened = [r for r in recs if float(r["gt"][-1]) <= -0.5]

    def arm_mae(group: list[dict[str, Any]]) -> float | None:
        if not group:
            return None
        vals = [sum(abs(a - b) for a, b in zip(r["gt"][:-1], r["pred"][:-1])) / (dim - 1)
                for r in group]
        return sum(vals) / len(vals)

    vision = [math.dist(r["pred"], r["pred_blind"]) for r in recs if "pred_blind" in r]
    run = {
        "vla_ol_mse": data["mse"],
        "vla_ol_mse_ratio": data.get("mse_ratio_to_baseline"),
        "vla_ol_gripper_sign_acc": data.get("gripper_sign_accuracy"),
        "vla_ol_vision_delta": (sum(vision) / len(vision)) if vision else None,
    }
    semantic = {
        "per_dim": per_dim,
        "progress": {str(k): sum(v) / len(v) for k, v in sorted(progress.items())},
        "vla_ol_mae_gripper_closed": arm_mae(closed),
        "vla_ol_mae_gripper_open": arm_mae(opened),
        "n_frames": len(recs),
    }
    return {"run": run, "semantic": semantic}


def _corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx < 1e-12 or sy < 1e-12:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


# --------------------------------------------------------------------------- #
# 输出
# --------------------------------------------------------------------------- #
def write_run_csv(path: pathlib.Path, runs: list[dict[str, Any]]) -> None:
    names = sorted({n for r in runs for n in r["factors"]})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["run_id", "n"] + names)
        for r in runs:
            writer.writerow([r["run_id"], r["n"]] +
                            [("" if r["factors"].get(n) is None else f"{r['factors'][n]:.6g}")
                             for n in names])


def write_rollout_csv(path: pathlib.Path, table: list[dict[str, Any]]) -> None:
    factor_names = [f.name for f in FACTORS if f.level == "rollout" and f.available]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["run_id", "entity"] + factor_names)
        for row in table:
            writer.writerow([row["run_id"], row["entity"]] +
                            [("" if row["factors"].get(n) is None else f"{row['factors'][n]:.6g}")
                             for n in factor_names])


def write_catalog_md(path: pathlib.Path, cat: dict[str, Any], runs: list[dict[str, Any]]) -> None:
    lines: list[str] = []
    add = lines.append
    add("# 量化因子库（自动生成）\n")
    add("> 本文件由 `python analysis/extract.py` 生成，请勿手改；改定义请改 `analysis/factors.py`。\n")
    add(f"- 因子总数 **{cat['total']}**：已可采集 **{cat['available']}**，待补埋点 **{cat['missing']}**")
    add(f"- 生成日期：{DATE}")
    add("- 同一因子在所有脚本里只允许有一个实现（`analysis/factors.py`），避免口径漂移\n")

    group_titles = {
        "tracking": "速度跟踪",
        "stability": "稳定性 / 扰动恢复",
        "task": "任务完成",
        "generalization": "泛化",
        "vla_openloop": "VLA 开环",
        "vla_closed": "VLA 闭环",
        "training": "训练效率",
    }
    for group, items in cat["groups"].items():
        add(f"## {group_titles.get(group, group)}（`{group}`）\n")
        add("| 因子 | 层级 | 单位 | 方向 | 阈值 | 定义 | 来源 |")
        add("|---|---|---|---|---|---|---|")
        for it in items:
            flag = "" if it["available"] else " ⚠️"
            direction = {"lower": "越小越好", "higher": "越大越好", "neutral": "看符号"}[it["direction"]]
            add(f"| `{it['name']}`{flag} | {it['level']} | {it['unit']} | {direction} | "
                f"{'' if it['gate'] is None else it['gate']} | {it['definition']} | {it['source']} |")
        add("")

    add("## 待补埋点（`available=false`）\n")
    add("这些因子已定义但当前数据采不到。它们决定了下一轮实验应该记录什么：\n")
    add("| 因子 | 单位 | 需要的遥测 |")
    add("|---|---|---|")
    for f in FACTORS:
        if not f.available:
            add(f"| `{f.name}` | {f.unit} | {f.source} |")
    add("")

    add("## 当前 run 级因子快照\n")
    key_names = [n for n in ("track_vx_rmse", "track_wz_rmse", "track_axis_ratio_wz_vx",
                             "stab_fall_rate", "task_success_rate", "task_steps",
                             "task_steps_cv", "task_terminal_miss_m") if BY_NAME.get(n)]
    add("| run | n | " + " | ".join(f"`{n}`" for n in key_names) + " |")
    add("|---|---|" + "---|" * len(key_names))
    for r in runs:
        cells = []
        for n in key_names:
            v = r["factors"].get(n)
            cells.append("—" if v is None else f"{v:.4g}")
        add(f"| `{r['run_id']}` | {r['n']} | " + " | ".join(cells) + " |")
    add("")

    add("## 用法\n")
    add("```python")
    add("from analysis.factors import rollout_factors, run_factors, summarize, gate_check")
    add("")
    add("rec = {'vx_rmse': 0.07, 'fall': False, 'success': True, 'steps': 474, 'control_dt': 0.02}")
    add("rollout_factors(rec)            # 单条 rollout → 因子")
    add("run_factors('my_run', [rec])    # 一组 rollout → run 级聚合")
    add("gate_check('track_vx_rmse', 0.0773)  # {'gate': 0.15, 'pass': True, 'margin': 1.94}")
    add("```\n")
    add("```bash")
    add("python analysis/extract.py       # 重建因子表与本文档")
    add("```")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--date", default=DATE)
    parser.add_argument("--out-dir", type=pathlib.Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    out_dir: pathlib.Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) rollout 级
    rollout_table: list[dict[str, Any]] = []
    run_inputs: list[tuple[str, list[dict[str, Any]], dict[str, Any]]] = []
    run_inputs.append(velocity_rollouts())
    run_inputs.append(sequence_rollouts("layered"))
    run_inputs.append(sequence_rollouts("single"))
    run_inputs.extend(generalization_runs())
    run_inputs.append(s2_run())
    run_inputs.extend(c0_runs())
    run_inputs.extend(payload_runs())
    run_inputs.extend(attribution_runs())
    run_inputs.extend(tolerance_runs())
    run_inputs.extend(friction_runs())
    run_inputs.extend(vla_closed_runs())

    runs: list[dict[str, Any]] = []
    for run_id, rows, meta in run_inputs:
        tol = POS_TOLERANCE_M if run_id.startswith("s4_sequence") else None
        agg = run_factors(run_id, rows, meta.get("control_dt", CONTROL_DT), tol)
        agg["meta"] = meta
        runs.append(agg)
        for rec in rows:
            from analysis.factors import rollout_factors as _rf
            rollout_table.append({
                "run_id": run_id,
                "entity": rec.get("entity"),
                "factors": _rf({**rec, "control_dt": CONTROL_DT, "tolerance": tol}),
            })

    # 2) VLA 开环
    ol = openloop_factors()
    ol_run = {"run_id": "vla_openloop_060k", "n": ol["semantic"]["n_frames"],
              "factors": ol["run"], "detail": {}, "meta": {"checkpoint": "060000"}}
    runs.append(ol_run)

    # 2b) 几何成本拟合：steps = 固定开销 + 每米步数 × 距离
    by_id = {r["run_id"]: r["factors"] for r in runs}

    def _f(run_id: str, name: str) -> float | None:
        return by_id.get(run_id, {}).get(name)

    s2m = _f("gen_layered_trained_2m_90deg", "task_steps")
    s3m = _f("gen_layered_unseen_3m_90deg", "task_steps")
    s60 = _f("gen_layered_unseen_2m_60deg", "task_steps")
    s120 = _f("gen_layered_unseen_2m_120deg", "task_steps")
    fit: dict[str, float] = {}
    if s2m is not None and s3m is not None:
        fit["gen_per_metre_steps"] = s3m - s2m
        fit["gen_fixed_overhead_steps"] = s2m - 2.0 * (s3m - s2m)
    if s60 is not None and s120 is not None:
        fit["gen_turn_cost_per_deg"] = (s120 - s60) / 60.0
    if fit:
        runs.append({"run_id": "gen_geometry_fit", "n": 4, "factors": fit, "detail": {},
                     "meta": {"model": "steps = fixed + per_metre * distance"}})

    # 3) 配置类因子：数据帧率与评测控制频率的错配倍数
    runs.append({
        "run_id": "vla_libero_setup",
        "n": 1,
        "factors": {"vla_control_freq_ratio": LIBERO_ENV_FPS / LIBERO_DATASET_FPS},
        "detail": {},
        "meta": {"dataset_fps": LIBERO_DATASET_FPS, "env_fps": LIBERO_ENV_FPS},
    })

    # 4) 训练
    training_specs = [
        ("train_teacher_bs8", "train_smolvla_002.log", None),
        ("train_libero_bs8", "train_libero_003.log", None),
        ("train_libero_bs32", "train_libero_005.log", 30000.0),
    ]
    for run_id, log_name, split in training_specs:
        factors = training_factors(ROOT / "logs" / "vla" / log_name, split)
        factors = {k: v for k, v in factors.items() if v is not None}
        runs.append({"run_id": run_id, "n": 1, "factors": factors, "detail": {},
                     "meta": {"log": f"logs/vla/{log_name}"}})

    # 5) 落盘
    write_run_csv(out_dir / f"factors_run_{args.date}.csv", runs)
    write_rollout_csv(out_dir / f"factors_rollout_{args.date}.csv", rollout_table)

    cat = catalog()
    payload = {
        "date": args.date,
        "catalog": cat,
        "runs": runs,
        "openloop_semantic": ol["semantic"],
        "gates": {r["run_id"]: {n: gate_check(n, v) for n, v in r["factors"].items()
                                if BY_NAME.get(n) and BY_NAME[n].gate is not None}
                  for r in runs},
    }
    (out_dir / f"factors_{args.date}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_catalog_md(ROOT / "docs" / "FACTORS.md", cat, runs)

    print(f"[factors] {len(FACTORS)} 个因子（可用 {cat['available']} / 待补 {cat['missing']}）")
    print(f"[factors] runs={len(runs)} rollouts={len(rollout_table)}")
    print(f"[factors] 写出 {out_dir}/factors_run_{args.date}.csv")
    print(f"[factors] 写出 {out_dir}/factors_rollout_{args.date}.csv")
    print(f"[factors] 写出 {out_dir}/factors_{args.date}.json")
    print("[factors] 写出 docs/FACTORS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
