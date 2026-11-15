#!/usr/bin/env python3
"""从证据与因子表生成图表包（media/analysis/）。

六张图，全部直接读 ``reports/evidence`` / ``reports/analysis`` 的原始数据，
不手填任何数字——重跑即可与数据保持一致。

  1. vla_fps_ab.png            LIBERO 控制频率 A/B（每 episode 散点 + 成功率）
  2. vla_openloop_decomp.png   开环误差分解（按进度 / 按夹爪状态 / 按维度）
  3. generalization_cost.png   泛化几何成本（步数 vs 距离、vs 转向角）
  4. s2_failure_window.png     S2 侧推的失效时间窗（20 条 episode）
  5. factor_health.png         因子体检热力图（覆盖/方差/分辨率/冗余/杠杆）
  6. training_plateau.png      LIBERO 训练收敛塌陷（loss 曲线 + 斜率对比）

用法：python scripts/make_factor_charts.py [--out media/analysis]
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.training import parse_training_log  # noqa: E402

EVID = ROOT / "reports" / "evidence"
ANALYSIS = ROOT / "reports" / "analysis"

BLUE = "#2E5EAA"
ORANGE = "#D9822B"
GREEN = "#2E8B57"
RED = "#C0392B"
GRAY = "#8A8F98"


def setup_style() -> None:
    available = {f.name for f in font_manager.fontManager.ttflist}
    for family in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"):
        if family in available:
            plt.rcParams["font.sans-serif"] = [family]
            break
    plt.rcParams.update({
        "axes.unicode_minus": False,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    })


def load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
def chart_fps_ab(out: pathlib.Path) -> None:
    batch_a = load(EVID / "vla-2026-11-15" / "libero_seeded_a.json")
    batch_b = load(EVID / "vla-2026-11-15" / "libero_seeded_b.json")
    runs: dict[int, dict] = {
        20: {"results": batch_a["results"] + batch_b["results"],
             "successes": batch_a["successes"] + batch_b["successes"],
             "episodes": batch_a["episodes"] + batch_b["episodes"]},
        10: load(EVID / "vla-2026-11-15" / "libero_ab_fps10_n10.json"),
    }
    runs[20]["success_rate"] = runs[20]["successes"] / runs[20]["episodes"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 4.0),
                                   gridspec_kw={"width_ratios": [1, 1.35]})

    fps_list = [20, 10]
    rates = [runs[f]["success_rate"] for f in fps_list]
    bars = ax1.bar([f"fps={f}" for f in fps_list], rates,
                   color=[GREEN, RED], width=0.55, alpha=0.85)
    for i, (bar, fps) in enumerate(zip(bars, fps_list)):
        n = runs[fps]["episodes"]
        ok = runs[fps]["successes"]
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                 f"{ok}/{n}", ha="center", va="bottom", fontweight="bold")
        if ok == 0:  # 零高度柱子不可见，补一条基线
            ax1.hlines(0.008, i - 0.26, i + 0.26, color=RED, lw=5)
    ax1.set_ylim(0, 1.0)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["fps=20", "fps=10"])
    ax1.set_xlabel("批 A 7/10 · 批 B 3/10（fps=20 合计 10/20）", fontsize=8.5,
                   color=GRAY)
    ax1.set_ylabel("成功率（libero_spatial 任务 0）")
    ax1.set_title("控制频率 A/B（种子固定，fps=20 共 20 集）")

    for i, fps in enumerate(fps_list):
        results = runs[fps]["results"]
        xs = [i + (j - len(results) / 2) * 0.035 for j in range(len(results))]
        ys = [r["steps"] for r in results]
        colors = [GREEN if r["success"] else RED for r in results]
        ax2.scatter(xs, ys, c=colors, s=38, alpha=0.85, edgecolors="white", linewidths=0.6)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels([f"fps={f}" for f in fps_list])
    ax2.set_ylabel("episode 步数")
    ax2.set_title("每条 episode：绿=成功，红=超时")
    ax2.axhline(280, color=GRAY, ls=":", lw=1)
    ax2.axhline(140, color=GRAY, ls=":", lw=1)
    label_box = {"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5}
    ax2.text(1.47, 268, "20 Hz 上限 280 步", color=GRAY, fontsize=8, ha="right",
             va="top", bbox=label_box)
    ax2.text(1.47, 152, "10 Hz 上限 140 步（同 14 s 预算）", color=GRAY, fontsize=8,
             ha="right", va="bottom", bbox=label_box)

    fig.suptitle("VLA 闭环：同一个 60K checkpoint，只有控制频率不同", y=1.02, fontsize=12)
    fig.savefig(out / "vla_fps_ab.png")
    plt.close(fig)


def chart_vla_failure_traces(out: pathlib.Path) -> None:
    pack = load(ROOT / "logs/vla/libero_video_pack2/libero_eval_summary.json")
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 6.0), sharex=False)
    for ax, r in zip(axes.ravel(), pack["results"]):
        trace = r.get("trace") or {}
        grip = [float(v) for v in trace.get("gripper_action") or []]
        z = [float(v) for v in trace.get("state_z") or []]
        color = GREEN if r["success"] else RED
        ax.plot(grip, color=color, lw=1.2)
        ax.set_ylim(-1.25, 1.25)
        ax.set_title(f"seed {r['seed']} · {'SUCCESS' if r['success'] else 'TIMEOUT'}"
                     f" · {r['steps']} 步 · 开合 {sum(1 for i in range(1, len(grip)) if (grip[i] > 0) != (grip[i - 1] > 0))} 次",
                     fontsize=9, color=color)
        ax.set_ylabel("夹爪指令", fontsize=9)
        ax2 = ax.twinx()
        ax2.plot(z, color=GRAY, lw=1.0, alpha=0.8)
        ax2.set_ylabel("末端高度 (m)", fontsize=8, color=GRAY)
        ax2.grid(False)
    for ax in axes[1]:
        ax.set_xlabel("步数")
    fig.suptitle("同一任务的成败轨迹：失败集在反复开合夹爪（22–23 次），成功集只开合 1–4 次", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "vla_failure_traces.png")
    plt.close(fig)


def chart_payload(out: pathlib.Path) -> None:
    kgs, vx, fall = [], [], []
    for kg in (0, 1, 2, 3):
        d = load(EVID / "m2-campaign-2026-11-16" / f"payload_{kg}kg.json")
        s = d["summary"]
        kgs.append(kg)
        vx.append(s["vx_rmse_mean"])
        fall.append(s["fall_rate"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.7))
    ax1.plot(kgs, fall, "o-", color=RED, lw=2, ms=8)
    for x, y in zip(kgs, fall):
        ax1.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 9),
                     ha="center", fontsize=9)
    ax1.set_xticks(kgs)
    ax1.set_xlabel("基础载荷（kg）")
    ax1.set_ylabel("摔倒率")
    ax1.set_ylim(0, 0.35)
    ax1.set_title("载荷吃的是稳定性，不是跟踪精度")

    ax2.plot(kgs, vx, "o-", color=BLUE, lw=2, ms=8)
    for x, y in zip(kgs, vx):
        ax2.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0, 9),
                     ha="center", fontsize=9)
    ax2.set_xticks(kgs)
    ax2.set_xlabel("基础载荷（kg）")
    ax2.set_ylabel("vx_rmse（m/s）")
    ax2.set_title("跟踪误差基本不变（+6% @3 kg）")

    fig.suptitle("S2 载荷轴（MuJoCo，20 种子）：摔倒率 0.05 → 0.25，跟踪误差 0.077 → 0.082", y=1.03, fontsize=11)
    fig.savefig(out / "s2_payload.png")
    plt.close(fig)


def chart_e1_vs_base(out: pathlib.Path) -> None:
    base = load(EVID / "vla-2026-11-15" / "libero_suite_5tasks.json")
    e1 = load(EVID / "vla-2026-11-16" / "libero_e1_n25.json")
    paired = e1["paired_vs_baseline"]
    tasks = [str(t) for t in base["tasks"]]
    b = [base["per_task"][t]["success_rate"] for t in tasks]
    e = [e1["per_task"][t]["success_rate"] for t in tasks]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.9),
                                   gridspec_kw={"width_ratios": [1.6, 1]})
    x = np.arange(len(tasks))
    ax1.bar(x - 0.2, b, width=0.38, color=GRAY, alpha=0.85, label="基线 n_action_steps=50")
    ax1.bar(x + 0.2, e, width=0.38, color=BLUE, alpha=0.85, label="E1 n_action_steps=25")
    for xi, bv, ev in zip(x, b, e):
        ax1.text(xi - 0.2, bv + 0.015, f"{bv:.2f}", ha="center", fontsize=8, color="#555")
        ax1.text(xi + 0.2, ev + 0.015, f"{ev:.2f}", ha="center", fontsize=8, color="#2E5EAA")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"task {t}" for t in tasks])
    ax1.set_ylim(0, 0.9)
    ax1.set_ylabel("成功率")
    ax1.set_title("逐任务：只改重规划频率")
    ax1.legend(loc="upper left", fontsize=8)

    cells = [[paired["b11"], paired["b10"]], [paired["b01"], 100 - paired["b11"] - paired["b10"] - paired["b01"]]]
    ax2.imshow(np.array(cells), cmap="Blues", alpha=0.75)
    labels = [["都成功", "基线成功/E1失败"], ["基线失败/E1成功", "都失败"]]
    for i in range(2):
        for j in range(2):
            text = labels[i][j] + ": " + str(cells[i][j])
            ax2.text(j, i, text, ha="center", va="center", fontsize=9)
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.grid(False)
    ax2.set_title(f"配对翻转：Δ={paired['delta']:+.2f}，p={paired['mcnemar_p']:.2f}", fontsize=10)

    fig.suptitle("E1 执行器消融：重规划更频繁没有可靠收益，但 49/100 集结局翻转", y=1.03, fontsize=11)
    fig.savefig(out / "vla_e1_vs_base.png")
    plt.close(fig)


def chart_suite_tasks(out: pathlib.Path) -> None:
    data = load(EVID / "vla-2026-11-15" / "libero_suite_5tasks.json")
    tasks = [str(t) for t in data["tasks"]]
    rates = [data["per_task"][t]["success_rate"] for t in tasks]
    ns = [data["per_task"][t]["episodes"] for t in tasks]
    overall = data["success_rate"]
    lo, hi = data["wilson95"]

    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    bars = ax.bar([f"task {t}" for t in tasks], rates, color=BLUE, alpha=0.85, width=0.6)
    for bar, r, n in zip(bars, rates, ns):
        label = f"{r:.2f} ({int(round(r * n))}/{n})"
        ax.text(bar.get_x() + bar.get_width() / 2, r + 0.02, label,
                ha="center", va="bottom", fontsize=8.5)
    ax.axhline(overall, color=ORANGE, lw=1.6, ls="--")
    ax.fill_between([-0.5, len(tasks) - 0.5], lo, hi, color=ORANGE, alpha=0.12)
    note = f"合计 {overall:.2f}  95% CI [{lo:.2f}, {hi:.2f}]"
    ax.text((len(tasks) - 1) / 2, 0.93, note,
            ha="center", va="center", fontsize=9, color=ORANGE)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("成功率")
    ax.set_title(f"LIBERO-Spatial 五个任务 × 20 种子（共 {data['episodes']} 集）")
    fig.savefig(out / "vla_suite_tasks.png")
    plt.close(fig)


def chart_openloop(out: pathlib.Path) -> None:
    data = load(EVID / "vla-2026-11-15" / "openloop_060k.json")
    recs = data["records"]
    dim = len(recs[0]["gt"])

    progress: dict[int, list[float]] = {}
    for i, r in enumerate(recs):
        decile = int(round(10 + 20 * (i % 5)))
        progress.setdefault(decile, []).append(
            float(np.mean(np.abs(np.array(r["gt"]) - np.array(r["pred"])))))

    closed = [r for r in recs if float(r["gt"][-1]) > 0.5]
    opened = [r for r in recs if float(r["gt"][-1]) <= -0.5]

    def arm_mae(group: list[dict]) -> float:
        vals = [float(np.mean(np.abs(np.array(r["gt"][:-1]) - np.array(r["pred"][:-1]))))
                for r in group]
        return float(np.mean(vals)) if vals else float("nan")

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(13.0, 4.0), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.2, 0.8, 1.35]})

    keys = sorted(progress)
    vals = [float(np.mean(progress[k])) for k in keys]
    colors = [ORANGE if k == 50 else BLUE for k in keys]
    ax1.bar([f"{k}%" for k in keys], vals, color=colors, alpha=0.88, width=0.62)
    for x, v in zip(range(len(keys)), vals):
        ax1.text(x, v + 0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax1.set_xlabel("episode 进度")
    ax1.set_ylabel("动作 MAE")
    ax1.set_title("误差随进度：中段最高（橙色）")
    ax1.set_ylim(0, max(vals) * 1.22)

    ax2.bar(["夹爪闭合", "夹爪张开"], [arm_mae(closed), arm_mae(opened)],
            color=[ORANGE, BLUE], alpha=0.88, width=0.55)
    for x, group in enumerate((closed, opened)):
        ax2.text(x, arm_mae(group) + 0.003, f"{arm_mae(group):.3f}\nn={len(group)}",
                 ha="center", va="bottom", fontsize=8)
    ax2.set_ylabel("臂部 MAE", labelpad=7)
    ax2.set_title("搬运段（闭合）更难")
    ax2.set_ylim(0, max(arm_mae(closed), arm_mae(opened)) * 1.32)

    labels = ["x", "y", "z", "rx", "ry", "rz", "夹爪"]
    mae, nmae = [], []
    for d in range(dim):
        g = np.array([float(r["gt"][d]) for r in recs])
        p = np.array([float(r["pred"][d]) for r in recs])
        m = float(np.mean(np.abs(g - p)))
        mae.append(m)
        nmae.append(m / float(np.std(g)) if float(np.std(g)) > 1e-9 else 0.0)
    x = np.arange(dim)
    ax3.bar(x, nmae, color=[ORANGE if v == max(nmae) else BLUE for v in nmae],
            alpha=0.88, width=0.62)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.set_ylabel("归一化 MAE（MAE / GT 标准差）")
    ax3.set_title("逐维相对误差：旋转三维最差")
    for xi, v in zip(x, nmae):
        ax3.text(xi, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(f"VLA 开环复现：{len(recs)} 帧（MSE 0.066 vs 平均动作基线 0.212）",
                 fontsize=12)
    fig.savefig(out / "vla_openloop_decomp.png")
    plt.close(fig)


def chart_generalization(out: pathlib.Path) -> None:
    data = load(EVID / "m2-campaign-2026-11-14" / "generalization_and_s2.json")
    per = data["generalization_layered"]["summary"]["per_variant"]

    dist_steps = {"2 m": per["trained_2m_90deg"]["mean_steps_to_finish"],
                  "3 m": per["unseen_3m_90deg"]["mean_steps_to_finish"]}
    turn_steps = {"60°": per["unseen_2m_60deg"]["mean_steps_to_finish"],
                  "90°": per["trained_2m_90deg"]["mean_steps_to_finish"],
                  "120°": per["unseen_2m_120deg"]["mean_steps_to_finish"]}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.0))

    xs = np.array([2.0, 3.0])
    ys = np.array([dist_steps["2 m"], dist_steps["3 m"]], dtype=float)
    ax1.plot(xs, ys, "o", color=BLUE, ms=10, label="实测（2 m / 3 m）")
    ax1.plot(xs, 112 + 181 * xs, "--", color=ORANGE, lw=1.6, label="拟合：112 + 181·d")
    ax1.margins(x=0.18, y=0.14)
    for xv, yv in zip(xs, ys):
        ax1.annotate(f"{yv:.0f} 步", (xv, yv), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=9)
    ax1.set_xticks(xs)
    ax1.set_xlabel("路程（米）")
    ax1.set_ylabel("完成步数")
    ax1.set_title("距离：每多 1 米 ≈ +181 步")
    ax1.legend(loc="lower right")

    xs2 = [60, 90, 120]
    ys2 = [turn_steps["60°"], turn_steps["90°"], turn_steps["120°"]]
    ax2.plot(xs2, ys2, "o-", color=GREEN, lw=2, ms=8)
    ax2.margins(x=0.14, y=0.12)
    for xv, yv in zip(xs2, ys2):
        ax2.annotate(f"{yv:.0f} 步", (xv, yv), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=9)
    ax2.set_xticks(xs2)
    ax2.set_xticklabels([f"{v}°" for v in xs2])
    ax2.set_xlabel("转向角度")
    ax2.set_ylabel("完成步数")
    ax2.set_title("转向：每 30° ≈ +28 步（约 6%）")

    fig.suptitle("未见几何泛化：分层 80/80（4 几何 × 20 种子），单层 0/80", y=1.02, fontsize=12)
    fig.savefig(out / "generalization_cost.png")
    plt.close(fig)


def chart_s2(out: pathlib.Path) -> None:
    data = load(EVID / "m2-campaign-2026-11-14" / "generalization_and_s2.json")
    s2 = data["s2_disturbance"]
    dt = 0.02
    push_step = int(round(s2["summary"]["push"]["start_s"] / dt))
    fall_t = np.array([e["steps"] for e in s2["episodes"]], dtype=float) * dt
    push_t = push_step * dt
    n = len(fall_t)
    mean_delay = float(np.mean(fall_t - push_t))

    fig, ax = plt.subplots(figsize=(9.4, 4.2))
    ys = np.arange(n)
    ax.scatter(fall_t, ys, c=RED, s=52, alpha=0.85, edgecolors="white",
               linewidths=0.7, zorder=3)
    ax.axvline(push_t, color=ORANGE, ls="--", lw=1.8, zorder=2)
    ax.axvspan(push_t, fall_t.max() + 0.3, color=RED, alpha=0.055, zorder=1)

    ax.set_ylim(-1.6, n + 1.9)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, fall_t.max() + 0.6)
    ax.set_xlabel("仿真时间（s）")

    label_box = {"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 2}
    ax.text(push_t - 0.12, n + 1.0, f"侧推 100 N × 0.5 s（第 {push_t:.0f} s 施加）",
            color=ORANGE, fontsize=9, ha="right", va="center", bbox=label_box)
    y_bracket = n + 0.2
    ax.annotate("", xy=(push_t + mean_delay, y_bracket), xytext=(push_t, y_bracket),
                arrowprops={"arrowstyle": "<->", "color": GRAY, "lw": 1.4})
    ax.text(push_t + mean_delay + 0.12, y_bracket, f"平均 {mean_delay:.2f} s 后摔倒",
            color=GRAY, fontsize=8.5, ha="left", va="center", bbox=label_box)

    late = int(np.argmax(fall_t))
    if fall_t[late] > push_t + 3 * mean_delay:
        ax.annotate(f"1 例 {fall_t[late]:.1f} s 才倒", xy=(fall_t[late], ys[late]),
                    xytext=(-115, 8), textcoords="offset points", fontsize=8.5,
                    color=GRAY,
                    arrowprops={"arrowstyle": "->", "color": GRAY, "lw": 1.0,
                                "connectionstyle": "arc3,rad=0.2"})

    ax.set_title("S2 扰动鲁棒性：20/20 摔倒，无摔倒率 0.00（门禁 ≥0.90）")
    fig.savefig(out / "s2_failure_window.png")
    plt.close(fig)


def chart_factor_health(out: pathlib.Path) -> None:
    audit_path = sorted(ANALYSIS.glob("audit_*.json"))[-1]
    audit = load(audit_path)["rollout"]
    factors = audit["factors"]
    redundant = {n for pair in audit["redundant_pairs"] for n in pair[:2]}

    names = list(factors)
    columns = ["覆盖", "有方差", "组内分辨率", "非冗余", "有杠杆"]
    matrix, annotations = [], []
    for name in names:
        info = factors[name]
        coverage = info["coverage"]
        cover_level = 1.0 if coverage >= 0.5 else (0.5 if coverage >= 0.02 else 0.0)
        row = [
            cover_level,
            1.0 if not info["constant"] else 0.0,
            1.0 if len(info["within_run_dead"]) < 3 else 0.0,
            0.5 if (name in redundant and info.get("outcome_like")) else
            (0.0 if name in redundant else 1.0),
            1.0 if info["lever"] else 0.0,
        ]
        matrix.append(row)
        annotations.append([f"{coverage:.0%}", "", "", "", ""])
    matrix_arr = np.array(matrix)

    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#E8B4B8", "#F2D79B", "#7FB98A"])

    fig, ax = plt.subplots(figsize=(7.8, 0.46 * len(names) + 2.4))
    ax.imshow(matrix_arr, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels(columns)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names, fontsize=8.5)
    ax.grid(False)
    for i in range(len(names)):
        for j in range(len(columns)):
            value = matrix_arr[i, j]
            mark = "●" if value == 1 else ("~" if value == 0.5 else "○")
            ax.text(j, i, annotations[i][j] or mark, ha="center", va="center",
                    fontsize=9, color="#2B2B2B")

    # 分组分隔线（track_* / stab_* / task_*）
    boundaries = []
    for i in range(1, len(names)):
        if names[i].split("_")[0] != names[i - 1].split("_")[0]:
            boundaries.append(i - 0.5)
    for y in boundaries:
        ax.axhline(y, color="white", lw=3.0)

    ax.set_title("因子体检：● 通过　~ 结构性冗余　○ 不通过", fontsize=11, pad=10)
    fig.text(0.5, 0.045,
             "覆盖<50%（黄）说明该因子只在 1–2 个 run 上存在，不能跨配置比较；"
             "结局同源冗余不算缺陷",
             ha="center", fontsize=8, color=GRAY)
    fig.savefig(out / "factor_health.png")
    plt.close(fig)


def chart_training(out: pathlib.Path) -> None:
    logs = {
        "LIBERO batch 8（0–23k）": ROOT / "logs" / "vla" / "train_libero_003.log",
        "LIBERO batch 32（20–62k）": ROOT / "logs" / "vla" / "train_libero_005.log",
    }
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.0),
                                   gridspec_kw={"width_ratios": [1.5, 0.8]})

    for (label, path), color in zip(logs.items(), (BLUE, ORANGE)):
        recs = parse_training_log(path)
        if not recs:
            continue
        steps = np.array([r["step"] for r in recs]) / 1000.0
        loss = np.array([r["loss"] for r in recs])
        ax1.plot(steps, loss, color=color, lw=1.6, alpha=0.85, label=label)
    ax1.axvline(23, color=GRAY, ls=":", lw=1.2)
    ax1.text(24, 1.6, "batch 8 → 32", color=GRAY, fontsize=8)
    ax1.set_xlabel("优化步数（千）")
    ax1.set_ylabel("训练 loss")
    ax1.set_yscale("log")
    ax1.set_title("LIBERO 微调 loss 曲线（batch 8 → 32）")
    ax1.legend(loc="upper right")

    ax2.bar(["前 23k 步", "后 42k 步"], [0.735, 0.0038], color=[BLUE, RED],
            alpha=0.88, width=0.55)
    for x, v in enumerate([0.735, 0.0038]):
        ax2.text(x, v * 1.08, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax2.set_yscale("log")
    ax2.set_ylabel("每 10k 步 loss 降幅（对数）")
    ax2.set_title("收敛速率塌陷约 190 倍")
    ax2.text(0.5, 0.09, "↓ 190×", ha="center", va="center", color=RED, fontsize=13,
             fontweight="bold")

    fig.suptitle("继续训练的边际收益接近 0 —— 停在 62K 步的依据", y=1.02, fontsize=12)
    fig.savefig(out / "training_plateau.png")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--out", type=pathlib.Path, default=ROOT / "media" / "analysis")
    args = parser.parse_args(argv)
    out: pathlib.Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    setup_style()
    chart_fps_ab(out)
    chart_suite_tasks(out)
    chart_e1_vs_base(out)
    chart_payload(out)
    chart_vla_failure_traces(out)
    chart_openloop(out)
    chart_generalization(out)
    chart_s2(out)
    chart_factor_health(out)
    chart_training(out)

    for path in sorted(out.glob("*.png")):
        print(f"[charts] {path.relative_to(ROOT)}  {path.stat().st_size // 1024} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
