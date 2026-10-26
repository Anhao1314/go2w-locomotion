#!/usr/bin/env python3
"""Plot convergence curves for all six L3 training seeds (pure CPU)."""

from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]

RUNS = {
    "single": [
        ("seed 0", ROOT / "logs/rsl_rl/2026-10-24_01-17-59/convergence_log.jsonl"),
        ("seed 1", ROOT / "logs/rsl_rl/2026-10-26_23-36-23/convergence_log.jsonl"),
        ("seed 2", ROOT / "logs/rsl_rl/2026-10-27_00-18-16/convergence_log.jsonl"),
    ],
    "layered": [
        ("seed 0", ROOT / "logs/rsl_rl_layered/2026-10-24_01-18-00/convergence_log.jsonl"),
        ("seed 1", ROOT / "logs/rsl_rl_layered/2026-10-26_23-36-22/convergence_log.jsonl"),
        ("seed 2", ROOT / "logs/rsl_rl_layered/2026-10-26_23-41-53/convergence_log.jsonl"),
    ],
}

fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=120)

for group, color, marker, label in (
    ("single", "#ff5c5c", "o", "single layer (12-dim joints)"),
    ("layered", "#3ddc97", "s", "layered (2-dim commands -> frozen S0)"),
):
    for seed, path in RUNS[group]:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        xs = [r["iteration"] for r in rows]
        ys = [r["completion_rate"] * 100 for r in rows]
        ax.plot(xs, ys, marker=marker, markersize=4, linewidth=1.6, alpha=0.85,
                color=color, label=f"{label} {seed}" if seed == "seed 0" else None)

ax.set_xlabel("training iterations")
ax.set_ylabel("deterministic completion rate [%] (2048 envs)")
ax.set_ylim(-4, 104)
ax.grid(alpha=0.3)
ax.legend(loc="center right", fontsize=9)
ax.set_title("L3 convergence across training seeds:\nlayered converges on every seed; the single layer is seed-dependent",
             fontsize=12, loc="left")
fig.tight_layout()
out = ROOT / "media/training/convergence_all_seeds.png"
fig.savefig(out)
print(f"wrote {out}")
