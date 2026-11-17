#!/usr/bin/env python3
"""冒烟：验证 sitecustomize 技能标签注入真的生效（需 VLA_SKILL_LABELS=1）。"""

from __future__ import annotations

import json
import pathlib
import sys

labels = json.loads(pathlib.Path("logs/vla/skill_labels.json").read_text(encoding="utf-8"))
table = labels["labels"]

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

SNAP = pathlib.Path(
    "/root/.cache/huggingface/lerobot/hub/datasets--lerobot--libero/snapshots/"
    "a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4"
)
ds = LeRobotDataset("lerobot/libero", root=SNAP)
eps_meta = ds.meta.episodes
by_ep = {int(r["episode_index"]): r for r in eps_meta}

ep = next(e for e, v in table.items() if v["t1"] is not None)
t1, t2, frames = table[ep]["t1"], table[ep]["t2"], table[ep]["frames"]
base = int(by_ep[int(ep)]["dataset_from_index"])
print(f"episode {ep}: t1={t1} t2={t2} frames={frames} base={base}")

checks = [("approach", 5), ("transport", (t1 + t2) // 2), ("retreat", min(t2 + 3, frames - 1))]
ok = 0
for name, fi in checks:
    item = ds[base + fi]
    text = item["task"]
    hit = text.startswith(name)
    ok += hit
    print(f"  frame {fi:>4} ({name:>9}): {'PASS' if hit else 'FAIL'} | {text[:75]}")

uep = next(e for e, v in table.items() if v["t1"] is None)
uitem = ds[int(by_ep[int(uep)]["dataset_from_index"]) + 5]
utext = uitem["task"]
u_ok = not utext.startswith(("approach", "transport", "retreat"))
print(f"  unknown ep {uep}: {'PASS' if u_ok else 'FAIL'} | {utext[:75]}")

print("SMOKE", "PASS" if ok == 3 and u_ok else "FAIL")
sys.exit(0 if ok == 3 and u_ok else 1)
