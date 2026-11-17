#!/usr/bin/env python3
"""冒烟：验证 sitecustomize 目标图像拼接（需 VLA_GOAL_IMAGES=1 + VLA_GOAL_DIR）。"""

from __future__ import annotations

import pathlib
import sys

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # pyright: ignore[reportMissingImports]

SNAP = pathlib.Path(
    "/root/.cache/huggingface/lerobot/hub/datasets--lerobot--libero/snapshots/"
    "a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4"
)

ds = LeRobotDataset("lerobot/libero", root=SNAP)
item = ds[100]
img = item["observation.images.image"]
print("composed image shape:", tuple(img.shape), "| dtype:", img.dtype,
      "| range:", float(img.min()), "-", float(img.max()))
assert img.shape[0] == 3 and img.shape[1] == 512, "expected (3, 512, 256) vertical stack"

# 保存为 png 目检（上半=当前，下半=目标）
import numpy as np  # noqa: E402

arr = (img.permute(1, 2, 0).numpy() * 255).astype("uint8")
out = pathlib.Path("logs/vla/frames/goal_compose_check.png")
out.parent.mkdir(parents=True, exist_ok=True)
import imageio  # pyright: ignore[reportMissingImports]

imageio.imwrite(out, arr)
print("SMOKE PASS:", out)
sys.exit(0)
