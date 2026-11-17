"""训练时技能标签注入（sitecustomize，opt-in）。

仅当环境变量 VLA_SKILL_LABELS=1 且 VLA_SKILL_LABELS_FILE 指向标签文件时生效：
给 LeRobotDataset.__getitem__ 打补丁，把当前帧的技能阶段作为语言前缀拼进
task 字符串（"transport | pick up the black bowl ..."）。

- 标签规则与 scripts/build_skill_labels.py 一致（模板 episode：approach/transport/
  retreat；非模板/unknown → 保持原任务串）。
- 不改数据文件、不动原始数据集；训练管线的其余部分零侵入。
- 无环境变量时完全不生效（评测与其他工具不受影响）。
"""

from __future__ import annotations

import json
import os
import pathlib


def _apply_patch() -> None:
    if os.environ.get("VLA_SKILL_LABELS") != "1":
        return
    label_file = os.environ.get("VLA_SKILL_LABELS_FILE")
    if not label_file or not pathlib.Path(label_file).exists():
        return

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    payload = json.loads(pathlib.Path(label_file).read_text(encoding="utf-8"))
    labels = payload["labels"]

    def phase_of(ep: int, frame: int) -> str | None:
        entry = labels.get(str(ep))
        if not entry:
            return None
        t1, t2 = entry["t1"], entry["t2"]
        if t1 is None or t2 is None:
            return None
        if frame < t1:
            return "approach"
        if frame < t2:
            return "transport"
        return "retreat"

    original = LeRobotDataset.__getitem__

    def patched(self, idx):  # noqa: ANN001, ANN202
        item = original(self, idx)
        task = item.get("task")
        if isinstance(task, str) and task and not task.startswith(("approach", "transport", "retreat")):
            ep = int(item.get("episode_index", -1))
            frame = int(item.get("frame_index", -1))
            phase = phase_of(ep, frame)
            if phase:
                item["task"] = f"{phase} | {task}"
        return item

    LeRobotDataset.__getitem__ = patched
    print(f"[skill-labels] active: {label_file}", flush=True)


_apply_patch()
