#!/usr/bin/env python3
"""预计算每个示范 episode 的目标图像（GIFT-lite 数据侧）。

目标图像 = 该 episode 最后一帧的前视相机画面（任务完成态：碗已在盘上）。
训练时把 [当前画面; 目标画面] 垂直拼接作为 image 输入（GIFT 的单卡简化版：
不改架构，用图像拼接注入目标条件）。

同时为评测任务生成 task{t}.png：按任务字符串匹配任意一段示范的末帧
（init_states 固定 → 同任务目标状态一致）。

输出：logs/vla/goal_images/ep{N}.png 与 task{t}.png
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

SNAP = pathlib.Path(
    "/root/.cache/huggingface/lerobot/hub/datasets--lerobot--libero/snapshots/"
    "a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4"
)

# 评测任务 → 数据集任务串（与 evaluate_libero 的 info['task'] 一致）
EVAL_TASKS = {
    0: "pick up the black bowl between the plate and the ramekin and place it on the plate",
    1: "pick up the black bowl next to the ramekin and place it on the plate",
    2: "pick up the black bowl from table center and place it on the plate",
    3: "pick up the black bowl on the cookie box and place it on the plate",
    4: "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 个 episode（调试）")
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # pyright: ignore[reportMissingImports]
    import numpy as np  # noqa: F401

    ds = LeRobotDataset("lerobot/libero", root=SNAP)
    eps_meta = ds.meta.episodes
    pass
    # task_index → 任务串（meta.tasks 是 pandas DataFrame：索引=任务串）
    task_str = {}
    for tstr, row in ds.meta.tasks.iterrows():
        task_str[int(row["task_index"])] = str(tstr)

    out_dir = pathlib.Path("logs/vla/goal_images")
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(eps_meta)
    limit = args.limit or n
    # ep → task_index（从数据 parquet 重建；episodes 元数据不含 task 列）
    import pyarrow.parquet as pq  # pyright: ignore[reportMissingImports]
    ep_task: dict[int, int] = {}
    for path in sorted((SNAP / "data").rglob("*.parquet")):
        t = pq.read_table(path, columns=["episode_index", "task_index"])
        for ep, ti in zip(t.column("episode_index").to_numpy(),
                          t.column("task_index").to_numpy()):
            ep_task[int(ep)] = int(ti)
    print(f"[goal] ep->task map: {len(ep_task)} episodes")

    done = 0
    goal_by_task: dict[str, int] = {}
    for row in eps_meta:
        ep = int(row["episode_index"])
        if ep >= limit:
            break
        out = out_dir / f"ep{ep}.png"
        if not out.exists():
            last_idx = int(row["dataset_to_index"]) - 1
            item = ds[last_idx]
            img = item["observation.images.image"]
            # CHW float [0,1] → HWC uint8
            import torch  # pyright: ignore[reportMissingImports]

            if img.ndim == 3 and img.shape[0] == 3:
                arr = (img.permute(1, 2, 0).numpy() * 255).astype("uint8")
            else:
                arr = np.asarray(img)
                if arr.ndim == 4:
                    arr = arr[0]
                arr = arr[..., ::-1] if arr.shape[-1] == 3 else arr
            import imageio  # pyright: ignore[reportMissingImports]

            imageio.imwrite(out, arr)
        done += 1
        if done % 200 == 0:
            print(f"[goal] {done}/{min(n, limit)}", flush=True)
        # 评测任务映射：该 episode 的任务串 ∈ EVAL_TASKS → 记录候选
        ti = ep_task.get(ep)
        tstr = task_str.get(ti)
        if tstr in EVAL_TASKS.values():
            tid = [k for k, v in EVAL_TASKS.items() if v == tstr][0]
            goal_by_task.setdefault(str(tid), ep)

    for tid, ep in sorted(goal_by_task.items()):
        src = out_dir / f"ep{ep}.png"
        dst = out_dir / f"task{tid}.png"
        dst.write_bytes(src.read_bytes())
        print(f"[goal] task{tid}.png <- episode {ep}")

    (out_dir / "manifest.json").write_text(
        json.dumps({"episodes": done, "goal_by_task": goal_by_task,
                    "snapshot": str(SNAP)}), encoding="utf-8")
    print(f"[goal] done: {done} episodes -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
