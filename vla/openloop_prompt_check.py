#!/usr/bin/env python3
"""技能提示敏感性检查：模型到底理不理"技能前缀"？

分层 VLA 的第一前提不是数据能切段，而是**策略对技能条件有反应**。若把同一帧
配上 "approach" / "transport" 前缀后输出几乎不变，那么分支条件化只能靠重训，
不能靠提示工程——这决定了下一步要不要花几小时训练。

做法：同一批帧，分别用三种提示各跑一次前向，比较动作差异：
  (a) 原始任务串
  (b) "approach | 原始任务串"
  (c) "transport | 原始任务串"
参照尺度：遮住图像造成的动作偏移（openloop_check 里约 0.49）——提示带来的偏移
若远小于它，说明文本通道几乎没被使用。

用法（WSL vla-env，离线）：
    HF_HUB_OFFLINE=1 python vla/openloop_prompt_check.py \
        --checkpoint logs/vla/libero_smolvla/checkpoints/060000/pretrained_model \
        --dataset-root /root/.cache/.../snapshots/<rev> --frames 24
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--dataset-repo", type=str, default="lerobot/libero")
    parser.add_argument("--dataset-root", type=pathlib.Path, required=True)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--out", type=pathlib.Path,
                        default=PROJECT_ROOT / "logs" / "vla" / "prompt_check")
    args = parser.parse_args(argv)

    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    import numpy as np
    import torch  # pyright: ignore[reportMissingImports]
    from lerobot.configs.policies import PreTrainedConfig  # pyright: ignore[reportMissingImports]
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # pyright: ignore[reportMissingImports]
    from lerobot.policies import get_policy_class, make_pre_post_processors  # pyright: ignore[reportMissingImports]

    ckpt = pathlib.Path(args.checkpoint)
    cfg = PreTrainedConfig.from_pretrained(str(ckpt))
    cfg.pretrained_path = str(ckpt)
    device = cfg.device or "cuda"
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg, pretrained_path=str(ckpt),
        preprocessor_overrides={"device_processor": {"device": device}})
    policy = get_policy_class(cfg.type).from_pretrained(str(ckpt), config=cfg)
    policy.to(device)
    policy.eval()

    ds = LeRobotDataset(args.dataset_repo, root=args.dataset_root)
    step = max(1, len(ds) // args.frames)
    indices = [i * step for i in range(args.frames)]

    def task_of(item: dict) -> str:
        task = item.get("task")
        if isinstance(task, str):
            return task
        tasks = getattr(ds.meta, "tasks", None)
        ti = item.get("task_index")
        if tasks is not None and ti is not None:
            try:
                return str(tasks.loc[int(ti)]["task"])
            except Exception:  # noqa: BLE001
                pass
        return ""

    def predict(item: dict, prompt: str) -> np.ndarray:
        batch = {
            "observation.images.image": torch.as_tensor(item["observation.images.image"]),
            "observation.images.image2": torch.as_tensor(item["observation.images.image2"]),
            "observation.state": torch.as_tensor(
                np.asarray(item["observation.state"], dtype=np.float32)),
            "task": prompt,
        }
        processed = preprocessor(batch)
        policy.reset()
        with torch.inference_mode():
            action = policy.select_action(processed)
        action = postprocessor(action)
        arr = action.detach().cpu().numpy() if hasattr(action, "detach") else np.asarray(action)
        return np.asarray(arr, dtype=np.float64).reshape(-1)

    diffs = {"approach": [], "transport": []}
    grip_sign_flip = {"approach": 0, "transport": 0}
    records = []
    for index in indices:
        item = ds[index]
        base_task = task_of(item)
        base = predict(item, base_task)
        preds = {}
        for skill in ("approach", "transport"):
            pred = predict(item, f"{skill} | {base_task}")
            preds[skill] = pred
            diffs[skill].append(float(np.linalg.norm(pred - base)))
            grip_sign_flip[skill] += int(np.sign(pred[6]) != np.sign(base[6]))
        records.append({"index": int(index), "base": base.tolist(),
                        "approach": preds["approach"].tolist(),
                        "transport": preds["transport"].tolist()})

    summary = {
        "frames": len(indices),
        "prompt_delta_l2": {k: float(np.mean(v)) for k, v in diffs.items()},
        "gripper_sign_flip_rate": {
            k: v / max(1, len(indices)) for k, v in grip_sign_flip.items()},
        "reference_scale": {
            "vision_blind_delta_l2": 0.489,
            "note": "遮住图像造成的动作偏移，用于判断提示效应是否可观",
        },
        "records": records,
    }
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "prompt_check.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "records"},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
