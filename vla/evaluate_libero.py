#!/usr/bin/env python3
"""LIBERO closed-loop evaluation with an explicit, verified observation contract.

Why this file exists (2026-11-15): the first version fed the policy raw robosuite
observations — uint8 HWC (1,360,360,3) with a batch axis and a tuple task string —
while SmolVLA was trained on float CHW [0,1] 256x256 tensors.  The policy was
therefore looking at garbage.  This version normalises the contract in one place
and records enough diagnostics to tell "policy is wrong" from "interface is wrong".

Fair A/B of control frequency: pass the SAME ``--time-budget-s`` and let the step
cap follow from it (``steps = time_budget * fps``), because the env caps episodes
at a fixed number of STEPS (280), not seconds.

Usage (WSL vla-env, offline):
    MUJOCO_GL=egl python vla/evaluate_libero.py \
        --checkpoint logs/vla/libero_smolvla/checkpoints/060000/pretrained_model \
        --episodes 3 --suite libero_spatial --task-ids 0 \
        --fps 10 --time-budget-s 14 --out logs/vla/libero_ab_fps10
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _as_image(value: Any) -> Any:
    """robosuite obs → the exact tensor layout the policy was trained on."""
    import numpy as np
    import torch

    arr = np.asarray(value)
    if arr.ndim == 4:          # (B, H, W, C) vectorised wrapper
        arr = arr[0]
    if arr.dtype == np.uint8:  # 0..255 → 0..1
        arr = arr.astype(np.float32) / 255.0
    if arr.ndim == 3 and arr.shape[-1] == 3:  # HWC → CHW
        arr = np.transpose(arr, (2, 0, 1))
    return torch.as_tensor(np.ascontiguousarray(arr, dtype=np.float32))


def _as_state(value: Any) -> Any:
    import numpy as np
    import torch

    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 2:  # (B, D) → (D,)
        arr = arr[0]
    return torch.as_tensor(arr)


def _as_task(info: dict[str, Any]) -> str:
    task = info.get("task")
    if isinstance(task, (tuple, list)):
        return str(task[0]) if task else ""
    return "" if task is None else str(task)


def _flag(value: Any) -> bool:
    import numpy as np

    if value is None:
        return False
    arr = np.asarray(value)
    return bool(arr.any()) if arr.size else False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "usage").splitlines()[0])
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--suite", type=str, default="libero_spatial")
    parser.add_argument("--task-ids", type=str, default="0",
                        help="task indices within the suite, comma-separated")
    parser.add_argument("--fps", type=int, default=20,
                        help="env control frequency (robosuite control_freq)")
    parser.add_argument("--time-budget-s", type=float, default=14.0,
                        help="simulated seconds per episode; step cap = budget * fps")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="override the derived step cap")
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument("--video", action="store_true", help="record the first episode")
    parser.add_argument("--out", type=pathlib.Path,
                        default=PROJECT_ROOT / "logs" / "vla" / "libero_eval")
    args = parser.parse_args(argv)

    import os

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    import numpy as np
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies import get_policy_class, make_pre_post_processors

    from vla.env import make_libero_env

    max_steps = args.max_steps or int(round(args.time_budget_s * args.fps))

    ckpt = pathlib.Path(args.checkpoint)
    policy_config = PreTrainedConfig.from_pretrained(str(ckpt))
    policy_config.pretrained_path = str(ckpt)
    device = policy_config.device or "cuda"
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_config,
        pretrained_path=str(ckpt),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    policy = get_policy_class(policy_config.type).from_pretrained(
        str(ckpt), config=policy_config
    )
    policy.to(device)
    policy.eval()

    task_ids = [int(t) for t in args.task_ids.split(",")]
    env = make_libero_env(args.suite, task_ids=task_ids, fps=args.fps)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    video_frames: list[np.ndarray] = []

    for episode in range(args.episodes):
        seed = args.seed_base + episode
        obs, info = env.reset(seed=[seed])
        policy.reset()
        task = _as_task(info if isinstance(info, dict) else {})

        steps = 0
        success = False
        action_sum = np.zeros(7, dtype=np.float64)
        gripper_closed_steps = 0
        while steps < max_steps:
            batch = {
                "observation.images.image": _as_image(obs["observation.images.image"]),
                "observation.images.image2": _as_image(obs["observation.images.image2"]),
                "observation.state": _as_state(obs["observation.state"]),
                "task": task,
            }
            processed = preprocessor(batch)
            with torch.inference_mode():
                action = policy.select_action(processed)
            if action.ndim == 1:
                action = action.unsqueeze(0)
            action = postprocessor(action)
            action_np = (action.detach().cpu().numpy()
                         if hasattr(action, "detach") else np.asarray(action))
            action_np = np.asarray(action_np, dtype=np.float32)
            if action_np.ndim == 1:
                action_np = action_np[None, :]
            action_sum += np.abs(action_np[0][:7])
            if float(action_np[0][6]) > 0.0:
                gripper_closed_steps += 1

            obs, reward, terminated, truncated, info = env.step(action_np)
            steps += 1
            if args.video and episode == 0:
                raw = np.asarray(obs["observation.images.image"])
                frame = raw[0] if raw.ndim == 4 else raw
                video_frames.append(frame.astype(np.uint8).copy())
            if isinstance(info, dict) and _flag(info.get("is_success")):
                success = True
                break
            if _flag(terminated) or _flag(truncated):
                break

        results.append({
            "episode": episode,
            "seed": seed,
            "steps": steps,
            "success": success,
            "time_s": steps / args.fps,
            "mean_abs_action": (action_sum / max(1, steps)).round(4).tolist(),
            "gripper_closed_fraction": gripper_closed_steps / max(1, steps),
        })
        print(f"[eval] fps={args.fps} episode {episode} seed {seed}: "
              f"{'SUCCESS' if success else 'incomplete'} "
              f"({steps} steps / {steps / args.fps:.1f}s sim)", flush=True)

    if args.video and video_frames:
        import imageio

        video_path = out / "libero_eval_episode0.mp4"
        with imageio.get_writer(video_path, fps=args.fps, codec="libx264", quality=8) as writer:
            for frame in video_frames:
                writer.append_data(frame)
        print(f"      video: {video_path}", flush=True)

    successes = sum(1 for r in results if r["success"])
    summary = {
        "checkpoint": str(ckpt),
        "suite": args.suite,
        "task_ids": task_ids,
        "fps": args.fps,
        "time_budget_s": args.time_budget_s,
        "max_steps": max_steps,
        "episodes": args.episodes,
        "successes": successes,
        "success_rate": successes / max(1, args.episodes),
        "mean_steps": float(np.mean([r["steps"] for r in results])),
        "results": results,
        "obs_contract": {
            "images": "float32 CHW [0,1] (uint8 HWC normalised)",
            "state": "float32 (D,)",
            "task": "str",
        },
    }
    (out / "libero_eval_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
