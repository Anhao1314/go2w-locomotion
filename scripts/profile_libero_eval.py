#!/usr/bin/env python3
"""LIBERO 评测的耗时分解：env.step / 渲染 / 策略前向各占多少。

只有先量清楚，才知道 5 任务 × 20 集该优化哪一块：
  * 若 env.step 占大头 → 降渲染分辨率 / 复用场景 / 多进程并行
  * 若策略前向占大头 → 减 flow-matching 步数 / bf16 / torch.compile

用法（WSL vla-env）：
    MUJOCO_GL=egl python scripts/profile_libero_eval.py --steps 30
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402


def time_env(height: int, width: int, steps: int) -> dict[str, float]:
    from vla.env import make_libero_env

    env = make_libero_env("libero_spatial", task_ids=[0], fps=20,
                          observation_height=height, observation_width=width)
    env.reset(seed=[1000])
    action = np.zeros((1, 7), dtype=np.float32)

    # 预热（首步含大量惰性初始化，不计入）
    for _ in range(3):
        env.step(action)

    t_reset = time.perf_counter()
    env.reset(seed=[1001])
    reset_s = time.perf_counter() - t_reset

    last_obs: dict = {}
    t0 = time.perf_counter()
    for _ in range(steps):
        last_obs, *_ = env.step(action)
    step_s = (time.perf_counter() - t0) / steps

    # 单独的渲染成本：再拿一次观测里的图像（已渲染，但转换/拷贝有成本）
    t1 = time.perf_counter()
    for _ in range(steps):
        img = np.asarray(last_obs["observation.images.image"])
        img2 = np.asarray(last_obs["observation.images.image2"])
        _ = (img.shape, img2.shape)
    copy_s = (time.perf_counter() - t1) / steps

    if hasattr(env, "close"):
        env.close()
    return {"height": height, "step_s": step_s, "reset_s": reset_s, "copy_s": copy_s}


def time_policy(steps: int) -> dict[str, float]:
    import torch  # pyright: ignore[reportMissingImports]
    from lerobot.configs.policies import PreTrainedConfig  # pyright: ignore[reportMissingImports]
    from lerobot.policies import get_policy_class, make_pre_post_processors  # pyright: ignore[reportMissingImports]

    ckpt = PROJECT_ROOT / "logs/vla/libero_smolvla/checkpoints/060000/pretrained_model"
    cfg = PreTrainedConfig.from_pretrained(str(ckpt))
    cfg.pretrained_path = str(ckpt)
    device = cfg.device or "cuda"
    pre, post = make_pre_post_processors(
        policy_cfg=cfg, pretrained_path=str(ckpt),
        preprocessor_overrides={"device_processor": {"device": device}})
    policy = get_policy_class(cfg.type).from_pretrained(str(ckpt), config=cfg)
    policy.to(device).eval()

    batch = {
        "observation.images.image": torch.rand(3, 360, 360, device=device),
        "observation.images.image2": torch.rand(3, 360, 360, device=device),
        "observation.state": torch.rand(8, device=device),
        "task": "pick up the black bowl",
    }
    processed = pre(batch)
    with torch.inference_mode():
        policy.reset()
        policy.select_action(processed)  # 预热

    t0 = time.perf_counter()
    for _ in range(steps):
        policy.reset()
        with torch.inference_mode():
            policy.select_action(processed)
    infer_s = (time.perf_counter() - t0) / steps
    return {"infer_s": infer_s}


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()

    print(f"=== env 步进（{args.steps} 步）===", flush=True)
    for hw in ((360, 360), (256, 256), (224, 224)):
        res = time_env(hw[0], hw[1], args.steps)
        print(f"  {hw[0]}x{hw[1]}: step={res['step_s'] * 1000:.1f} ms  "
              f"reset={res['reset_s'] * 1000:.0f} ms  copy={res['copy_s'] * 1000:.2f} ms",
              flush=True)

    print("=== 策略前向 ===", flush=True)
    pol = time_policy(5)
    print(f"  select_action（每 50 步一次）= {pol['infer_s'] * 1000:.1f} ms", flush=True)

    per_step = None
    for hw in ((360, 360),):
        per_step = time_env(hw[0], hw[1], 10)["step_s"]
    if per_step:
        ep_280 = per_step * 280 + pol["infer_s"] * 6
        print(f"\n估算：单集（280 步）≈ {ep_280:.1f} s → "
              f"100 集 ≈ {ep_280 * 100 / 60:.0f} 分钟（单进程）", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
