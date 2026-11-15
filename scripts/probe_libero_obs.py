#!/usr/bin/env python3
"""Probe the LIBERO env observation contract before running any A/B.

Checks, for both control frequencies:
  * obs keys / shapes / dtypes / value ranges
  * whether images are CHW float [0,1] (what SmolVLA was trained on) or
    HWC uint8 [0,255] (what raw robosuite returns)
  * task string availability, env episode cap, action space bounds

Usage: MUJOCO_GL=egl python scripts/probe_libero_obs.py
"""

from __future__ import annotations

import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from vla.env import make_libero_env  # noqa: E402


def describe(value: object) -> str:
    arr = np.asarray(value)
    if arr.dtype.kind in "fiu":
        return (f"shape={arr.shape} dtype={arr.dtype} "
                f"min={float(arr.min()):.4f} max={float(arr.max()):.4f}")
    return f"type={type(value).__name__} value={value!r}"


def main() -> int:
    for fps in (10, 20):
        print(f"\n=== fps={fps} ===", flush=True)
        env = make_libero_env("libero_spatial", task_ids=[0], fps=fps)
        obs, info = env.reset(seed=[1000])
        print("obs keys:", sorted(obs.keys()))
        for key in sorted(obs.keys()):
            print(f"  {key}: {describe(obs[key])}")
        if isinstance(info, dict):
            for key in ("task", "is_success", "task_id", "task_description"):
                if key in info:
                    print(f"  info[{key!r}] = {info[key]!r}")
            print("  info keys:", sorted(info.keys())[:12])
        action_space = getattr(env, "action_space", None)
        if action_space is not None:
            print(f"  action_space: {action_space}")
        unwrapped = getattr(env, "unwrapped", env)
        for attr in ("control_freq", "_control_freq", "max_steps", "_max_steps", "fps"):
            if hasattr(unwrapped, attr):
                print(f"  env.{attr} = {getattr(unwrapped, attr)}")
        zero = np.zeros(action_space.shape, dtype=np.float32) if action_space is not None else None
        if zero is not None:
            obs2, reward, terminated, truncated, info2 = env.step(zero)
            print(f"  zero-action step: reward={reward} terminated={terminated} "
                  f"truncated={truncated} img_changed="
                  f"{not np.array_equal(np.asarray(obs['observation.images.image']), np.asarray(obs2['observation.images.image']))}")
        env.close() if hasattr(env, "close") else None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
