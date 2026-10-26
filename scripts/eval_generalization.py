#!/usr/bin/env python3
"""Generalisation test: evaluate trained policies on unseen task geometries.

Both groups were trained with left turns (+90 deg) over 2 m legs only.  This
script re-evaluates the frozen checkpoints on geometries they have never seen
(right turn, longer legs, both) by patching the task constants at runtime --
the MDP reads them per step, so no retraining and no code change is needed.

Configs (per group):
    baseline      turn +90, leg 2.0 m   (seen in training)
    right-turn    turn -90, leg 2.0 m   (unseen direction)
    long-leg      turn +90, leg 3.0 m   (unseen length)
    right+long    turn -90, leg 3.0 m   (both unseen)

Usage::

    PYTHONPATH=. OMNI_KIT_ACCEPT_EULA=YES \
      .venv311/Scripts/python.exe scripts/eval_generalization.py \
        --single-checkpoint <A model.pt> --layered-checkpoint <B model.pt> \
        --s0-checkpoint <S0 model.pt> --episodes 20 --out <json>

Runs only inside the Isaac Lab environment (``.venv311``).
"""

# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from typing import Any, cast

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from locomotion.eval.seeds import EVAL_SEEDS  # noqa: E402

CONFIGS = (
    ("baseline(+90deg,2m)", 1.5707963267948966, 2.0),
    ("right-turn(-90deg,2m)", -1.5707963267948966, 2.0),
    ("long-leg(+90deg,3m)", 1.5707963267948966, 3.0),
    ("right+long(-90deg,3m)", -1.5707963267948966, 3.0),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Go2Loco-Sequence-v0")
    parser.add_argument("--single-checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--layered-checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--s0-checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--episodes", type=int, default=len(EVAL_SEEDS))
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)

    seeds = list(EVAL_SEEDS[: args.episodes])

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import torch

    import locomotion.isaac  # noqa: F401
    import locomotion.sequence as seq
    from locomotion.sequence import mdp as seq_mdp
    from locomotion.sequence.layered_env import LayeredSequenceEnv, load_s0_actor
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from rsl_rl.runners import OnPolicyRunner

    device = "cuda:0"

    def build_env(layered: bool, num_envs: int):
        env_cfg = parse_env_cfg(args.task, device=device, num_envs=num_envs, use_fabric=True)
        env_cfg.seed = 0
        env = gym.make(args.task, cfg=env_cfg)
        if layered:
            s0_actor = load_s0_actor(args.s0_checkpoint, device=device)
            env = LayeredSequenceEnv(env, s0_actor, device=device)
        vec_env = RslRlVecEnvWrapper(env)
        if layered:
            vec_env.num_actions = int(gym.spaces.flatdim(env.action_space))
        runner_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
        runner = OnPolicyRunner(vec_env, runner_cfg.to_dict(), log_dir=None, device=device)
        return env, vec_env, runner, runner_cfg

    def evaluate(runner: OnPolicyRunner, inner: Any, vec_env: Any, steps_max: int) -> float:
        num_envs = vec_env.num_envs
        inner._seq_success_log = set()
        active = np.ones(num_envs, dtype=bool)
        with torch.inference_mode():
            policy = runner.get_inference_policy(device=device)
            obs, _ = vec_env.reset()
            for _ in range(steps_max):
                actions = policy(obs)
                obs, _, dones, _ = vec_env.step(actions)
                dones_np = dones.cpu().numpy().astype(bool) if torch.is_tensor(dones) else np.asarray(dones, dtype=bool)
                active &= ~dones_np
                if not active.any():
                    break
        return len(getattr(inner, "_seq_success_log", set())) / num_envs

    num_envs = len(seeds)
    results: list[dict[str, Any]] = []

    for group, checkpoint, layered in (
        ("layered", args.layered_checkpoint, True),
        ("single", args.single_checkpoint, False),
    ):
        env, vec_env, runner, runner_cfg = build_env(layered, num_envs)
        inner = cast(Any, env.unwrapped)
        steps_max = int(inner.max_episode_length)
        runner.load(str(checkpoint))

        for label, turn, distance in CONFIGS:
            # runtime patch: the MDP reads these constants on every step, so a
            # reset under the patched values fully re-geometries the task
            seq_mdp.TURN_RADIANS = turn
            seq_mdp.FORWARD_DISTANCE_M = distance
            seen = "seen" if label.startswith("baseline") else "UNSEEN"
            started = time.time()
            rate = evaluate(runner, inner, vec_env, steps_max)
            results.append(
                {
                    "group": group,
                    "config": label,
                    "seen": seen,
                    "success_rate": round(rate, 3),
                    "episodes": num_envs,
                    "eval_time_s": round(time.time() - started, 1),
                }
            )
            print(
                f"[gen] {group:>7} | {label:<20} ({seen}) -> success {rate * 100:5.1f}%",
                flush=True,
            )
        vec_env.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"results": results}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[gen] wrote {args.out}", flush=True)

    simulation_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
