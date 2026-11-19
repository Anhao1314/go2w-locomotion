#!/usr/bin/env python3
"""S2b 轻地形冒烟：环境实例化 + 200 步滚动 + 真实仿真截图。

验证四件事：
  1. Go2Loco-S2b-LightTerrain-v0 可实例化（轻地形生成器挂载正确）
  2. 高度扫描器确认为盲（None），子地形集合确认为轻档
  3. 200 步滚动无崩溃、无全局 NaN
  4. 相机抓帧保存 PNG（真实仿真画面）

用法：PYTHONPATH=. OMNI_KIT_ACCEPT_EULA=YES .venv311/Scripts/python.exe scripts/smoke_s2b_env.py
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from isaaclab.app import AppLauncher  # noqa: E402

app_launcher = AppLauncher(headless=True)  # enable_cameras 在 Windows headless 下段错误，截图改用离线高度场预览
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402


def main() -> int:
    import isaaclab_tasks  # noqa: F401
    import locomotion.isaac  # noqa: F401
    import locomotion.sequence  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    device = "cuda:0"
    env_cfg = parse_env_cfg("Go2Loco-S2b-LightTerrain-v0", device=device,
                            num_envs=16, use_fabric=True)
    env = gym.make("Go2Loco-S2b-LightTerrain-v0", cfg=env_cfg, render_mode="rgb_array")
    inner = env.unwrapped

    checks: dict[str, bool] = {}

    # 1) 轻地形子地形集合确认（TerrainImporter 的生成器挂在 .cfg 下）
    gen = inner.scene.terrain.cfg.terrain_generator
    keys = set(gen.sub_terrains.keys())
    checks["light_terrain_keys"] = keys == {
        "random_rough", "hf_pyramid_slope", "hf_pyramid_slope_inv"}
    noise = gen.sub_terrains["random_rough"].noise_range
    slope = gen.sub_terrains["hf_pyramid_slope"].slope_range
    checks["noise_2_5cm"] = noise == (0.02, 0.05)
    checks["slope_le_15deg"] = slope == (0.0, 0.26)
    print(f"[smoke] sub_terrains={sorted(keys)} noise={noise} slope={slope}", flush=True)

    # 2) 盲感知确认（从 env cfg 侧验证）
    checks["blind"] = env_cfg.scene.height_scanner is None
    print(f"[smoke] height_scanner is None: {checks['blind']}", flush=True)

    # 3) 200 步滚动
    obs, info = env.reset(seed=7)
    base_z: list[float] = []
    nan_seen = False
    for step in range(200):
        action = torch.zeros((inner.num_envs, inner.action_space.shape[-1]),
                             device=device)
        obs, reward, terminated, truncated, info = env.step(action)
        if isinstance(reward, torch.Tensor):
            if not torch.isfinite(reward).all():
                nan_seen = True
        root_pos = inner.scene["robot"].data.root_pos_w
        base_z.append(float(root_pos[0, 2].item()))
    checks["roll_200_steps"] = not nan_seen
    print(f"[smoke] 200 steps OK | base_z last={base_z[-1]:.3f} m "
          f"| range=[{min(base_z):.3f},{max(base_z):.3f}]", flush=True)

    # 4) 真实仿真截图（可选：S2B_TRY_RENDER=1 才启用——enable_cameras 在
    #    Windows headless 下触发段错误，默认跳过；视觉证据改用离线高度场预览）
    frames_saved = 0
    if os.environ.get("S2B_TRY_RENDER") == "1":
        try:
            frame = env.render()
            if frame is not None:
                arr = np.asarray(frame)
                if arr.ndim == 4:  # tiled camera (num_envs, H, W, C)
                    arr = arr[0]
                import imageio

                out_dir = os.path.join(PROJECT_ROOT, "logs", "s2b_preview")
                os.makedirs(out_dir, exist_ok=True)
                imageio.imwrite(os.path.join(out_dir, "s2b_sim_view.png"),
                                arr.astype("uint8"))
                frames_saved = 1
                print(f"[smoke] sim view saved ({arr.shape})", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[smoke] render unavailable: {exc}", flush=True)
    checks["sim_screenshot"] = frames_saved == 1

    # 判定在 Kit 卸载前落盘 + 打印（关闭后 stdout 可能被吞）
    all_ok = all(checks.values())
    out_dir = os.path.join(PROJECT_ROOT, "logs", "s2b_preview")
    os.makedirs(out_dir, exist_ok=True)
    import json as _json

    verdict = {"checks": checks, "all_ok": all_ok,
               "base_z_last": base_z[-1] if base_z else None,
               "base_z_range": [min(base_z), max(base_z)] if base_z else None}
    with open(os.path.join(out_dir, "smoke_verdict.json"), "w", encoding="utf-8") as fh:
        _json.dump(verdict, fh, indent=1)
    print("\n=== S2b SMOKE VERDICT ===", flush=True)
    for k, v in checks.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}", flush=True)
    print("SMOKE", "PASS" if all_ok else "FAIL", flush=True)

    simulation_app.close()
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
