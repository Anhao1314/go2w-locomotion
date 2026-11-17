#!/usr/bin/env python3
"""实时仿真观察流：用当前策略在 LIBERO 里连续做题，MJPEG 推流。

- 端口 8766：/stream.mjpg（浏览器可直接 <img>）；/status（JSON）
- 无限循环做题：task 0（黑碗→盘子），种子 5000 起轮转（不与评测种子冲突）
- HUD 叠字：episode / step / 结果；每 3 个物理步刷新一次画面

用法（WSL vla-env）：MUJOCO_GL=egl python scripts/vla_live_viewer.py \
    --checkpoint logs/vla/libero_smolvla/checkpoints/060000/pretrained_model
"""

from __future__ import annotations

import argparse
import io
import json
import pathlib
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

STATE: dict[str, Any] = {
    "jpeg": None,
    "lock": threading.Lock(),
    "episode": 0,
    "step": 0,
    "max_steps": 0,
    "outcome": "",
    "clients": 0,
    "policy": "",
}


def hud(frame, lines: list[str]):
    from PIL import Image, ImageDraw

    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    y = 6
    for ln in lines:
        draw.rectangle([4, y - 2, 4 + 7 * len(ln) + 8, y + 14], fill=(0, 0, 0))
        draw.text((8, y), ln, fill=(255, 255, 255))
        y += 18
    return __import__("numpy").asarray(img)


OUT_DIR = pathlib.Path("/mnt/d/work/go2w-locomotion/logs/vla/live_episodes")


def run_loop(args) -> None:
    import os

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    import imageio
    import numpy as np
    import torch
    from lerobot.configs.policies import PreTrainedConfig  # pyright: ignore[reportMissingImports]
    from lerobot.policies import get_policy_class, make_pre_post_processors  # pyright: ignore[reportMissingImports]

    sys.path.insert(0, str(PROJECT_ROOT))
    from vla.env import make_libero_env  # noqa: E402
    from vla.evaluate_libero import _as_image, _as_state  # noqa: E402

    ckpt = pathlib.Path(args.checkpoint)
    cfg = PreTrainedConfig.from_pretrained(str(ckpt))
    cfg.pretrained_path = str(ckpt)
    device = cfg.device or "cuda"
    pre, post = make_pre_post_processors(
        policy_cfg=cfg, pretrained_path=str(ckpt),
        preprocessor_overrides={"device_processor": {"device": device}})
    policy = get_policy_class(cfg.type).from_pretrained(str(ckpt), config=cfg)
    policy.to(device).eval()
    STATE["policy"] = str(ckpt)

    env = make_libero_env(args.suite, task_ids=[args.task], fps=20)

    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    ep = 0
    while True:
        seed = args.seed_base + ep
        torch.manual_seed(seed)
        obs, info = env.reset(seed=[seed])
        policy.reset()
        task = info.get("task")
        task = task[0] if isinstance(task, (tuple, list)) else task
        STATE["episode"] = ep + 1
        STATE["outcome"] = "running"
        max_steps = 280
        STATE["max_steps"] = max_steps
        success = False
        frames = []
        for step in range(max_steps):
            batch = {
                "observation.images.image": _as_image(obs["observation.images.image"]),
                "observation.images.image2": _as_image(obs["observation.images.image2"]),
                "observation.state": _as_state(obs["observation.state"]),
                "task": task,
            }
            processed = pre(batch)
            with torch.inference_mode():
                action = policy.select_action(processed)
            if action.ndim == 1:
                action = action.unsqueeze(0)
            action = post(action)
            action_np = (action.detach().cpu().numpy()
                         if hasattr(action, "detach") else np.asarray(action))
            action_np = np.asarray(action_np, dtype=np.float32)
            if action_np.ndim == 1:
                action_np = action_np[None, :]
            obs, _reward, terminated, truncated, info = env.step(action_np)
            STATE["step"] = step + 1
            if isinstance(info, dict) and np.asarray(info.get("is_success", False)).any():
                success = True
            raw = np.asarray(obs["observation.images.image"])
            frame = raw[0] if raw.ndim == 4 else raw
            result = "SUCCESS" if success else ("RUNNING" if step < max_steps - 1 else "TIMEOUT")
            lines = [
                f"EP {ep + 1}  seed {seed}  STEP {step + 1}/{max_steps}",
                f"RESULT: {result}",
                (task[:52] + "...") if task and len(task) > 52 else (task or ""),
            ]
            frames.append(hud(np.asarray(frame), lines))
            # 当前帧快照（面板缩略图）
            jpeg_buf = io.BytesIO()
            imageio.imwrite(jpeg_buf, frames[-1], format="jpeg", quality=80)
            with STATE["lock"]:
                STATE["jpeg"] = jpeg_buf.getvalue()
            if step % 20 == 0:
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "latest.jpg").write_bytes(jpeg_buf.getvalue())
            if success or terminated or truncated:
                break
        STATE["outcome"] = "SUCCESS" if success else "TIMEOUT"
        # 逐集录制 MP4 → latest_episode.mp4（面板定时加载）
        try:
            rec_path = out_dir / f"ep{ep + 1:03d}_{('SUCCESS' if success else 'TIMEOUT')}.mp4"
            with imageio.get_writer(rec_path, fps=20, codec="libx264", quality=7) as w:
                for fr in frames:
                    w.append_data(fr)
            latest = out_dir / "latest_episode.mp4"
            latest.write_bytes(rec_path.read_bytes())
            (out_dir / "latest_status.json").write_text(json.dumps({
                "episode": ep + 1, "seed": seed, "outcome": STATE["outcome"],
                "steps": len(frames), "file": rec_path.name,
                "updated": time.strftime("%H:%M:%S")}), encoding="utf-8")
            # 只保留最近 5 个录像，防磁盘膨胀
            old = sorted(out_dir.glob("ep*.mp4"))
            for o in old[:-5]:
                o.unlink(missing_ok=True)
        except Exception as exc:
            print(f"[viewer] record failed: {exc}", flush=True)
        time.sleep(1.0)
        ep += 1


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/status"):
            body = json.dumps({k: v for k, v in STATE.items()
                               if k not in ("jpeg", "lock")}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/stream"):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            STATE["clients"] += 1
            try:
                while True:
                    with STATE["lock"]:
                        jpeg = STATE["jpeg"]
                    if jpeg is None:
                        time.sleep(0.2)
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                    time.sleep(0.1)
            except Exception:
                pass
            finally:
                STATE["clients"] -= 1
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *a):
        pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="logs/vla/libero_smolvla/checkpoints/060000/pretrained_model")
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--seed-base", type=int, default=5000)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()

    threading.Thread(target=run_loop, args=(args,), daemon=True).start()
    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"live viewer: http://localhost:{args.port}/stream  (policy: {args.checkpoint})")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
