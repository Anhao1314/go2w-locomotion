#!/usr/bin/env python3
"""从录好的 LIBERO 评测输出拼装 VLA 证据视频。

输入是 ``vla/evaluate_libero.py --video`` 的输出目录：每个 episode 两路相机
（front / wrist）的 mp4，以及 ``libero_eval_summary.json``（含 seed、结局、步数、
逐步的夹爪指令与末端高度）。

产出：
  * ``media/training/libero_quad.mp4``  2×2 网格，每格前视相机 + 标签（seed/结局/步数）
  * ``media/training/libero_dual.mp4``  前视 + 腕部双视角，底部同步夹爪/高度曲线

用法（Windows .venv312，需 imageio-ffmpeg）：
    python scripts/make_vla_video_pack.py \
        --run-dir logs/vla/libero_video_pack2 --quad 0,1,2,3 --dual 0
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

import imageio  # pyright: ignore[reportMissingImports]
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = pathlib.Path(__file__).resolve().parents[1]

# 尺寸取 16 的倍数，避免 imageio/ffmpeg 隐式缩放（688x688 / 688x592）
CELL = 320
LABEL_H = 32
GAP = 16
PANEL_H = 192

FONT_CANDIDATES = (
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
)


def load_font(size: int) -> Any:
    for path in FONT_CANDIDATES:
        if pathlib.Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def load_frames(path: pathlib.Path) -> list[np.ndarray]:
    reader = imageio.get_reader(str(path))
    return [np.asarray(frame) for frame in reader]


def label_for(result: dict[str, Any]) -> str:
    outcome = "SUCCESS" if result["success"] else "TIMEOUT"
    return f"seed {result['seed']} · {outcome} · {result['steps']} 步"


def make_cell(frame: np.ndarray, text: str, color: tuple[int, int, int],
              font: Any) -> Image.Image:
    img = Image.fromarray(frame).resize((CELL, CELL), Image.Resampling.BILINEAR)
    cell = Image.new("RGB", (CELL, CELL + LABEL_H), (255, 255, 255))
    cell.paste(img, (0, LABEL_H))
    draw = ImageDraw.Draw(cell)
    draw.rectangle([0, 0, CELL, LABEL_H], fill=(245, 245, 245))
    draw.text((8, 6), text, fill=color, font=font)
    return cell


def build_quad(run_dir: pathlib.Path, results: list[dict[str, Any]],
               picks: list[int], out: pathlib.Path) -> pathlib.Path:
    font = load_font(17)
    cells: list[list[str]] = []
    for idx in picks[:4]:
        res = results[idx]
        cells.append([str(run_dir / res["video"]["front"])])
    frame_sets = [load_frames(pathlib.Path(cells[i][0])) for i in range(len(cells))]
    n = max(len(f) for f in frame_sets)

    width = CELL * 2 + GAP * 3
    height = (CELL + LABEL_H) * 2 + GAP * 3
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out), fps=20, codec="libx264", quality=8)
    try:
        for t in range(n):
            canvas = Image.new("RGB", (width, height), (255, 255, 255))
            for cell_i, idx in enumerate(picks[:4]):
                frames = frame_sets[cell_i]
                res = results[idx]
                color = (46, 139, 87) if res["success"] else (192, 57, 43)
                cell = make_cell(frames[min(t, len(frames) - 1)],
                                 label_for(res), color, font)
                row, col = divmod(cell_i, 2)
                x = GAP + col * (CELL + GAP)
                y = GAP + row * (CELL + LABEL_H + GAP)
                canvas.paste(cell, (x, y))
            writer.append_data(np.asarray(canvas))
    finally:
        writer.close()
    return out


def _panel_curve(values: list[float], box: tuple[int, int, int, int],
                 lo: float, hi: float, color: tuple[int, int, int],
                 draw: ImageDraw.ImageDraw, upto: int) -> None:
    x0, y0, x1, y1 = box
    n = max(1, len(values) - 1)
    span = max(1e-6, hi - lo)
    pts = []
    for i, v in enumerate(values[: max(2, upto + 1)]):
        px = x0 + (x1 - x0) * i / n
        py = y1 - (y1 - y0) * (v - lo) / span
        pts.append((px, py))
    if len(pts) >= 2:
        draw.line(pts, fill=color, width=2)


def build_dual(run_dir: pathlib.Path, result: dict[str, Any], out: pathlib.Path) -> pathlib.Path:
    font = load_font(16)
    small = load_font(12)
    front = load_frames(run_dir / result["video"]["front"])
    wrist = load_frames(run_dir / result["video"]["wrist"])
    n = min(len(front), len(wrist))
    trace = result.get("trace") or {}
    grip = [float(v) for v in (trace.get("gripper_action") or [0.0])]
    height = [float(v) for v in (trace.get("state_z") or [0.0])]

    width = CELL * 2 + GAP * 3
    height_px = CELL + LABEL_H + PANEL_H + GAP * 3
    writer = imageio.get_writer(str(out), fps=20, codec="libx264", quality=8)
    try:
        for t in range(n):
            canvas = Image.new("RGB", (width, height_px), (255, 255, 255))
            draw = ImageDraw.Draw(canvas)
            left = make_cell(front[t], "前视相机", (46, 94, 170), font)
            right = make_cell(wrist[t], "腕部相机", (46, 94, 170), font)
            canvas.paste(left, (GAP, GAP))
            canvas.paste(right, (GAP * 2 + CELL, GAP))

            panel_top = GAP * 2 + CELL + LABEL_H
            box_g = (GAP + 34, panel_top + 18, width - GAP - 10, panel_top + PANEL_H - 46)
            draw.rectangle([box_g[0], box_g[1], box_g[2], box_g[3]], outline=(210, 210, 210))
            draw.text((GAP, panel_top), "夹爪指令（+1 闭合 / −1 张开）", fill=(70, 70, 70), font=font)
            mid = (box_g[1] + box_g[3]) // 2
            draw.line([(box_g[0], mid), (box_g[2], mid)], fill=(228, 228, 228), width=1)
            for label, y in (("+1", box_g[1] + 2), ("0", mid - 8), ("-1", box_g[3] - 16)):
                draw.text((box_g[0] - 26, y), label, fill=(150, 150, 150), font=small)
            _panel_curve(grip, box_g, -1.05, 1.05, (217, 130, 43), draw, t)
            x_now = box_g[0] + (box_g[2] - box_g[0]) * t / max(1, n - 1)
            draw.line([(x_now, box_g[1]), (x_now, box_g[3])], fill=(150, 150, 150), width=1)
            g_now = grip[min(t, len(grip) - 1)]
            y_now = box_g[3] - (box_g[3] - box_g[1]) * (g_now + 1.05) / 2.1
            draw.ellipse([x_now - 4, y_now - 4, x_now + 4, y_now + 4],
                         fill=(192, 57, 43))
            draw.text((box_g[0], box_g[3] + 6),
                      f"{label_for(result)} · t={t / 20:.2f}s · 高度 {height[min(t, len(height)-1)]:.3f}m",
                      fill=(70, 70, 70), font=font)
            writer.append_data(np.asarray(canvas))
    finally:
        writer.close()
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--run-dir", type=pathlib.Path, required=True)
    parser.add_argument("--out-dir", type=pathlib.Path, default=ROOT / "media" / "training")
    parser.add_argument("--quad", type=str, default="0,1,2,3")
    parser.add_argument("--dual", type=int, default=0)
    args = parser.parse_args(argv)

    run_dir = pathlib.Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    summary = json.loads((run_dir / "libero_eval_summary.json").read_text(encoding="utf-8"))
    results = summary["results"]
    picks = [int(x) for x in args.quad.split(",")]

    quad = build_quad(run_dir, results, picks, args.out_dir / "libero_quad.mp4")
    print(f"[pack] {quad.relative_to(ROOT)}  {quad.stat().st_size // 1024} KB")
    if 0 <= args.dual < len(results):
        dual = build_dual(run_dir, results[args.dual], args.out_dir / "libero_dual.mp4")
        print(f"[pack] {dual.relative_to(ROOT)}  {dual.stat().st_size // 1024} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
