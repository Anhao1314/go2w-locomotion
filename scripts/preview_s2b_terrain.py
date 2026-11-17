#!/usr/bin/env python3
"""S2b 轻地形建模预览：按 Isaac Lab 官方高度场算法（hf_terrains.py）的
同款公式做纯 numpy 复刻，离线渲染 3D + 俯视图（不依赖 Isaac 运行时）。

复刻的两个算子：
  * random_uniform_terrain —— 均匀噪声高度场（协议：2-5 cm，步长 1 cm）
  * pyramid_sloped_terrain —— 截顶金字塔坡面（协议：坡度 ≤ 15°，平台 2 m）
正式训练/评测使用 Isaac Lab 本体生成器（同参数），此处仅用于可视化。
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LightSource  # noqa: E402
from matplotlib.tri import LinearTriInterpolator  # noqa: E402

# ---- 与 Go2S2bLightTerrainEnvCfg / 官方生成器一致的离散化参数 ----
SIZE = 8.0          # 每个子地形 8 m × 8 m
H_SCALE = 0.1       # 水平分辨率 0.1 m
V_SCALE = 0.005     # 垂直分辨率 5 mm
NOISE_RANGE = (0.02, 0.05)   # 协议：噪声 2–5 cm
NOISE_STEP = 0.01            # 1 cm
SLOPE_MAX = 0.26             # tan(15°) ≈ 0.268 → 协议 ≤15°
PLATFORM_W = 2.0             # 中央平台 2 m
GRID = (3, 3)                # 3×3 子地形拼图


def random_uniform_terrain() -> np.ndarray:
    """复刻 hf_terrains.random_uniform_terrain：噪声格均匀采样 + 竖直离散。"""
    px = int(SIZE / H_SCALE)
    h_min = int(NOISE_RANGE[0] / V_SCALE)
    h_max = int(NOISE_RANGE[1] / V_SCALE)
    h_step = int(NOISE_STEP / V_SCALE)
    rng = np.random.default_rng(42)
    heights = rng.choice(np.arange(h_min, h_max + h_step, h_step), size=(px, px))
    return heights.astype(np.int16) * V_SCALE


def pyramid_sloped_terrain(slope: float, inverted: bool) -> np.ndarray:
    """复刻 hf_terrains.pyramid_sloped_terrain：到中心平台为线性坡，slope=高/宽。"""
    px = int(SIZE / H_SCALE)
    x = np.linspace(0, SIZE, px)
    xx, yy = np.meshgrid(x, x)
    # 到平台边缘的距离（截顶金字塔：中心 platform_w 内为平的）
    half = (SIZE - PLATFORM_W) / 2.0
    dist_x = np.maximum(np.abs(xx - SIZE / 2) - PLATFORM_W / 2, 0.0)
    dist_y = np.maximum(np.abs(yy - SIZE / 2) - PLATFORM_W / 2, 0.0)
    dist = np.sqrt(dist_x**2 + dist_y**2)
    z = slope * np.minimum(dist, half)          # 坡面到边缘截止
    if inverted:
        z = slope * half - z                     # 反转：坑形
    # 官方约定：高度场以垂直刻度（5 mm）整数单位存储，网格化时才转回米
    return np.rint(z / V_SCALE).astype(np.int16) * V_SCALE


def compose() -> tuple[np.ndarray, list[str]]:
    rows, cols = GRID
    patches = [
        ("random_rough\n噪声 2–5 cm", random_uniform_terrain()),
        ("上坡 ≤15°", pyramid_sloped_terrain(SLOPE_MAX, False)),
        ("下坑 ≤15°", pyramid_sloped_terrain(SLOPE_MAX, True)),
    ]
    tiles, names = [], []
    for r in range(rows):
        row_tiles = []
        for c in range(cols):
            name, t = patches[(r * cols + c) % len(patches)]
            row_tiles.append(t)
            names.append(name)
        tiles.append(np.concatenate(row_tiles, axis=1))
    full = np.concatenate(tiles, axis=0)
    return full, names


def main() -> int:
    out = sys.argv[1] if len(sys.argv) > 1 else "logs/s2b_preview"
    field, names = compose()
    px_w = field.shape[1] * H_SCALE

    fig = plt.figure(figsize=(17, 7))
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False

    # 3D 视图
    ax1 = fig.add_subplot(121, projection="3d")
    gx = np.linspace(0, px_w, field.shape[1])
    gy = np.linspace(0, px_w, field.shape[0])
    GX, GY = np.meshgrid(gx, gy)
    ls = LightSource(azdeg=315, altdeg=55)
    rgb = ls.shade(field.reshape(field.shape[0], field.shape[1]),
                   cmap=plt.cm.terrain, vert_exag=2.0, blend_mode="overlay")
    ax1.plot_surface(GX, GY, field, facecolors=rgb, rstride=1, cstride=1,
                     linewidth=0, antialiased=False, shade=False)
    ax1.set_zlim(field.min() - 0.1, field.max() + 0.1)
    ax1.set_title("S2b 轻地形 3D（噪声 2–5 cm · 坡度 ≤15° · 盲感知）", fontsize=11)
    ax1.set_xlabel("x (m)"), ax1.set_ylabel("y (m)"), ax1.set_zlabel("高度 (m)")
    ax1.view_init(elev=45, azim=-55)

    # 俯视高度图
    ax2 = fig.add_subplot(122)
    img = ax2.imshow(field, origin="lower",
                     extent=(0, px_w, 0, px_w), cmap="terrain",
                     vmin=field.min(), vmax=field.max(), interpolation="bilinear")
    # 网格线标出子地形边界
    for k in range(1, GRID[0]):
        ax2.axhline(k * SIZE, color="k", lw=1.2)
        ax2.axvline(k * SIZE, color="k", lw=1.2)
    fig.colorbar(img, ax=ax2, shrink=0.8, label="高度 (m)")
    ax2.set_title("俯视高度场（3×3 子地形，每格 8 m）", fontsize=11)
    ax2.set_xlabel("x (m)"), ax2.set_ylabel("y (m)")

    import pathlib

    outp = pathlib.Path(out)
    outp.mkdir(parents=True, exist_ok=True)
    fig.savefig(outp / "s2b_terrain_preview.png", dpi=140, bbox_inches="tight")
    print(f"[preview] height range: {field.min():.3f} ~ {field.max():.3f} m")
    print(f"[preview] saved {outp / 's2b_terrain_preview.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
