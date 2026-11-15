#!/usr/bin/env python3
"""技能提示是否"有意义"：匹配提示能否降低该阶段的动作误差？

openloop_prompt_check 证明加前缀会改变输出（ΔL2≈0.41），但那可能只是扰动。
这里做判定性检验：按示范的夹爪状态切出"接近段 / 搬运段"，两段各取若干帧，
分别用三种提示预测，比较**该阶段的动作 MAE**：

    若 approach 帧上 "approach | 任务" 的 MAE < 原始任务的 MAE，
    且 transport 帧上 "transport | 任务" 更优 → 提示携带真实技能信息，
    分层可以靠条件化实现，不必重训；
    否则必须训练技能条件化策略。

用法（WSL vla-env，离线）：
    HF_HUB_OFFLINE=1 python vla/phase_prompt_eval.py \
        --checkpoint logs/vla/libero_smolvla/checkpoints/060000/pretrained_model \
        --dataset-root /root/.cache/.../snapshots/<rev> --episodes 6
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def episode_actions(root: pathlib.Path) -> dict[int, list[list[float]]]:
    import pyarrow.parquet as pq  # pyright: ignore[reportMissingImports]

    out: dict[int, list[list[float]]] = {}
    for path in sorted((root / "data").rglob("*.parquet")):
        table = pq.read_table(path, columns=["episode_index", "action"])
        for ep, act in zip(table.column("episode_index").to_numpy(),
                           table.column("action").to_pylist()):
            out.setdefault(int(ep), []).append([float(v) for v in act])
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--dataset-repo", type=str, default="lerobot/libero")
    parser.add_argument("--dataset-root", type=pathlib.Path, required=True)
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--frames-per-phase", type=int, default=2)
    parser.add_argument("--out", type=pathlib.Path,
                        default=PROJECT_ROOT / "logs" / "vla" / "phase_prompt_eval")
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
    actions = episode_actions(args.dataset_root)
    episodes_meta = ds.meta.episodes
    starts = list(episodes_meta["dataset_from_index"])

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

    # 选符合模板的 episode，按夹爪跳变切段
    picked: list[tuple[int, int, int]] = []  # (episode_index, close_at, open_at)
    for ep_id in sorted(actions):
        if len(picked) >= args.episodes:
            break
        arr = np.asarray(actions[ep_id])
        grip = np.sign(arr[:, 6])
        grip[grip == 0] = 1.0
        changes = np.flatnonzero(np.diff(grip) != 0) + 1
        if len(changes) >= 2 and grip[0] < 0:
            picked.append((ep_id, int(changes[0]), int(changes[1])))

    prompts = ("base", "approach", "transport")
    errors: dict[str, dict[str, list[float]]] = {
        phase: {p: [] for p in prompts} for phase in ("approach", "transport")}

    for ep_id, close_at, open_at in picked:
        arr = np.asarray(actions[ep_id])
        base_task = ""
        spans = {
            "approach": np.linspace(0, max(1, close_at - 1), args.frames_per_phase + 2)[1:-1],
            "transport": np.linspace(close_at + 1, max(close_at + 2, open_at - 1),
                                     args.frames_per_phase + 2)[1:-1],
        }
        for phase, frames in spans.items():
            for frame in frames.astype(int):
                index = int(starts[ep_id]) + int(frame)
                item = ds[index]
                if not base_task:
                    task = item.get("task")
                    base_task = task if isinstance(task, str) else ""
                gt = np.asarray(arr[frame], dtype=np.float64)
                for p in prompts:
                    prompt = base_task if p == "base" else f"{p} | {base_task}"
                    pred = predict(item, prompt)
                    errors[phase][p].append(float(np.mean(np.abs(pred - gt))))

    table = {
        phase: {p: statistics.fmean(v) if v else None for p, v in per.items()}
        for phase, per in errors.items()
    }
    def matched_wins(phase: str) -> bool | None:
        base = table[phase]["base"]
        matched = table[phase][phase]
        if base is None or matched is None:
            return None
        return matched < base

    result = {
        "episodes": len(picked),
        "mae_by_phase_and_prompt": table,
        "matched_prompt_wins": {phase: matched_wins(phase)
                                for phase in ("approach", "transport")},
        "samples_per_cell": {phase: len(per["base"]) for phase, per in errors.items()},
    }

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "phase_prompt_eval.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
