#!/usr/bin/env python3
"""Open-loop reproduction check: does the fine-tuned policy reproduce the
demonstrations it was trained on?

This is the cheap discriminator between "the model never learned to act" and
"the model is fine but our closed-loop harness / interface is wrong":

  * feed the policy a frame it saw during training (same images, same state,
    same task string, same preprocessing pipeline as training)
  * compare its predicted action with the recorded ground-truth action
  * compare against two baselines: the dataset-mean action, and a
    vision-blind version of the same policy (images replaced by zeros)

If the predicted action tracks the demonstration (MSE well below the
mean-action baseline, gripper channel decoded correctly), the policy learned
something and failures live in the evaluation harness.  If it does not, the
problem is in the model/data/normalisation side.

Usage (WSL vla-env, offline):
    HF_HUB_OFFLINE=1 python vla/openloop_check.py \
        --checkpoint logs/vla/libero_smolvla/checkpoints/060000/pretrained_model \
        --dataset-root /root/.cache/huggingface/lerobot/hub/datasets--lerobot--libero/snapshots/<rev>
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _task_string(item: dict[str, object], dataset: object) -> str:
    """Resolve the language instruction for a dataset frame."""
    task = item.get("task")
    if isinstance(task, str):
        return task
    task_index = item.get("task_index")
    if task_index is not None:
        tasks = getattr(getattr(dataset, "meta", None), "tasks", None)
        if tasks is not None:
            try:
                row = tasks.loc[int(task_index)]  # type: ignore[union-attr]
                if hasattr(row, "__getitem__"):
                    return str(row["task"])
                return str(row)
            except Exception:  # noqa: BLE001 - best effort metadata lookup
                pass
    return ""


def _episode_bounds(dataset: object) -> list[tuple[int, int]]:
    """[(from_index, to_index)] per episode, v3 metadata with a fallback."""
    meta = getattr(dataset, "meta", None)
    episodes = getattr(meta, "episodes", None)
    if episodes is not None and hasattr(episodes, "columns"):
        cols = list(episodes.columns)
        if "dataset_from_index" in cols and "dataset_to_index" in cols:
            return [
                (int(a), int(b))
                for a, b in zip(
                    episodes["dataset_from_index"], episodes["dataset_to_index"]
                )
            ]
    total = len(dataset)  # type: ignore[arg-type]
    n = int(getattr(meta, "total_episodes", 1) or 1)
    step = max(1, total // max(1, n))
    return [(i * step, min(total, (i + 1) * step)) for i in range(n)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--dataset-repo", type=str, default="lerobot/libero")
    parser.add_argument("--dataset-root", type=pathlib.Path, required=True)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--frames-per-episode", type=int, default=5)
    parser.add_argument("--black-image-frames", type=int, default=8,
                        help="frames used for the vision-ablation probe")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out", type=pathlib.Path,
                        default=PROJECT_ROOT / "logs" / "vla" / "openloop_check")
    args = parser.parse_args(argv)

    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    import numpy as np
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies import get_policy_class, make_pre_post_processors

    ckpt = pathlib.Path(args.checkpoint)
    policy_config = PreTrainedConfig.from_pretrained(str(ckpt))
    policy_config.pretrained_path = str(ckpt)
    device = args.device or policy_config.device or "cuda"

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

    dataset = LeRobotDataset(args.dataset_repo, root=args.dataset_root)
    bounds = _episode_bounds(dataset)
    print(f"[data] frames={len(dataset)} episodes={len(bounds)} "
          f"fps={getattr(dataset.meta, 'fps', '?')}", flush=True)

    fractions = np.linspace(0.1, 0.9, max(1, args.frames_per_episode))
    picks: list[int] = []
    if bounds:
        chosen = np.linspace(0, len(bounds) - 1, min(args.episodes, len(bounds)))
        for ep in sorted({int(round(c)) for c in chosen}):
            start, end = bounds[ep]
            span = max(1, end - start - 1)
            for frac in fractions:
                picks.append(min(end - 1, start + int(round(frac * span))))

    def predict(item: dict[str, object], blank_images: bool) -> np.ndarray:
        image = torch.as_tensor(item["observation.images.image"])
        image2 = torch.as_tensor(item["observation.images.image2"])
        if blank_images:
            image = torch.zeros_like(image)
            image2 = torch.zeros_like(image2)
        batch = {
            "observation.images.image": image,
            "observation.images.image2": image2,
            "observation.state": torch.as_tensor(
                np.asarray(item["observation.state"], dtype=np.float32)
            ),
            "task": _task_string(item, dataset),
        }
        processed = preprocessor(batch)
        policy.reset()
        with torch.inference_mode():
            action = policy.select_action(processed)
        action = postprocessor(action)
        arr = action.detach().cpu().numpy() if hasattr(action, "detach") else np.asarray(action)
        return np.asarray(arr, dtype=np.float64).reshape(-1)

    records: list[dict[str, object]] = []
    vision_deltas: list[float] = []
    for n, index in enumerate(picks):
        item = dataset[index]
        gt = np.asarray(item["action"], dtype=np.float64).reshape(-1)
        pred = predict(item, blank_images=False)
        rec: dict[str, object] = {
            "frame_index": int(index),
            "gt": gt.tolist(),
            "pred": pred.tolist(),
            "abs_err": np.abs(pred - gt).tolist(),
        }
        if n < args.black_image_frames:
            blind = predict(item, blank_images=True)
            vision_deltas.append(float(np.linalg.norm(pred - blind)))
            rec["pred_blind"] = blind.tolist()
        records.append(rec)

    gt_all = np.array([r["gt"] for r in records], dtype=np.float64)
    pred_all = np.array([r["pred"] for r in records], dtype=np.float64)
    mean_action = gt_all.mean(axis=0)

    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        if a.size < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    mse = float(np.mean((pred_all - gt_all) ** 2))
    mae_per_dim = np.mean(np.abs(pred_all - gt_all), axis=0)
    baseline_mse = float(np.mean((mean_action[None, :] - gt_all) ** 2))
    corr_per_dim = [_corr(gt_all[:, i], pred_all[:, i]) for i in range(gt_all.shape[1])]
    gripper_dim = int(np.argmax(np.std(gt_all, axis=0)[-3:]) + gt_all.shape[1] - 3)
    gripper_sign_acc = float(
        np.mean(np.sign(pred_all[:, gripper_dim]) == np.sign(gt_all[:, gripper_dim]))
    )

    summary = {
        "checkpoint": str(ckpt),
        "frames": len(records),
        "action_dim": int(gt_all.shape[1]),
        "mse": mse,
        "mean_baseline_mse": baseline_mse,
        "mse_ratio_to_baseline": mse / baseline_mse if baseline_mse > 0 else None,
        "mae_per_dim": mae_per_dim.tolist(),
        "corr_per_dim": corr_per_dim,
        "gripper_dim": gripper_dim,
        "gripper_sign_accuracy": gripper_sign_acc,
        "pred_std_per_dim": pred_all.std(axis=0).tolist(),
        "gt_std_per_dim": gt_all.std(axis=0).tolist(),
        "vision_delta_mean_l2": float(np.mean(vision_deltas)) if vision_deltas else None,
        "records": records,
    }

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "openloop_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    print(f"[result] frames={len(records)} action_dim={gt_all.shape[1]}")
    print(f"[result] MSE={mse:.4f}  baseline(mean-action) MSE={baseline_mse:.4f}  "
          f"ratio={mse / baseline_mse:.3f}" if baseline_mse > 0 else "[result] no baseline")
    print("[result] per-dim MAE:", " ".join(f"{v:.3f}" for v in mae_per_dim))
    print("[result] per-dim corr:", " ".join(f"{v:+.2f}" for v in corr_per_dim))
    print(f"[result] gripper dim={gripper_dim} sign-acc={gripper_sign_acc:.2f}")
    print(f"[result] pred std:", " ".join(f"{v:.3f}" for v in pred_all.std(axis=0)))
    print(f"[result] gt   std:", " ".join(f"{v:.3f}" for v in gt_all.std(axis=0)))
    if vision_deltas:
        print(f"[result] vision dependence: mean ||pred(real)-pred(black)|| = "
              f"{np.mean(vision_deltas):.4f} over {len(vision_deltas)} frames")
    print(f"[result] wrote {out / 'openloop_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
