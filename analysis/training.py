"""训练日志 → 训练类因子。

LeRobot 的训练日志是进度条 + INFO 行的混合体（`\\r` 分隔），每行形如::

    step:62K smpl:2M ep:12K epch:7.23 loss:0.380 grdn:1.066 lr:2.5e-06
    data_s:0.002 prep_s:0.004 updt_s:0.683 step_s:0.689 smp/s:46 mem_gb:6.80

这里只做一件事：把这种行解析成结构化记录，并折算成 `training` 组的因子。
"""

from __future__ import annotations

import math
import pathlib
import re

LOG_RE = re.compile(
    r"step:(?P<step>\d+(?:\.\d+)?)(?P<step_u>[KM]?).*?"
    r"loss:(?P<loss>[0-9.]+).*?"
    r"grdn:(?P<grdn>[0-9.]+).*?"
    r"lr:(?P<lr>[0-9.eE+-]+).*?"
    r"step_s:(?P<step_s>[0-9.]+).*?"
    r"smp/s:(?P<smp_s>[0-9.]+).*?"
    r"mem_gb:(?P<mem>[0-9.]+)"
)


def parse_training_log(path: pathlib.Path) -> list[dict[str, float]]:
    """解析训练日志，返回按 step 去重排序的记录列表。"""
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8", errors="ignore").replace("\r", "\n")
    by_step: dict[float, dict[str, float]] = {}
    for line in raw.split("\n"):
        m = LOG_RE.search(line)
        if not m:
            continue
        mult = 1000.0 if m.group("step_u") == "K" else 1e6 if m.group("step_u") == "M" else 1.0
        step = float(m.group("step")) * mult
        by_step[step] = {
            "step": step,
            "loss": float(m.group("loss")),
            "grdn": float(m.group("grdn")),
            "lr": float(m.group("lr")),
            "step_s": float(m.group("step_s")),
            "smp_s": float(m.group("smp_s")),
            "mem_gb": float(m.group("mem")),
        }
    return [by_step[k] for k in sorted(by_step)]


def training_factors(path: pathlib.Path, split_step: float | None = None) -> dict[str, float | None]:
    """折算训练类因子。

    ``split_step`` 把曲线切成前后两段以计算 ``train_plateau_ratio``；缺省取中点。
    """
    recs = parse_training_log(path)
    if not recs:
        return {"train_smp_per_s": None, "train_step_s": None, "train_mem_gb": None,
                "train_loss_decay_per_10k": None, "train_plateau_ratio": None,
                "train_grad_norm": None, "train_lr_end": None}
    first, last = recs[0], recs[-1]
    span = max(1.0, last["step"] - first["step"])
    decay = (first["loss"] - last["loss"]) / span * 10000.0

    cut = split_step if split_step is not None else (first["step"] + last["step"]) / 2.0
    early = [r for r in recs if r["step"] <= cut]
    late = [r for r in recs if r["step"] > cut]

    def segment_decay(seg: list[dict[str, float]]) -> float | None:
        if len(seg) < 2:
            return None
        seg_span = max(1.0, seg[-1]["step"] - seg[0]["step"])
        return (seg[0]["loss"] - seg[-1]["loss"]) / seg_span * 10000.0

    early_decay = segment_decay(early)
    late_decay = segment_decay(late)
    plateau = (abs(late_decay) / abs(early_decay)) if (early_decay and late_decay is not None
                                                       and abs(early_decay) > 1e-9) else None

    def mean(key: str) -> float:
        vals = [r[key] for r in recs]
        return sum(vals) / len(vals)

    return {
        "train_smp_per_s": mean("smp_s"),
        "train_step_s": mean("step_s"),
        "train_mem_gb": mean("mem_gb"),
        "train_loss_decay_per_10k": decay,
        "train_plateau_ratio": plateau,
        "train_grad_norm": mean("grdn"),
        "train_lr_end": last["lr"],
    }
