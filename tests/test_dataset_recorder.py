"""Unit tests for the go2-sequence-v1 dataset writer (no Isaac required)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from locomotion.eval.dataset import (
    CONTACT_THRESHOLD_N,
    SCHEMA_VERSION,
    STEP_FEATURES,
    EpisodeRecorder,
)


def _fake_step(recorder: EpisodeRecorder, step: int, envs: int = 2) -> None:
    recorder.add(obs=np.full((envs, 53), step, dtype=np.float32))
    recorder.add(action=np.full((envs, 2), 0.1 * step, dtype=np.float32))
    recorder.add(
        cmd=np.zeros((envs, 3), dtype=np.float32),
        joint_target=np.zeros((envs, 12), dtype=np.float32),
        joint_torque=np.full((envs, 12), 2.5, dtype=np.float32),
        base_pos=np.zeros((envs, 3), dtype=np.float32),
        base_quat=np.zeros((envs, 4), dtype=np.float32),
        base_lin_vel_b=np.zeros((envs, 3), dtype=np.float32),
        base_ang_vel_b=np.zeros((envs, 3), dtype=np.float32),
        joint_pos=np.zeros((envs, 12), dtype=np.float32),
        joint_vel=np.zeros((envs, 12), dtype=np.float32),
        foot_contact_force=np.zeros((envs, 4), dtype=np.float32),
        stage=np.zeros((envs,), dtype=np.int8),
        reward=np.full((envs,), 0.5, dtype=np.float32),
        body_pos=np.zeros((envs, 13, 3), dtype=np.float32),
        done=np.zeros((envs,), dtype=bool),
    )


def _recorder(root: Path, group: str = "l3_layered_seed0") -> EpisodeRecorder:
    return EpisodeRecorder(
        root,
        group,
        run_meta={
            "task": "Go2Loco-Sequence-v0",
            "checkpoint": "ckpt.pt",
            "layered": "single" not in group,
            "interface": {"high_output": "velocity_2d", "low_output": "joint_target"},
            "config": {
                "turn_radians": 1.5707963267948966,
                "forward_distance_m": 2.0,
                "patched": False,
            },
            "joint_names": [f"j{i}" for i in range(12)],
            "body_names": [f"b{i}" for i in range(13)],
            "foot_names": [f"f{i}" for i in range(4)],
            "obs_layout": [],
            "git_commit": "test",
        },
    )


class EpisodeRecorderTest(unittest.TestCase):
    def test_roundtrip_outcomes_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recorder = _recorder(root)
            for step in range(5):
                _fake_step(recorder, step)
            path_fall = recorder.finalize(
                0, eval_seed=0, done_step=3, outcome="fall", final_stage=1
            )
            path_timeout = recorder.finalize(
                1, eval_seed=1, done_step=5, outcome="timeout", final_stage=2
            )
            self.assertIsNotNone(path_fall)
            self.assertIsNotNone(path_timeout)

            with np.load(path_fall, allow_pickle=False) as data:
                self.assertEqual(data["obs"].shape, (3, 53))
                self.assertEqual(data["body_pos"].shape, (3, 13, 3))
                self.assertEqual(data["terminated"].tolist(), [False, False, True])
                self.assertEqual(data["truncated"].tolist(), [True, True, False])
                self.assertEqual(
                    data["foot_contact"].tolist(),
                    [[False] * 4] * 3,
                )
                entry = json.loads(str(data["meta"]))
                self.assertEqual(entry["fail_step"], 2)
                self.assertFalse(entry["success"])

            with np.load(path_timeout, allow_pickle=False) as data:
                self.assertEqual(data["terminated"].tolist(), [False] * 5)
                self.assertEqual(data["truncated"].tolist(), [True] * 5)

            manifest = recorder.flush()
            self.assertEqual(manifest["schema_version"], SCHEMA_VERSION)
            self.assertEqual(manifest["total_episodes"], 2)
            self.assertEqual(manifest["total_steps"], 8)
            self.assertEqual(manifest["splits"]["failure"], manifest["splits"]["all"])
            self.assertIn("obs", manifest["features"])
            self.assertIn("foot_contact", manifest["features"])
            self.assertIn("obs", manifest["stats"])
            checksums = (root / "checksums.sha256").read_text(encoding="utf-8").strip()
            self.assertEqual(len(checksums.splitlines()), 2)
            card = (root / "README.md").read_text(encoding="utf-8")
            self.assertIn("outcome-distribution", card)

    def test_contact_threshold_derivation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recorder = _recorder(root, group="l1_single_seed0")
            recorder.add(obs=np.zeros((1, 53), dtype=np.float32))
            recorder.add(action=np.zeros((1, 12), dtype=np.float32))
            recorder.add(
                cmd=np.zeros((1, 3), dtype=np.float32),
                joint_target=np.zeros((1, 12), dtype=np.float32),
                joint_torque=np.zeros((1, 12), dtype=np.float32),
                base_pos=np.zeros((1, 3), dtype=np.float32),
                base_quat=np.zeros((1, 4), dtype=np.float32),
                base_lin_vel_b=np.zeros((1, 3), dtype=np.float32),
                base_ang_vel_b=np.zeros((1, 3), dtype=np.float32),
                joint_pos=np.zeros((1, 12), dtype=np.float32),
                joint_vel=np.zeros((1, 12), dtype=np.float32),
                foot_contact_force=np.full((1, 4), CONTACT_THRESHOLD_N + 0.5, dtype=np.float32),
                stage=np.zeros((1,), dtype=np.int8),
                reward=np.zeros((1,), dtype=np.float32),
                body_pos=np.zeros((1, 13, 3), dtype=np.float32),
                done=np.zeros((1,), dtype=bool),
            )
            path = recorder.finalize(0, eval_seed=3, done_step=1, outcome="timeout", final_stage=0)
            assert path is not None
            with np.load(path, allow_pickle=False) as data:
                self.assertEqual(data["foot_contact"].tolist(), [[True] * 4])

    def test_manifest_merges_across_invocations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = _recorder(root)
            for step in range(2):
                _fake_step(first, step)
            first.finalize(0, eval_seed=0, done_step=2, outcome="success", final_stage=4)
            first.flush()

            second = _recorder(root, group="l3_single_seed2")
            for step in range(4):
                _fake_step(second, step)
            second.finalize(0, eval_seed=0, done_step=4, outcome="fall", final_stage=1)
            manifest = second.flush()

            self.assertEqual(manifest["total_episodes"], 2)
            self.assertEqual(len(manifest["splits"]["layered"]), 1)
            self.assertEqual(len(manifest["splits"]["single"]), 1)
            self.assertEqual(manifest["total_steps"], 6)

    def test_lockstep_violation_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = _recorder(Path(tmp))
            recorder.add(obs=np.zeros((2, 53), dtype=np.float32))
            recorder.add(action=np.zeros((2, 2), dtype=np.float32))
            recorder.add(obs=np.zeros((2, 53), dtype=np.float32))
            with self.assertRaises(RuntimeError):
                recorder.finalize(0, eval_seed=0, done_step=2, outcome="fall", final_stage=0)

    def test_unknown_feature_rejected(self) -> None:
        recorder = _recorder(Path(tempfile.mkdtemp()))
        with self.assertRaises(ValueError):
            recorder.add(not_a_feature=np.zeros((1,), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
