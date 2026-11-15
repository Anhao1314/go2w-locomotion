"""因子体检的契约测试（纯逻辑，Windows CI 可跑）。"""

from __future__ import annotations

import unittest

from analysis.audit import (
    cohen_d,
    discrimination,
    pair_kind,
    redundancy,
    verdict,
    within_run_degenerate,
)


class TestDiscrimination(unittest.TestCase):
    def test_within_run_preferred_over_pooled(self) -> None:
        # 同一 run 内 x 恒定 → 组内无法估计，必须标 confounded
        rows = [
            {"_run_id": "A", "x": 0.1, "fall": 0.0},
            {"_run_id": "A", "x": 0.1, "fall": 0.0},
            {"_run_id": "A", "x": 0.1, "fall": 1.0},
            {"_run_id": "A", "x": 0.1, "fall": 1.0},
            {"_run_id": "B", "x": 5.0, "fall": 1.0},
            {"_run_id": "B", "x": 5.0, "fall": 1.0},
            {"_run_id": "B", "x": 5.0, "fall": 0.0},
            {"_run_id": "B", "x": 5.0, "fall": 0.0},
        ]
        res = discrimination(rows, ["x"], "fall")["x"]
        self.assertIsNone(res["d_within"])
        self.assertTrue(res["confounded"])

    def test_simpson_confound_is_flagged(self) -> None:
        # run A：低误差全不摔；run B：高误差全摔 → pooled d 很大，但纯属 run 间差异
        rows = (
            [{"_run_id": "A", "x": v, "fall": 0.0} for v in (0.10, 0.12, 0.11, 0.13, 0.09)]
            + [{"_run_id": "B", "x": v, "fall": 1.0} for v in (5.00, 5.10, 4.90, 5.05, 4.95)]
        )
        res = discrimination(rows, ["x"], "fall")["x"]
        self.assertIsNone(res["d_within"])      # 组内没有对照，估不出来
        self.assertTrue(res["confounded"])      # 必须标注混杂
        self.assertIsNotNone(res["d_pooled"])   # pooled 看着很大，但不能用
        self.assertGreater(abs(res["d_pooled"]), 2.0)

    def test_within_run_estimate_available(self) -> None:
        rows = [
            {"_run_id": "A", "x": 1.0, "fall": 0.0},
            {"_run_id": "A", "x": 1.2, "fall": 0.0},
            {"_run_id": "A", "x": 3.0, "fall": 1.0},
            {"_run_id": "A", "x": 3.2, "fall": 1.0},
        ]
        res = discrimination(rows, ["x"], "fall")["x"]
        self.assertIsNotNone(res["d_within"])
        self.assertEqual(res["runs_used"], ["A"])
        self.assertFalse(res["confounded"])

    def test_cohen_d_sign(self) -> None:
        self.assertGreater(cohen_d([3.0, 3.2], [1.0, 1.2]) or 0.0, 0.0)


class TestDegeneracy(unittest.TestCase):
    def test_constant_within_run_flagged(self) -> None:
        rows = [{"_run_id": "A", "x": 5.0} for _ in range(6)]
        self.assertEqual(within_run_degenerate(rows, "x"), ["A"])

    def test_varying_not_flagged(self) -> None:
        rows = [{"_run_id": "A", "x": float(i)} for i in range(6)]
        self.assertEqual(within_run_degenerate(rows, "x"), [])

    def test_small_group_ignored(self) -> None:
        rows = [{"_run_id": "A", "x": 5.0} for _ in range(3)]
        self.assertEqual(within_run_degenerate(rows, "x"), [])


class TestRedundancy(unittest.TestCase):
    def test_alias_detected(self) -> None:
        rows = [{"task_steps": float(i), "task_time_s": float(i) * 0.02} for i in range(20)]
        pairs = redundancy(rows, ["task_steps", "task_time_s"])
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0][4], "构造别名")

    def test_pair_kind(self) -> None:
        self.assertEqual(pair_kind("task_success", "stab_fall"), "结果同源")
        self.assertEqual(pair_kind("track_vx_rmse", "track_wz_rmse"), "真冗余")


class TestVerdict(unittest.TestCase):
    def _info(self, **kwargs):
        base = {"coverage": 0.5, "constant": False, "binary": False,
                "within_run_dead": [], "outcome_like": False, "lever": "x"}
        base.update(kwargs)
        return base

    def test_outcome_like_is_separate(self) -> None:
        self.assertEqual(verdict("task_success", self._info(outcome_like=True), set()),
                         "结局指标")

    def test_constant_is_degenerate(self) -> None:
        self.assertEqual(verdict("x", self._info(constant=True), set()), "退化（常数）")

    def test_no_lever_is_diagnostic(self) -> None:
        self.assertEqual(verdict("x", self._info(lever=""), set()), "诊断项")


if __name__ == "__main__":
    unittest.main()
