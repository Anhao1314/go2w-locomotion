"""因子库契约测试（纯逻辑，Windows CI 可跑）。"""

from __future__ import annotations

import unittest

from analysis.factors import (
    BY_NAME,
    FACTORS,
    catalog,
    gate_check,
    percentile,
    rollout_factors,
    run_factors,
    summarize,
)


class TestRegistry(unittest.TestCase):
    def test_names_unique(self) -> None:
        names = [f.name for f in FACTORS]
        self.assertEqual(len(names), len(set(names)))

    def test_levels_and_directions_are_valid(self) -> None:
        levels = {"rollout", "run", "frame", "train_step", "semantic"}
        directions = {"lower", "higher", "neutral"}
        for f in FACTORS:
            self.assertIn(f.level, levels, f.name)
            self.assertIn(f.direction, directions, f.name)
            self.assertTrue(f.definition, f.name)
            self.assertTrue(f.source, f.name)

    def test_catalog_counts_match(self) -> None:
        cat = catalog()
        self.assertEqual(cat["total"], len(FACTORS))
        self.assertEqual(cat["available"] + cat["missing"], cat["total"])


class TestRolloutFactors(unittest.TestCase):
    def test_successful_rollout(self) -> None:
        rec = {"success": True, "fall": False, "steps": 474, "control_dt": 0.02,
               "terminal_miss": 0.15, "tolerance": 0.30, "vx_rmse": 0.08}
        out = rollout_factors(rec)
        self.assertEqual(out["task_success"], 1.0)
        self.assertEqual(out["task_steps"], 474)
        self.assertAlmostEqual(out["task_time_s"], 9.48)
        self.assertAlmostEqual(out["task_miss_ratio"], 0.5)
        self.assertEqual(out["stab_fall"], 0.0)

    def test_failed_rollout_has_no_completion_factors(self) -> None:
        rec = {"success": False, "fall": True, "steps": 52, "terminal_miss": 2.4,
               "tolerance": 0.30}
        out = rollout_factors(rec)
        self.assertIsNone(out["task_steps"])
        self.assertIsNone(out["task_miss_ratio"])
        self.assertEqual(out["stab_fall"], 1.0)

    def test_fall_delay_conversion(self) -> None:
        rec = {"fall": True, "fail_after_push_steps": 60}
        self.assertAlmostEqual(rollout_factors(rec)["stab_fall_after_push_s"], 1.2)

    def test_missing_fields_are_none(self) -> None:
        out = rollout_factors({"vx_rmse": 0.1})
        self.assertIsNone(out["task_success"])
        self.assertIsNone(out["stab_fall"])


class TestAggregation(unittest.TestCase):
    def test_run_factors_mean_and_rate(self) -> None:
        rollouts = [
            {"success": True, "fall": False, "steps": 400, "vx_rmse": 0.05, "wz_rmse": 0.10},
            {"success": True, "fall": False, "steps": 500, "vx_rmse": 0.10, "wz_rmse": 0.20},
            {"success": False, "fall": True, "steps": 50, "vx_rmse": 0.15, "wz_rmse": 0.30},
        ]
        agg = run_factors("t", rollouts)["factors"]
        self.assertAlmostEqual(agg["task_success_rate"], 2 / 3)
        self.assertAlmostEqual(agg["stab_fall_rate"], 1 / 3)
        self.assertAlmostEqual(agg["task_steps"], 450.0)  # 只统计成功 episode
        self.assertAlmostEqual(agg["track_axis_ratio_wz_vx"], 2.0)

    def test_axis_ratio(self) -> None:
        rollouts = [{"vx_rmse": 0.1, "wz_rmse": 0.2}]
        agg = run_factors("t", rollouts)["factors"]
        self.assertAlmostEqual(agg["track_axis_ratio_wz_vx"], 2.0)


class TestGatesAndStats(unittest.TestCase):
    def test_lower_is_better_gate(self) -> None:
        res = gate_check("track_vx_rmse", 0.0773)
        self.assertTrue(res["pass"])
        self.assertAlmostEqual(res["margin"], 0.15 / 0.0773, places=6)

    def test_higher_is_better_gate(self) -> None:
        self.assertTrue(gate_check("task_success_rate", 1.0)["pass"])
        self.assertFalse(gate_check("task_success_rate", 0.5)["pass"])

    def test_no_gate_returns_none(self) -> None:
        self.assertIsNone(gate_check("track_wz_rmse", 0.1)["pass"])

    def test_percentiles(self) -> None:
        vals = [0.0, 1.0, 2.0, 3.0, 4.0]
        self.assertAlmostEqual(percentile(vals, 0.5), 2.0)
        self.assertAlmostEqual(percentile(vals, 0.0), 0.0)
        self.assertAlmostEqual(percentile(vals, 1.0), 4.0)

    def test_summarize_ignores_none(self) -> None:
        res = summarize([1.0, None, 3.0])
        self.assertEqual(res["n"], 2)
        self.assertAlmostEqual(res["mean"], 2.0)


if __name__ == "__main__":
    unittest.main()
