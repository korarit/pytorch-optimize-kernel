"""Unit tests for Phase 2.3: telemetry stats, parity, telemetry hook, and text report generation."""

from __future__ import annotations

import datetime
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = REPO_ROOT / "pipline-for-test"
for _p in (str(REPO_ROOT), str(BENCH_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from report import build_comparison_rows, export_csv_report, export_json_report, generate_benchmark_text_report
from ptopt.config import OptimizeConfig
from ptopt.hooks import HookManager
from ptopt.hooks.telemetry_hook import TelemetryCollector
from telemetry import (
    RunRecord,
    StepStats,
    check_parity,
    compute_speedup,
    compute_step_stats,
    compute_vram_delta,
    percentile,
)


class TestTelemetryMathAndParity(unittest.TestCase):
    """Tests for percentile, steady-state stats, speedup, VRAM delta, and parity thresholds."""

    def test_percentile_and_step_stats(self) -> None:
        self.assertEqual(percentile([], 50.0), 0.0)
        self.assertEqual(percentile([10.0], 95.0), 10.0)
        # 10 samples -> warmup=2, steady=8 samples [3..10]
        steps = [100.0, 50.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 20.0]
        stats = compute_step_stats(steps)
        self.assertEqual(stats.warmup_steps, 2)
        self.assertEqual(stats.steady_steps, 8)
        self.assertAlmostEqual(stats.median, 13.5, places=2)
        self.assertGreater(stats.p95, stats.median)
        self.assertEqual(stats.min, 10.0)
        self.assertEqual(stats.max, 20.0)

    def test_speedup_and_vram_delta(self) -> None:
        self.assertEqual(compute_speedup(100.0, 50.0), 2.0)
        self.assertEqual(compute_speedup(0.0, 50.0), 0.0)
        self.assertEqual(compute_vram_delta(1000.0, 750.5), -249.5)

    def test_parity_pass_fail_and_nan(self) -> None:
        res_ok = check_parity([1.0, 0.5], [1.0005, 0.5002], profile="safe")
        self.assertTrue(res_ok.passed)

        res_fail = check_parity([1.0, 0.5], [1.05, 0.5], profile="safe")
        self.assertFalse(res_fail.passed)
        self.assertFalse(res_fail.has_non_finite)

        res_nan = check_parity([1.0, 0.5], [1.0, float("nan")], profile="aggressive")
        self.assertFalse(res_nan.passed)
        self.assertTrue(res_nan.has_non_finite)


class TestTelemetryHookDummyLoop(unittest.TestCase):
    """Tests TelemetryCollector on a dummy CPU PyTorch training loop."""

    def test_collector_captures_steps_and_losses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "tel.json"
            mgr = HookManager()
            mgr.config = OptimizeConfig(profile="safe")
            mgr.record_event("runtime.tf32", "SKIPPED", "CPU-only")

            collector = TelemetryCollector(out_path, mgr)
            collector.install(torch)
            try:
                model = nn.Linear(8, 4)
                opt = torch.optim.SGD(model.parameters(), lr=0.01)
                for _ in range(4):
                    x = torch.randn(3, 8)
                    y = torch.randint(0, 4, (3,))
                    opt.zero_grad()
                    loss = F.cross_entropy(model(x), y)
                    loss.backward()
                    opt.step()
                collector.flush()
            finally:
                collector.uninstall()
                mgr.reset()

            self.assertTrue(out_path.is_file())
            data = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(data["telemetry_schema_version"], 1)
            self.assertEqual(len(data["step_times_ms"]), 4)
            self.assertEqual(len(data["losses"]), 4)
            self.assertEqual(len(data["fallback_events"]), 1)


class TestBenchmarkReportFormatter(unittest.TestCase):
    """Tests Detailed Text Report generation including crash/skip resilience."""

    def test_report_generation_handles_pass_and_fail(self) -> None:
        base_ok = RunRecord(
            workload="cnn_vision",
            mode="baseline",
            cache_state="none",
            exit_code=0,
            end_to_end_sec=2.0,
            step_stats=StepStats(40.0, 45.0, 38.0, 46.0, 2.0, 41.0, 2, 8),
            throughput_value=400.0,
            throughput_unit="samples/sec",
            peak_allocated_mb=120.0,
            peak_reserved_mb=150.0,
            losses=[2.0, 1.8],
        )
        wrap_ok = RunRecord(
            workload="cnn_vision",
            mode="safe",
            cache_state="warm",
            exit_code=0,
            end_to_end_sec=1.5,
            step_stats=StepStats(25.0, 28.0, 24.0, 29.0, 1.2, 25.5, 2, 8),
            throughput_value=640.0,
            throughput_unit="samples/sec",
            peak_allocated_mb=100.0,
            peak_reserved_mb=128.0,
            losses=[2.0001, 1.8002],
            fallback_events=[{"pass_name": "kernels.rmsnorm", "status": "SKIPPED", "reason": "disabled in safe"}],
        )
        base_fail = RunRecord(
            workload="broken_wl",
            mode="baseline",
            cache_state="none",
            exit_code=0,
            end_to_end_sec=1.0,
            step_stats=StepStats(10.0, 12.0, 9.0, 13.0, 1.0, 10.5, 1, 4),
            throughput_value=100.0,
            throughput_unit="samples/sec",
            peak_allocated_mb=50.0,
            peak_reserved_mb=64.0,
            losses=[1.0],
        )
        wrap_fail = RunRecord(
            workload="broken_wl",
            mode="balanced",
            cache_state="cold",
            exit_code=1,
            end_to_end_sec=0.3,
            step_stats=StepStats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0),
            throughput_value=0.0,
            throughput_unit="samples/sec",
            peak_allocated_mb=0.0,
            peak_reserved_mb=0.0,
            error="Simulated OOM error",
        )

        records = [base_ok, wrap_ok, base_fail, wrap_fail]
        comparisons = build_comparison_rows(records)
        self.assertEqual(len(comparisons), 2)
        self.assertEqual(comparisons[0].status, "PASS")
        self.assertEqual(comparisons[0].speedup, 1.6)
        self.assertEqual(comparisons[1].status, "FAIL")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            fixed_now = datetime.datetime(2026, 11, 17, 16, 35, 0)
            out_path, text = generate_benchmark_text_report(
                phase_title="PHASE 2.3 TEST REPORT",
                report_name="phase-2.3",
                records=records,
                report_dir=tmp_dir,
                repo_root=REPO_ROOT,
                now=fixed_now,
            )
            self.assertEqual(out_path.name, "163500_20261117-phase-2.3.txt")
            self.assertIn("[BENCHMARK COMPARISON (Baseline vs Wrapped)]", text)
            self.assertIn("Simulated OOM error", text)
            self.assertIn("1.60x", text)

            json_p = export_json_report(tmp_dir / "out.json", records, comparisons)
            csv_p = export_csv_report(tmp_dir / "out.csv", comparisons)
            self.assertTrue(json_p.is_file())
            self.assertTrue(csv_p.is_file())


if __name__ == "__main__":
    unittest.main()
