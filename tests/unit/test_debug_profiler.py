"""Unit tests for Phase 3.4 opt-in Deep Debug Profiler in TelemetryCollector."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ptopt.config import OptimizeConfig
from ptopt.hooks import HookManager
from ptopt.hooks.telemetry_hook import TelemetryCollector, maybe_install_telemetry_hook


class TestDebugProfilerTelemetry(unittest.TestCase):
    """Tests for phase timing breakdown, torch.profiler top ops, and default zero-overhead mode."""

    def test_default_mode_has_no_debug_profile_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "tel.json"
            mgr = HookManager()
            mgr.config = OptimizeConfig(profile="balanced")
            collector = maybe_install_telemetry_hook(
                torch,
                mgr,
                env={"PTOPT_TELEMETRY_OUT": str(out_path)},
            )
            assert collector is not None
            try:
                model = nn.Linear(8, 4)
                opt = torch.optim.SGD(model.parameters(), lr=0.01)
                for _ in range(3):
                    opt.zero_grad()
                    loss = model(torch.randn(4, 8)).sum()
                    loss.backward()
                    opt.step()
                snap = collector.snapshot()
                self.assertEqual(len(snap["step_times_ms"]), 3)
                self.assertNotIn("debug_profile", snap)
            finally:
                collector.uninstall()
                mgr.reset()

    def test_debug_and_torch_profiler_captures_phases_and_top_ops(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "tel_debug.json"
            mgr = HookManager()
            mgr.config = OptimizeConfig(profile="balanced")
            collector = maybe_install_telemetry_hook(
                torch,
                mgr,
                env={
                    "PTOPT_TELEMETRY_OUT": str(out_path),
                    "PTOPT_DEBUG_PROFILE": "1",
                    "PTOPT_TORCH_PROFILE": "1",
                    "PTOPT_TORCH_PROFILE_START": "1",
                    "PTOPT_TORCH_PROFILE_STEPS": "3",
                },
            )
            assert collector is not None
            try:
                model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4))
                opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
                for _ in range(5):
                    opt.zero_grad()
                    logits = model(torch.randn(8, 16))
                    loss = torch.nn.functional.cross_entropy(logits, torch.zeros(8, dtype=torch.long))
                    loss.backward()
                    opt.step()

                collector.flush()
                payload = json.loads(out_path.read_text(encoding="utf-8"))
                self.assertIn("debug_profile", payload)
                dbg = payload["debug_profile"]
                self.assertIn("phase_breakdown", dbg)
                pb = dbg["phase_breakdown"]
                self.assertEqual(pb["num_steps"], 5)
                self.assertIn("step_0_cold", pb)
                self.assertIn("steady_mean_ms", pb)
                self.assertIn("steady_shares_pct", pb)

                # Sum of 5 steady mean buckets should match steady mean total
                steady = pb["steady_mean_ms"]
                bucket_sum = (
                    steady["data_other_ms"]
                    + steady["forward_ms"]
                    + steady["loss_clip_ms"]
                    + steady["backward_ms"]
                    + steady["optim_ms"]
                )
                self.assertAlmostEqual(bucket_sum, steady["total_ms"], delta=0.5)

                # Profiler block should contain top CPU ops on CPU-only or CUDA ops on GPU
                self.assertIn("torch_profiler", dbg)
                tp = dbg["torch_profiler"]
                self.assertGreater(len(tp["top_cpu_ops"]), 0)
                self.assertIn("calls", tp["top_cpu_ops"][0])
                self.assertIn("self_cpu_ms", tp["top_cpu_ops"][0])
            finally:
                collector.uninstall()
                mgr.reset()

    def test_uninstall_safely_closes_active_profiler_if_steps_fewer_than_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "tel_short.json"
            mgr = HookManager()
            collector = TelemetryCollector(
                out_path,
                mgr,
                debug_profile=True,
                torch_profile=True,
                profile_start_step=1,
                profile_num_steps=10,
            )
            collector.install(torch)
            try:
                model = nn.Linear(4, 2)
                opt = torch.optim.SGD(model.parameters(), lr=0.01)
                for _ in range(2):
                    opt.zero_grad()
                    model(torch.randn(2, 4)).sum().backward()
                    opt.step()
            finally:
                collector.uninstall()
            snap = collector.snapshot()
            self.assertIn("torch_profiler", snap["debug_profile"])

    def test_eval_only_inference_loop_captures_debug_steps_and_profiler(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "tel_eval_debug.json"
            mgr = HookManager()
            collector = maybe_install_telemetry_hook(
                torch,
                mgr,
                env={
                    "PTOPT_TELEMETRY_OUT": str(out_path),
                    "PTOPT_DEBUG_PROFILE": "1",
                    "PTOPT_TORCH_PROFILE": "1",
                    "PTOPT_TORCH_PROFILE_START": "1",
                    "PTOPT_TORCH_PROFILE_STEPS": "3",
                },
            )
            assert collector is not None
            try:
                model = nn.Sequential(nn.Linear(12, 16), nn.ReLU(), nn.Linear(16, 4))
                model.eval()
                with torch.no_grad():
                    for _ in range(5):
                        _ = model(torch.randn(2, 12))
                collector.flush()
                payload = json.loads(out_path.read_text(encoding="utf-8"))
                dbg = payload["debug_profile"]
                self.assertEqual(dbg["phase_breakdown"]["num_steps"], 5)
                self.assertGreater(len(dbg["torch_profiler"]["top_cpu_ops"]), 0)
            finally:
                collector.uninstall()
                mgr.reset()


if __name__ == "__main__":
    unittest.main()
