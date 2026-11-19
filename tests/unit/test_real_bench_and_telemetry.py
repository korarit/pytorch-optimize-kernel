"""Unit and subprocess tests for real-world benchmark harness, artifact bundles, --telemetry-only, and deep diagnostics."""

from __future__ import annotations

import datetime
import io
import json
import os
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = REPO_ROOT / "pipline-for-test"
for _p in (str(REPO_ROOT), str(BENCH_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ptopt.cli.main import build_child_env, main as ptopt_main, parse_ptopt_argv
from ptopt.config import OptimizeConfig
from ptopt.hooks import HookManager
from ptopt.hooks.telemetry_hook import TelemetryCollector
from real_bench import (
    RealTargetSpec,
    _run_subprocess,
    export_real_csv,
    export_real_json,
    extract_metrics_from_output,
    generate_real_benchmark_report,
    load_manifest_specs,
    main as real_bench_main,
    parse_metric_specs,
    run_real_target,
)


class TestTelemetryOnlyMode(unittest.TestCase):
    """Verify `--telemetry-only` keeps PyTorch and prelaunch env untouched while recording telemetry."""

    def test_cli_parse_and_build_child_env_skips_prelaunch_env(self) -> None:
        parsed = parse_ptopt_argv([
            "--telemetry-only",
            "--telemetry-out",
            "/tmp/out.json",
            "--",
            "python3",
            "train.py",
        ])
        self.assertTrue(parsed.ptopt_args.telemetry_only)
        self.assertEqual(parsed.ptopt_args.telemetry_out, Path("/tmp/out.json"))

        clean_base = {k: v for k, v in os.environ.items() if k != "PYTORCH_CUDA_ALLOC_CONF"}
        env = build_child_env(
            OptimizeConfig(profile="aggressive").to_dict(),
            telemetry_only=True,
            telemetry_out="/tmp/out.json",
            base_env=clean_base,
            repo_root=REPO_ROOT,
        )
        self.assertEqual(env.get("PTOPT_TELEMETRY_ONLY"), "1")
        self.assertEqual(env.get("PTOPT_DISABLE_BUILTIN_PASSES"), "1")
        self.assertEqual(env.get("PTOPT_TELEMETRY_OUT"), "/tmp/out.json")
        self.assertNotIn("PYTORCH_CUDA_ALLOC_CONF", env)

    def test_telemetry_only_subprocess_skips_passes_and_captures_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            tel_path = tmp_dir / "base_tel.json"
            script_path = tmp_dir / "toy_train.py"
            script_path.write_text(
                "\n".join([
                    "import os, torch, torch.nn as nn, torch.nn.functional as F",
                    "assert 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ",
                    "assert not hasattr(torch.save, '__wrapped__')",
                    "m = nn.Linear(4, 2)",
                    "opt = torch.optim.SGD(m.parameters(), lr=0.05)",
                    "for _ in range(3):",
                    "    opt.zero_grad()",
                    "    loss = F.cross_entropy(m(torch.randn(2, 4)), torch.randint(0, 2, (2,)))",
                    "    loss.backward()",
                    "    opt.step()",
                    "print('OK')",
                ]) + "\n",
                encoding="utf-8",
            )
            env_backup = os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
            try:
                rc = ptopt_main([
                    "--profile",
                    "aggressive",
                    "--telemetry-only",
                    "--telemetry-out",
                    str(tel_path),
                    "--",
                    sys.executable,
                    str(script_path),
                ])
            finally:
                if env_backup is not None:
                    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = env_backup

            self.assertEqual(rc, 0)
            self.assertTrue(tel_path.is_file())
            data = json.loads(tel_path.read_text(encoding="utf-8"))
            self.assertEqual(data["mode"], "baseline")
            self.assertTrue(data["telemetry_only"])
            self.assertEqual(len(data["train_step_times_ms"]), 3)
            self.assertEqual(len(data["step_times_ms"]), 3)
            self.assertEqual(len(data["losses"]), 3)
            self.assertEqual(data["fallback_events"][0]["status"], "SKIPPED")


class TestInterleavedAndMultiRankTelemetry(unittest.TestCase):
    """Verify TelemetryCollector isolates eval loops, tracks module shapes/Dynamo stats, and guards workers."""

    def test_interleaved_train_eval_module_trace_and_dynamo_capture(self) -> None:
        for dbg in (False, True):
            with tempfile.TemporaryDirectory() as tmp:
                tel_path = Path(tmp) / "tel.json"
                mgr = HookManager()
                mgr.config = OptimizeConfig(profile="safe")
                collector = TelemetryCollector(tel_path, mgr, debug_profile=dbg)
                collector.install(torch)
                try:
                    model = nn.Linear(8, 4)
                    opt = torch.optim.SGD(model.parameters(), lr=0.01)

                    def _run_train_steps(count: int, bs: int) -> None:
                        model.train()
                        for _ in range(count):
                            opt.zero_grad()
                            loss = F.cross_entropy(model(torch.randn(bs, 8)), torch.randint(0, 4, (bs,)))
                            loss.backward()
                            opt.step()

                    _run_train_steps(2, 4)
                    model.eval()
                    with torch.no_grad():
                        for _ in range(2):
                            time.sleep(0.03)
                            _ = model(torch.randn(3, 8))
                    _run_train_steps(2, 6)

                    fake_du = SimpleNamespace(
                        counters={
                            "frames": {"total": 2, "ok": 2},
                            "stats": {"calls_captured": 4, "unique_graphs": 2},
                            "graph_break": {"Data-dependent branching": 1},
                            "inductor": {"fxgraph_cache_hit": 1, "fxgraph_cache_miss": 1},
                        },
                        compile_times=lambda repr="str": "compile_attempt_0, 0.25",
                    )
                    with patch.dict(sys.modules, {"torch._dynamo.utils": fake_du}):
                        collector.flush()
                finally:
                    collector.uninstall()
                    mgr.reset()

                snap = json.loads(tel_path.read_text(encoding="utf-8"))
                self.assertEqual(len(snap["train_step_times_ms"]), 4)
                self.assertEqual(len(snap["step_times_ms"]), 4)
                self.assertEqual(len(snap["eval_step_times_ms"]), 2)
                self.assertLess(snap["train_step_times_ms"][2], 40.0)
                if dbg:
                    dbg_blk = snap["debug_profile"]
                    self.assertEqual(len(dbg_blk["phase_breakdown"]["per_step_trace"]), 4)
                    self.assertIn("Linear", dbg_blk["module_summary"])
                    mod_info = dbg_blk["module_summary"]["Linear"]
                    self.assertEqual(mod_info["train_calls"], 4)
                    self.assertEqual(mod_info["eval_calls"], 2)
                    self.assertEqual(len(mod_info["input_shapes"]), 3)
                    self.assertEqual(dbg_blk["dynamo_compile"]["summary"]["graph_break_count"], 1)
                    self.assertEqual(dbg_blk["dynamo_compile"]["summary"]["inductor_cache_hit"], 1)

    def test_rank_suffix_and_dataloader_worker_skip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base_path = Path(tmp) / "tel.json"
            mgr = HookManager()
            mgr.config = OptimizeConfig(profile="safe")

            with patch.dict(os.environ, {"RANK": "1"}):
                c1 = TelemetryCollector(base_path, mgr)
                c1.flush()
            self.assertFalse(base_path.exists())
            self.assertTrue((Path(tmp) / "tel.rank1.json").is_file())

            with patch("ptopt.hooks.telemetry_hook._is_dataloader_worker", return_value=True):
                c2 = TelemetryCollector(base_path, mgr)
                c2._torch = torch
                c2.flush()
            self.assertFalse(base_path.exists())
            mgr.reset()


class TestRealBenchHarness(unittest.TestCase):
    """End-to-end test for `pipline-for-test/real_bench.py` including bundle folder & deep logs."""

    def test_metric_extraction_bundle_and_four_arm_run(self) -> None:
        regexes = parse_metric_specs(["val_loss=val_loss=([0-9.]+)", "acc=Acc@1 ([0-9.]+)"])
        found = extract_metrics_from_output(
            "step 1 val_loss=2.10\nstep 2 val_loss=1.85 Acc@1 74.5\n",
            regexes,
        )
        self.assertEqual(found, {"val_loss": 1.85, "acc": 74.5})

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            ext_repo = tmp_dir / "cloned_github_repo"
            (ext_repo / "src").mkdir(parents=True)
            (ext_repo / "src" / "__init__.py").write_text("", encoding="utf-8")
            (ext_repo / "src" / "model.py").write_text(
                "\n".join([
                    "import torch.nn as nn",
                    "class TinyNet(nn.Module):",
                    "    def __init__(self):",
                    "        super().__init__()",
                    "        self.fc = nn.Linear(8, 4)",
                    "    def forward(self, x):",
                    "        return self.fc(x)",
                ]) + "\n",
                encoding="utf-8",
            )
            (ext_repo / "train.py").write_text(
                "\n".join([
                    "import argparse, torch, torch.nn.functional as F",
                    "from src.model import TinyNet",
                    "p = argparse.ArgumentParser()",
                    "p.add_argument('--native-opt', action='store_true')",
                    "args = p.parse_args()",
                    "torch.manual_seed(42)",
                    "m = TinyNet()",
                    "opt = torch.optim.SGD(m.parameters(), lr=0.05)",
                    "for _ in range(4):",
                    "    opt.zero_grad()",
                    "    loss = F.cross_entropy(m(torch.randn(4, 8)), torch.randint(0, 4, (4,)))",
                    "    loss.backward()",
                    "    opt.step()",
                    "m.eval()",
                    "with torch.no_grad():",
                    "    vloss = F.cross_entropy(m(torch.randn(4, 8)), torch.randint(0, 4, (4,))).item()",
                    "print(f'val_loss={vloss:.4f}')",
                ]) + "\n",
                encoding="utf-8",
            )

            bundle_dir = tmp_dir / "bundle_out"
            spec = RealTargetSpec(
                name="ext-tiny",
                cwd=ext_repo,
                cmd=[sys.executable, "train.py"],
                native_cmd=[sys.executable, "train.py", "--native-opt"],
                test_conflict=True,
                profiles=("safe",),
                cache_modes=("auto",),
                metric_regexes={"val_loss": r"val_loss=([0-9.]+)"},
                measure_pure_wall=True,
                diag_mode="deep",
            )
            records = run_real_target(spec, cache_root=tmp_dir / "caches", bundle_dir=bundle_dir)
            arms = [r.arm for r in records]
            self.assertEqual(arms, ["baseline", "safe", "native_opt", "conflict:balanced"])
            self.assertTrue(all(r.exit_code == 0 for r in records))
            self.assertTrue(all(r.pure_wall_sec is not None for r in records))
            self.assertTrue(all("val_loss" in r.extracted_metrics for r in records))
            self.assertTrue(all("phase_breakdown" in r.debug_profile for r in records))
            self.assertTrue(all("torch_profiler" in r.debug_profile for r in records))

            # Verify per-run artifact files
            for rec in records:
                assert rec.artifact_dir is not None
                adir = Path(rec.artifact_dir)
                for fname in (
                    "run_meta.json",
                    "stdout_stderr.log",
                    "telemetry.json",
                    "ptopt_events.json",
                    "ptopt_fallback.txt",
                    "debug_profile.json",
                    "diag_stderr.log",
                    "pure_wall.log",
                ):
                    self.assertTrue((adir / fname).is_file(), f"Missing {fname} in {adir}")
                if rec.arm not in ("baseline", "native_opt"):
                    self.assertTrue((adir / "resolved_config.json").is_file())

            report_path, text, comparisons = generate_real_benchmark_report(
                records=records,
                specs_by_name={spec.name: spec},
                report_name="real-test",
                report_dir=tmp_dir / "reports",
                bundle_dir=bundle_dir,
                now=datetime.datetime(2026, 11, 19, 14, 50, 0),
            )
            self.assertEqual(report_path.name, "145000_20261119-real-test.txt")
            self.assertTrue((bundle_dir / "summary.txt").is_file())
            self.assertIn("4-ARM REAL PIPELINE COMPARISON", text)
            self.assertIn("PHASE BREAKDOWN COMPARISON", text)
            self.assertIn("TOP CUDA & CPU OPS HOTSPOTS", text)
            self.assertIn("TOP-LEVEL MODULE & DYNAMIC SHAPE TRACE", text)
            self.assertIn("ACTIONABLE OPTIMIZATION HINTS", text)
            self.assertEqual(len(comparisons), 3)
            self.assertTrue(all(c.status == "PASS" for c in comparisons))

            json_p = export_real_json(bundle_dir / "summary.json", records, comparisons)
            csv_p = export_real_csv(bundle_dir / "comparisons.csv", comparisons)
            self.assertTrue(json_p.is_file())
            self.assertTrue(csv_p.is_file())

            # Verify CLI entrypoint prints live per-profile cards in order and creates results bundle
            cli_results_dir = tmp_dir / "cli_results"
            cli_reports_dir = tmp_dir / "cli_reports"
            stdout_buf = io.StringIO()
            with redirect_stdout(stdout_buf):
                rc_cli = real_bench_main([
                    "--name",
                    "cli-tiny",
                    "--cwd",
                    str(ext_repo),
                    "--cmd",
                    "python3 train.py",
                    "--profiles",
                    "safe",
                    "--no-pure-wall",
                    "--diag-mode",
                    "off",
                    "--stream",
                    "--results-dir",
                    str(cli_results_dir),
                    "--report-dir",
                    str(cli_reports_dir),
                ])
            self.assertEqual(rc_cli, 0)
            cli_out = stdout_buf.getvalue()
            idx_base_done = cli_out.find("[DONE] cli-tiny | BASELINE")
            idx_safe_start = cli_out.find("[2/2] cli-tiny | arm=safe")
            idx_safe_done = cli_out.find("[DONE] cli-tiny | arm=safe")
            idx_final_hdr = cli_out.find("KERNEL-PYTOUCH-OPTIMIZE | REAL-WORLD")
            self.assertGreater(idx_base_done, -1)
            self.assertGreater(idx_safe_start, idx_base_done)
            self.assertGreater(idx_safe_done, idx_safe_start)
            self.assertGreater(idx_final_hdr, idx_safe_done)
            bundles = list(cli_results_dir.iterdir())
            self.assertEqual(len(bundles), 1)
            self.assertTrue((bundles[0] / "env_hardware.json").is_file())
            self.assertTrue((bundles[0] / "summary.json").is_file())
            self.assertTrue((bundles[0] / "comparisons.csv").is_file())
            self.assertTrue((bundles[0] / "summary.txt").is_file())

            manifest_p = tmp_dir / "manifest.json"
            manifest_p.write_text(
                json.dumps({
                    "targets": [{
                        "name": "manifest-tiny",
                        "cwd": str(ext_repo),
                        "cmd": "python3 train.py",
                        "profiles": ["safe"],
                    }]
                }),
                encoding="utf-8",
            )
            loaded = load_manifest_specs(manifest_p)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].name, "manifest-tiny")
            self.assertEqual(loaded[0].cmd[0], sys.executable)
            self.assertEqual(loaded[0].timeout_sec, 3600.0)
            self.assertEqual(RealTargetSpec(name="x", cwd=tmp_dir, cmd=[sys.executable]).timeout_sec, 3600.0)

    def test_subprocess_timeout_kills_process_group_and_keeps_stdout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            pid_file = tmp_dir / "grandchild.pid"
            script = tmp_dir / "hang_with_child.py"
            script.write_text(
                "\n".join([
                    "import pathlib, subprocess, sys, time",
                    f"child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])",
                    f"pathlib.Path({str(pid_file)!r}).write_text(str(child.pid), encoding='utf-8')",
                    "print('BEFORE_TIMEOUT', flush=True)",
                    "time.sleep(60)",
                ]) + "\n",
                encoding="utf-8",
            )
            rc, out, err, _wall, _, _, _ = _run_subprocess(
                [sys.executable, str(script)],
                cwd=tmp_dir,
                env=os.environ,
                timeout_sec=0.8,
                sample_gpu=False,
                stream_output=False,
            )
            self.assertEqual(rc, 124)
            self.assertIn("BEFORE_TIMEOUT", out)
            self.assertIn("Timed out after", err)
            if os.name == "posix" and pid_file.is_file():
                grandchild_pid = int(pid_file.read_text(encoding="utf-8").strip())
                time.sleep(0.1)
                stat_path = Path(f"/proc/{grandchild_pid}/stat")
                if stat_path.exists():
                    # Killed child reparented to PID 1 may linger briefly as a 'Z' (zombie) in containers.
                    stat_txt = stat_path.read_text(encoding="utf-8", errors="replace")
                    state = stat_txt.rsplit(")", 1)[-1].strip().split()[0]
                    self.assertEqual(state, "Z", f"Expected grandchild killed/zombie, got: {stat_txt}")
                else:
                    with self.assertRaises(ProcessLookupError):
                        os.kill(grandchild_pid, 0)


if __name__ == "__main__":
    unittest.main()
