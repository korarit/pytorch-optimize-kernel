#!/usr/bin/env python3
"""Phase 2 consolidated verification script: Unit tests + Migrate tests + Smoke A/B benchmark."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent.parent / "pipline-for-test"
REPO_ROOT = BENCH_DIR.parent
for _p in (str(REPO_ROOT), str(BENCH_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bench import run_benchmark_suite
from report import build_comparison_rows
from src.common.text_report import TextReportBuilder
from src.config import ConfigResolver
from src.detect import collect_fingerprint


def _run_suite(target_dir: Path) -> tuple[str, int, str]:
    rel = target_dir.relative_to(REPO_ROOT)
    if importlib.util.find_spec("pytest") is not None:
        cmd = [sys.executable, "-m", "pytest", str(target_dir), "-v"]
        runner = "pytest"
    else:
        cmd = [sys.executable, "-m", "unittest", "discover", "-s", str(rel), "-v"]
        runner = "unittest"
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    return runner, res.returncode, (res.stdout + "\n" + res.stderr).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Phase 2 (Unit + Migrate + Smoke Benchmark).")
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    fp = collect_fingerprint()
    target_device = "cuda" if fp.effective_backend == "cuda" else "cpu"
    bench_steps = args.steps if target_device == "cuda" else min(args.steps, 6)

    unit_runner, unit_rc, unit_out = _run_suite(REPO_ROOT / "tests" / "unit")
    mig_runner, mig_rc, mig_out = _run_suite(REPO_ROOT / "tests" / "migrate")

    workloads = ["cnn_vision", "transformer_lm", "deep_mlp", "infer_runner_bs1", "infer_runner_bs32", "edge_cases"]
    records = run_benchmark_suite(
        workloads=workloads,
        profiles=["safe"],
        cache_modes=["warm"],
        steps=bench_steps,
        device=target_device,
        seed=42,
        trials=1,
    )
    comparisons = build_comparison_rows(records)
    bench_fail = any(r.exit_code != 0 for r in records) or any(c.status == "FAIL" for c in comparisons)
    overall_rc = 0 if (unit_rc == 0 and mig_rc == 0 and not bench_fail) else 1

    safe_cfg = ConfigResolver().resolve(fp, profile="safe", env={}).config.to_dict()

    builder = TextReportBuilder(
        phase_title="PHASE 2 CONSOLIDATED REPORT: WRAPPER CORE, BENCHMARK & MIGRATE TESTS",
        repo_root=REPO_ROOT,
    )
    builder.add_header({
        "Google Colab": fp.is_colab,
        "Backend / GPU": (
            f"{fp.effective_backend} / {fp.primary_gpu.name} ({fp.primary_gpu.sm_name})"
            if fp.primary_gpu
            else fp.effective_backend
        ),
        "Unit Tests": f"{unit_runner} (exit_code={unit_rc})",
        "Migrate Tests": f"{mig_runner} (exit_code={mig_rc})",
        "Benchmark Smoke": "PASS" if not bench_fail else "FAIL",
    })
    builder.add_kv_section("SAFE PROFILE CONFIG", {
        "tf32": safe_cfg["runtime"]["tf32"],
        "amp": f"{safe_cfg['amp']['enabled']} ({safe_cfg['amp']['dtype']})",
        "compile": safe_cfg["compile"]["enabled"],
        "fused_optimizer": safe_cfg["optimizer"]["fused"],
        "dataloader_auto_tune": safe_cfg["dataloader"]["auto_tune"],
    })
    comp_rows = [
        [
            c.workload,
            c.mode,
            f"{c.base_median_ms:.2f}",
            f"{c.wrap_median_ms:.2f}",
            f"{c.wrap_p95_ms:.2f}",
            f"{c.speedup:.2f}x",
            f"{c.base_e2e_s:.2f}s / {c.wrap_e2e_s:.2f}s",
            f"{c.base_vram_mb:.1f} / {c.wrap_vram_mb:.1f} ({c.vram_delta_mb:+.1f})",
            c.max_rel_diff if c.max_rel_diff is not None else "N/A",
            c.status,
        ]
        for c in comparisons
    ]
    builder.add_table(
        "SMOKE BENCHMARK COMPARISON (Baseline vs safe)",
        ["Workload", "Mode", "Base ms", "Wrap ms", "Wrap P95", "Speedup", "E2E (B/W)", "VRAM MB (B/W/d)", "MaxRelDiff", "Status"],
        comp_rows,
    )
    builder.add_lines_section(f"MIGRATE TEST OUTPUT ({mig_runner})", mig_out.splitlines())
    builder.add_lines_section(f"UNIT TEST OUTPUT ({unit_runner})", unit_out.splitlines())
    builder.add_verification_footer(
        status="PENDING_USER_REVIEW" if overall_rc == 0 else "FAILED_NEEDS_ATTENTION",
        note="Verify Phase 2 consolidated report on Google Colab before entering Phase 3 Optimization Lab",
    )

    out_path = builder.write("phase-2", report_dir=args.report_dir)
    print(builder.render())
    print(f"[SAVED] Text report written to: {out_path}")
    return overall_rc


if __name__ == "__main__":
    raise SystemExit(main())
