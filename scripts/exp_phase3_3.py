#!/usr/bin/env python3
"""Phase 3.3 Experiment Runner: Optimizer Fusion + Auto-AMP (bf16 & fp16+GradScaler) Ablation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent.parent / "pipline-for-test"
REPO_ROOT = BENCH_DIR.parent
for _p in (str(REPO_ROOT), str(BENCH_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bench import WORKLOADS_DIR, WORKLOAD_SPECS, _extract_workload_summary
from ptopt.common.text_report import TextReportBuilder
from ptopt.detect import collect_fingerprint
from telemetry import RunRecord, check_parity, compute_speedup, compute_vram_delta, merge_summary_and_telemetry


def _run_test_file(rel_path: str) -> tuple[str, int, str]:
    test_file = REPO_ROOT / rel_path
    if importlib.util.find_spec("pytest") is not None:
        cmd = [sys.executable, "-m", "pytest", str(test_file), "-v"]
        runner = "pytest"
    else:
        cmd = [sys.executable, "-m", "unittest", rel_path, "-v"]
        runner = "unittest"
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    return runner, res.returncode, (res.stdout + "\n" + res.stderr).strip()


def _run_variant(
    *,
    wl_name: str,
    steps: int,
    device: str,
    variant_name: str,
    overrides: list[str],
    is_baseline: bool = False,
) -> RunRecord:
    spec = WORKLOAD_SPECS[wl_name]
    script_path = WORKLOADS_DIR / spec["script"]
    wl_args = [
        "--steps",
        str(steps),
        "--batch-size",
        str(spec["batch_size"]),
        "--seed",
        "42",
        "--device",
        device,
        *spec["extra"],
    ]
    env = os.environ.copy()

    with tempfile.TemporaryDirectory() as tmp:
        tel_path = Path(tmp) / "tel.json"
        if is_baseline:
            cmd = [sys.executable, str(script_path), *wl_args]
        else:
            env["PTOPT_TELEMETRY_OUT"] = str(tel_path)
            ovr_flags: list[str] = []
            for item in overrides:
                ovr_flags.extend(["--override", item])
            cmd = [
                sys.executable,
                "-m",
                "ptopt.cli.main",
                "--profile",
                "balanced",
                "--no-compile",
                "--no-dataloader-tune",
                *ovr_flags,
                "--",
                sys.executable,
                str(script_path),
                *wl_args,
            ]
        t0 = time.perf_counter()
        res = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=False)
        wall = time.perf_counter() - t0
        summary = _extract_workload_summary(res.stdout)
        tel_data = json.loads(tel_path.read_text(encoding="utf-8")) if tel_path.is_file() else None
        err = (res.stderr.strip() or res.stdout.strip())[-300:] if res.returncode != 0 else None
        return merge_summary_and_telemetry(
            workload=wl_name,
            mode=variant_name,
            cache_state="none",
            exit_code=res.returncode,
            summary=summary,
            telemetry=tel_data,
            wall_sec=wall,
            error=err,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3.3 Optimizer Fusion & Safe Auto-AMP Experiment")
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    fp = collect_fingerprint()
    target_device = "cuda" if fp.effective_backend == "cuda" else "cpu"
    steps = args.steps if target_device == "cuda" else min(args.steps, 6)
    is_cuda = target_device == "cuda"

    unit_runner, unit_rc, unit_out = _run_test_file("tests/unit/test_pass_optimizer_amp.py")
    mig_runner, mig_rc, mig_out = _run_test_file("tests/migrate/test_phase3_3_amp_fusion.py")

    variants: list[tuple[str, bool, str, list[str]]] = [
        ("baseline", True, "safe", []),
        (
            "3.3a_fused_only",
            False,
            "balanced",
            ["amp.enabled=false", "amp.grad_scaler=false"],
        ),
        (
            "3.3b_fused_plus_amp_auto",
            False,
            "balanced",
            [],
        ),
    ]
    if is_cuda:
        variants.append((
            "3.3c_fused_plus_fp16_scaler",
            False,
            "balanced",
            ["amp.enabled=true", "amp.dtype=fp16", "amp.grad_scaler=true"],
        ))

    workloads = ["transformer_lm", "deep_mlp", "cnn_vision", "edge_cases"]
    rows: list[list[object]] = []
    fb_lines: list[str] = []
    any_fail = False

    for wl in workloads:
        wl_steps = min(steps, 10) if wl == "edge_cases" else steps
        base_rec: RunRecord | None = None
        for var_name, is_base, parity_prof, ov in variants:
            rec = _run_variant(
                wl_name=wl,
                steps=wl_steps,
                device=target_device,
                variant_name=var_name,
                overrides=ov,
                is_baseline=is_base,
            )
            if is_base:
                base_rec = rec
                continue
            assert base_rec is not None
            if base_rec.exit_code != 0 or rec.exit_code != 0:
                any_fail = True
                rows.append([wl, var_name, base_rec.step_stats.median, rec.step_stats.median, "0.00x", "N/A", "FAIL"])
                fb_lines.append(f"[ERROR] {wl} ({var_name}): {rec.error or base_rec.error}")
            else:
                parity = check_parity(base_rec.losses, rec.losses, profile=parity_prof)
                speedup = compute_speedup(base_rec.step_stats.median, rec.step_stats.median)
                vram_d = compute_vram_delta(base_rec.peak_allocated_mb, rec.peak_allocated_mb)
                status = "PASS" if parity.passed else "FAIL"
                if not parity.passed:
                    any_fail = True
                rows.append([
                    wl,
                    var_name,
                    f"{base_rec.step_stats.median:.2f}",
                    f"{rec.step_stats.median:.2f}",
                    f"{rec.step_stats.p95:.2f}",
                    f"{speedup:.2f}x",
                    f"{base_rec.peak_allocated_mb:.1f}/{rec.peak_allocated_mb:.1f} ({vram_d:+.1f})",
                    parity.max_rel_diff,
                    status,
                ])
                for ev in rec.fallback_events:
                    fb_lines.append(f"[{ev.get('status')}] {wl} ({var_name}) -> {ev.get('pass_name')}: {ev.get('reason')}")

    overall_rc = 0 if (unit_rc == 0 and mig_rc == 0 and not any_fail) else 1
    builder = TextReportBuilder(
        phase_title="PHASE 3.3 EXPERIMENT REPORT: OPTIMIZER FUSION & SAFE AUTO-AMP",
        repo_root=REPO_ROOT,
    )
    builder.add_header({
        "Google Colab": fp.is_colab,
        "Backend / GPU": (
            f"{fp.effective_backend} / {fp.primary_gpu.name} ({fp.primary_gpu.sm_name})"
            if fp.primary_gpu
            else fp.effective_backend
        ),
        "Steps per Workload": steps,
        "Unit Tests": f"{unit_runner} (exit_code={unit_rc})",
        "Migrate Tests": f"{mig_runner} (exit_code={mig_rc})",
        "Overall Experiment": "PASS" if overall_rc == 0 else "FAIL",
    })
    builder.add_table(
        "FUSED OPTIMIZER & AUTO-AMP ABLATION",
        ["Workload", "Variant", "Base ms", "Wrap ms", "Wrap P95", "Speedup", "VRAM MB (B/W/d)", "MaxRelDiff", "Status"],
        rows,
    )
    builder.add_lines_section("OPTIMIZER & AUTO-AMP HOOK LOGS", fb_lines)
    builder.add_lines_section(f"MIGRATE TEST OUTPUT ({mig_runner})", mig_out.splitlines())
    builder.add_lines_section(f"UNIT TEST OUTPUT ({unit_runner})", unit_out.splitlines())
    builder.add_verification_footer(
        status="PENDING_USER_REVIEW" if overall_rc == 0 else "FAILED_NEEDS_ATTENTION",
        note="Verify Fused Optimizer & Auto-AMP (bf16 + fp16) speedup, VRAM, and parity on Google Colab before Phase 3.4",
    )

    out_path = builder.write("phase-3.3", report_dir=args.report_dir)
    print(builder.render())
    print(f"[SAVED] Text report written to: {out_path}")
    return overall_rc


if __name__ == "__main__":
    raise SystemExit(main())
