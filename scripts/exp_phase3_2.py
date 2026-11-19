#!/usr/bin/env python3
"""Phase 3.2 Experiment Runner: DataLoader worker sweep (0 vs 2 vs 4) + Safety Guard check."""

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

from bench import WORKLOADS_DIR, _extract_workload_summary
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


def _run_case(
    *,
    wl_name: str,
    script_name: str,
    steps: int,
    batch_size: int,
    device: str,
    extra: list[str],
    variant_name: str,
    overrides: list[str] | None,
    is_baseline: bool = False,
) -> RunRecord:
    script_path = WORKLOADS_DIR / script_name
    wl_args = ["--steps", str(steps), "--batch-size", str(batch_size), "--seed", "42", "--device", device, *extra]
    env = os.environ.copy()

    with tempfile.TemporaryDirectory() as tmp:
        tel_path = Path(tmp) / "tel.json"
        if is_baseline:
            cmd = [sys.executable, str(script_path), *wl_args]
        else:
            env["PTOPT_TELEMETRY_OUT"] = str(tel_path)
            ovr_flags: list[str] = []
            for item in overrides or []:
                ovr_flags.extend(["--override", item])
            cmd = [
                sys.executable,
                "-m",
                "ptopt.cli.main",
                "--profile",
                "safe",
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
    parser = argparse.ArgumentParser(description="Phase 3.2 DataLoader Auto-Tuning Experiment")
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()

    fp = collect_fingerprint()
    target_device = "cuda" if fp.effective_backend == "cuda" else "cpu"
    steps = args.steps if target_device == "cuda" else min(args.steps, 6)
    is_cuda = target_device == "cuda"

    unit_runner, unit_rc, unit_out = _run_test_file("tests/unit/test_pass_dataloader.py")
    mig_runner, mig_rc, mig_out = _run_test_file("tests/migrate/test_phase3_2_dataloader.py")

    def _dl_overrides(workers: int) -> list[str]:
        ov = [
            "dataloader.auto_tune=true",
            f"dataloader.num_workers={workers}",
            f"dataloader.persistent_workers={'true' if workers > 0 else 'false'}",
            f"dataloader.prefetch_factor={'2' if workers > 0 else 'null'}",
        ]
        if is_cuda:
            ov.append("dataloader.pin_memory=true")
        return ov

    variants = [
        ("baseline_w0", True, []),
        ("3.2_dl_workers_2", False, _dl_overrides(2)),
        ("3.2_dl_workers_3", False, _dl_overrides(3)),
        ("3.2_dl_workers_4", False, _dl_overrides(4)),
    ]

    targets = [
        ("cnn_vision", "cnn_vision.py", 32, []),
        ("edge_unpicklable", "edge_cases.py", 16, ["--case", "unpicklable_dataset"]),
    ]

    rows: list[list[object]] = []
    fb_lines: list[str] = []
    any_fail = False

    for wl_name, script_name, bs, extra in targets:
        base_rec: RunRecord | None = None
        wl_steps = min(steps, 12) if wl_name == "edge_unpicklable" else steps
        for var_name, is_base, ov in variants:
            rec = _run_case(
                wl_name=wl_name,
                script_name=script_name,
                steps=wl_steps,
                batch_size=bs,
                device=target_device,
                extra=extra,
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
                rows.append([wl_name, var_name, base_rec.step_stats.median, rec.step_stats.median, "0.00x", "N/A", "FAIL"])
                fb_lines.append(f"[ERROR] {wl_name} ({var_name}): {rec.error or base_rec.error}")
            else:
                parity = check_parity(base_rec.losses, rec.losses, profile="safe")
                speedup = compute_speedup(base_rec.step_stats.median, rec.step_stats.median)
                vram_d = compute_vram_delta(base_rec.peak_allocated_mb, rec.peak_allocated_mb)
                status = "PASS" if parity.passed else "FAIL"
                if not parity.passed:
                    any_fail = True
                rows.append([
                    wl_name,
                    var_name,
                    f"{base_rec.step_stats.median:.2f}",
                    f"{rec.step_stats.median:.2f}",
                    f"{rec.step_stats.p95:.2f}",
                    f"{speedup:.2f}x",
                    f"{base_rec.end_to_end_sec:.2f}s/{rec.end_to_end_sec:.2f}s",
                    f"{base_rec.peak_allocated_mb:.1f}/{rec.peak_allocated_mb:.1f} ({vram_d:+.1f})",
                    parity.max_rel_diff,
                    status,
                ])
                for ev in rec.fallback_events:
                    fb_lines.append(f"[{ev.get('status')}] {wl_name} ({var_name}) -> {ev.get('pass_name')}: {ev.get('reason')}")

    overall_rc = 0 if (unit_rc == 0 and mig_rc == 0 and not any_fail) else 1
    builder = TextReportBuilder(
        phase_title="PHASE 3.2 EXPERIMENT REPORT: DATALOADER AUTO-TUNING & SAFETY GUARDS",
        repo_root=REPO_ROOT,
    )
    builder.add_header({
        "Google Colab": fp.is_colab,
        "Backend / GPU": (
            f"{fp.effective_backend} / {fp.primary_gpu.name} ({fp.primary_gpu.sm_name})"
            if fp.primary_gpu
            else fp.effective_backend
        ),
        "CPU Cores / /dev/shm": f"{fp.host.physical_cores}/{fp.host.logical_cores} cores | {fp.host.shm_total_mb} MB",
        "Steps (cnn/edge)": f"{steps} / {min(steps, 12)}",
        "Unit Tests": f"{unit_runner} (exit_code={unit_rc})",
        "Migrate Tests": f"{mig_runner} (exit_code={mig_rc})",
        "Overall Experiment": "PASS" if overall_rc == 0 else "FAIL",
    })
    builder.add_table(
        "DATALOADER WORKER SWEEP (Baseline w=0 vs w=2,3,4)",
        ["Workload", "Variant", "Base ms", "Wrap ms", "Wrap P95", "Speedup", "E2E (B/W)", "VRAM MB (B/W/d)", "MaxRelDiff", "Status"],
        rows,
    )
    builder.add_lines_section("DATALOADER TUNING & SAFETY GUARD LOGS", fb_lines)
    builder.add_lines_section(f"MIGRATE TEST OUTPUT ({mig_runner})", mig_out.splitlines())
    builder.add_lines_section(f"UNIT TEST OUTPUT ({unit_runner})", unit_out.splitlines())
    builder.add_verification_footer(
        status="PENDING_USER_REVIEW" if overall_rc == 0 else "FAILED_NEEDS_ATTENTION",
        note="Verify DataLoader worker sweep & unpicklable guard on Google Colab before Phase 3.3",
    )

    out_path = builder.write("phase-3.2", report_dir=args.report_dir)
    print(builder.render())
    print(f"[SAVED] Text report written to: {out_path}")
    return overall_rc


if __name__ == "__main__":
    raise SystemExit(main())
