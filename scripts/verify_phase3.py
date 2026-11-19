#!/usr/bin/env python3
"""Full Phase 3 Verification Matrix: Baseline vs safe vs balanced vs aggressive + all tests."""

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


def _run_suite(test_dir: str) -> tuple[str, int, str]:
    if importlib.util.find_spec("pytest") is not None:
        cmd = [sys.executable, "-m", "pytest", test_dir, "-q"]
        runner = "pytest"
    else:
        cmd = [sys.executable, "-m", "unittest", "discover", "-s", test_dir, "-v"]
        runner = "unittest"
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    return runner, res.returncode, (res.stdout + "\n" + res.stderr).strip()


def _run_profile_case(
    *,
    wl_name: str,
    steps: int,
    device: str,
    profile: str,
    is_baseline: bool = False,
    extra_args_override: list[str] | None = None,
    batch_size_override: int | None = None,
    cache_dir: Path | None = None,
) -> RunRecord:
    spec = WORKLOAD_SPECS[wl_name]
    script_path = WORKLOADS_DIR / spec["script"]
    bs = batch_size_override if batch_size_override is not None else int(spec["batch_size"])
    extra = extra_args_override if extra_args_override is not None else list(spec["extra"])
    wl_args = ["--steps", str(steps), "--batch-size", str(bs), "--seed", "42", "--device", device, *extra]
    env = os.environ.copy()

    with tempfile.TemporaryDirectory() as tmp:
        tel_path = Path(tmp) / "tel.json"
        if is_baseline:
            cmd = [sys.executable, str(script_path), *wl_args]
        else:
            env["PTOPT_TELEMETRY_OUT"] = str(tel_path)
            if device == "cpu":
                env["PTOPT_COMPILE_CPU"] = "1"
            ovr_flags: list[str] = []
            if cache_dir is not None:
                ovr_flags.extend(["--override", f"compile.cache_dir={cache_dir}"])
            if profile == "aggressive" and device == "cuda":
                ovr_flags.extend(["--override", "compile.mode=default"])
            cmd = [
                sys.executable,
                "-m",
                "ptopt.cli.main",
                "--profile",
                profile,
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
        err = (res.stderr.strip() or res.stdout.strip())[-350:] if res.returncode != 0 else None
        return merge_summary_and_telemetry(
            workload=wl_name,
            mode="baseline" if is_baseline else profile,
            cache_state="warm" if cache_dir else "none",
            exit_code=res.returncode,
            summary=summary,
            telemetry=tel_data,
            wall_sec=wall,
            error=err,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3 Full Consolidation Verification Matrix")
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--skip-suites", action="store_true")
    args = parser.parse_args()

    fp = collect_fingerprint()
    target_device = "cuda" if fp.effective_backend == "cuda" else "cpu"
    is_cuda = target_device == "cuda"

    if args.skip_suites:
        unit_runner, unit_rc, unit_out = "skipped", 0, "(skipped via --skip-suites)"
        mig_runner, mig_rc, mig_out = "skipped", 0, "(skipped via --skip-suites)"
    else:
        unit_runner, unit_rc, unit_out = _run_suite("tests/unit")
        mig_runner, mig_rc, mig_out = _run_suite("tests/migrate")

    # Scaled representative matrix matching Phase 3 findings
    workloads: list[tuple[str, int, int | None, list[str] | None]] = (
        [
            ("transformer_lm", args.steps, 16, ["--d-model", "512", "--n-heads", "8", "--n-layers", "6", "--seq-len", "256"]),
            ("cnn_vision", args.steps, 32, ["--image-size", "128", "--width", "96"]),
            ("deep_mlp", args.steps, 128, ["--width", "1024", "--depth", "24"]),
            ("infer_runner_bs1", max(args.steps, 25), 1, []),
            ("infer_runner_bs32", args.steps, 32, []),
            ("edge_cases", min(args.steps, 10), 8, ["--case", "all"]),
        ]
        if is_cuda
        else [
            ("transformer_lm", 4, None, None),
            ("cnn_vision", 4, None, None),
            ("deep_mlp", 4, None, None),
            ("infer_runner_bs1", 6, 1, None),
            ("infer_runner_bs32", 6, 32, None),
            ("edge_cases", 4, None, None),
        ]
    )

    rows: list[list[object]] = []
    fb_lines: list[str] = []
    any_fail = False

    with tempfile.TemporaryDirectory() as cache_root:
        for wl, steps, bs_ovr, extra_ovr in workloads:
            wl_cache = Path(cache_root) / wl
            wl_cache.mkdir(parents=True, exist_ok=True)
            base_rec = _run_profile_case(
                wl_name=wl,
                steps=steps,
                device=target_device,
                profile="safe",
                is_baseline=True,
                extra_args_override=extra_ovr,
                batch_size_override=bs_ovr,
            )
            for prof in ("supersafe", "safe", "balanced", "aggressive"):
                rec = _run_profile_case(
                    wl_name=wl,
                    steps=steps,
                    device=target_device,
                    profile=prof,
                    extra_args_override=extra_ovr,
                    batch_size_override=bs_ovr,
                    cache_dir=wl_cache,
                )
                if base_rec.exit_code != 0 or rec.exit_code != 0:
                    any_fail = True
                    rows.append([wl, prof, f"{base_rec.step_stats.median:.2f}", "-", "-", "0.00x", "N/A", "None", "FAIL"])
                    fb_lines.append(f"[ERROR] {wl} ({prof}): {rec.error or base_rec.error}")
                else:
                    parity = check_parity(base_rec.losses, rec.losses, profile=prof)
                    speedup = compute_speedup(base_rec.step_stats.median, rec.step_stats.median)
                    vram_d = compute_vram_delta(base_rec.peak_allocated_mb, rec.peak_allocated_mb)
                    status = "PASS" if parity.passed else "FAIL"
                    if not parity.passed:
                        any_fail = True
                    rows.append([
                        wl,
                        prof,
                        f"{base_rec.step_stats.median:.2f}",
                        f"{rec.step_stats.median:.2f}",
                        f"{rec.step_stats.p95:.2f}",
                        f"{speedup:.2f}x",
                        f"{base_rec.peak_allocated_mb:.1f}/{rec.peak_allocated_mb:.1f} ({vram_d:+.1f})",
                        parity.max_rel_diff,
                        status,
                    ])
                    for ev in rec.fallback_events:
                        if ev.get("status") != "APPLIED":
                            fb_lines.append(
                                f"[{ev.get('status')}] {wl} ({prof}) -> {ev.get('pass_name')}: {ev.get('reason')}"
                            )

    overall_rc = 0 if (unit_rc == 0 and mig_rc == 0 and not any_fail) else 1
    builder = TextReportBuilder(
        phase_title="PHASE 3 FULL CONSOLIDATION REPORT: BASELINE VS SAFE / BALANCED / AGGRESSIVE",
        repo_root=REPO_ROOT,
    )
    builder.add_header({
        "Google Colab": fp.is_colab,
        "Backend / GPU": (
            f"{fp.effective_backend} / {fp.primary_gpu.name} ({fp.primary_gpu.sm_name})"
            if fp.primary_gpu
            else fp.effective_backend
        ),
        "Unit Suite": f"{unit_runner} (exit_code={unit_rc})",
        "Migrate Suite": f"{mig_runner} (exit_code={mig_rc})",
        "Overall Phase 3": "PASS" if overall_rc == 0 else "FAIL",
    })
    builder.add_table(
        "PHASE 3 CONSOLIDATED PROFILE MATRIX",
        ["Workload", "Profile", "Base ms", "Wrap ms", "Wrap P95", "Speedup", "VRAM MB (B/W/d)", "MaxRelDiff", "Status"],
        rows,
    )
    builder.add_lines_section("SKIP & FALLBACK AUDIT SUMMARY", fb_lines[:60])
    builder.add_lines_section(f"UNIT SUITE ({unit_runner})", unit_out.splitlines()[-30:])
    builder.add_lines_section(f"MIGRATE SUITE ({mig_runner})", mig_out.splitlines()[-30:])
    builder.add_verification_footer(
        status="PENDING_USER_REVIEW" if overall_rc == 0 else "FAILED_NEEDS_ATTENTION",
        note="Inspect Phase 3 consolidated profile matrix on Google Colab before starting Phase 4",
    )

    out_path = builder.write("phase-3", report_dir=args.report_dir)
    print(builder.render())
    print(f"[SAVED] Text report written to: {out_path}")
    return overall_rc


if __name__ == "__main__":
    raise SystemExit(main())
