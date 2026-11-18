#!/usr/bin/env python3
"""Phase 5.1 Verification: Tiny dynamic-shape compile bypass vs large dynamic-shape compile."""

from __future__ import annotations

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

from bench import WORKLOAD_SPECS, WORKLOADS_DIR, _extract_workload_summary
from src.common.text_report import TextReportBuilder
from src.detect import collect_fingerprint
from telemetry import RunRecord, check_parity, compute_speedup, compute_vram_delta, merge_summary_and_telemetry


def _run_case(
    *,
    wl_name: str,
    steps: int,
    device: str,
    profile: str,
    is_baseline: bool,
    extra_args: list[str],
    batch_size: int,
    cache_dir: Path | None,
) -> RunRecord:
    script_path = WORKLOADS_DIR / WORKLOAD_SPECS[wl_name]["script"]
    wl_args = ["--steps", str(steps), "--batch-size", str(batch_size), "--seed", "42", "--device", device, *extra_args]
    env = os.environ.copy()

    with tempfile.TemporaryDirectory() as tmp:
        tel_path = Path(tmp) / "tel.json"
        if is_baseline:
            cmd = [sys.executable, str(script_path), *wl_args]
        else:
            env["PTOPT_TELEMETRY_OUT"] = str(tel_path)
            ovr: list[str] = []
            if cache_dir is not None:
                ovr.extend(["--override", f"compile.cache_dir={cache_dir}"])
            if profile == "aggressive" and device == "cuda":
                ovr.extend(["--override", "compile.mode=default"])
            cmd = [sys.executable, "-m", "src.cli.main", "--profile", profile, *ovr, "--", sys.executable, str(script_path), *wl_args]

        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=False)
        wall = time.perf_counter() - t0
        summary = _extract_workload_summary(proc.stdout)
        tel = json.loads(tel_path.read_text(encoding="utf-8")) if tel_path.is_file() else None
        err = (proc.stderr.strip() or proc.stdout.strip())[-300:] if proc.returncode != 0 else None
        return merge_summary_and_telemetry(
            workload=wl_name,
            mode="baseline" if is_baseline else profile,
            cache_state="warm" if cache_dir else "none",
            exit_code=proc.returncode,
            summary=summary,
            telemetry=tel,
            wall_sec=wall,
            error=err,
        )


def main() -> int:
    fp = collect_fingerprint()
    device = "cuda" if fp.effective_backend == "cuda" else "cpu"
    is_cuda = device == "cuda"

    # 1. Unit test (`test_pass_compile.py` executed by path to avoid site-packages/tests shadowing)
    t0 = time.perf_counter()
    unit_proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tests/unit/test_pass_compile.py"), "-v"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    unit_dt = time.perf_counter() - t0
    unit_out = (unit_proc.stdout + "\n" + unit_proc.stderr).strip()
    unit_ok = unit_proc.returncode == 0
    ran_line = next((ln.strip() for ln in reversed(unit_out.splitlines()) if ln.strip().startswith("Ran ")), f"exit={unit_proc.returncode}")

    # 2. Targeted before/after regression matrix on edge_cases + transformer_lm
    cases: list[tuple[str, str, int, int, list[str]]] = (
        [
            ("edge_dynamic_only", "edge_cases", 15, 8, ["--case", "dynamic_shape"]),
            ("edge_all", "edge_cases", 15, 8, ["--case", "all"]),
            ("transformer_lm", "transformer_lm", 15, 16, ["--d-model", "512", "--n-heads", "8", "--n-layers", "6", "--seq-len", "256"]),
        ]
        if is_cuda
        else [
            ("edge_dynamic_only", "edge_cases", 6, 8, ["--case", "dynamic_shape"]),
            ("edge_all", "edge_cases", 6, 8, ["--case", "all"]),
            ("transformer_lm", "transformer_lm", 4, 4, []),
        ]
    )

    rows: list[list[object]] = []
    event_notes: list[str] = []
    all_ok = unit_ok

    with tempfile.TemporaryDirectory() as cache_root:
        for label, wl, steps, bs, extra in cases:
            wl_cache = Path(cache_root) / label
            wl_cache.mkdir(parents=True, exist_ok=True)
            base_rec = _run_case(
                wl_name=wl,
                steps=steps,
                device=device,
                profile="safe",
                is_baseline=True,
                extra_args=extra,
                batch_size=bs,
                cache_dir=None,
            )
            for prof in ("safe", "balanced", "aggressive"):
                rec = _run_case(
                    wl_name=wl,
                    steps=steps,
                    device=device,
                    profile=prof,
                    is_baseline=False,
                    extra_args=extra,
                    batch_size=bs,
                    cache_dir=wl_cache,
                )
                if base_rec.exit_code != 0 or rec.exit_code != 0:
                    all_ok = False
                    rows.append([label, prof, f"{base_rec.step_stats.median:.2f}", "-", "0.00x", "N/A", "N/A", "FAIL"])
                    event_notes.append(f"[ERROR] {label} ({prof}): {rec.error or base_rec.error}")
                    continue

                parity = check_parity(base_rec.losses, rec.losses, profile=prof)
                speedup = compute_speedup(base_rec.step_stats.median, rec.step_stats.median)
                vram_d = compute_vram_delta(base_rec.peak_allocated_mb, rec.peak_allocated_mb)
                if not parity.passed:
                    all_ok = False
                rows.append([
                    label,
                    prof,
                    f"{base_rec.step_stats.median:.2f}",
                    f"{rec.step_stats.median:.2f}",
                    f"{speedup:.2f}x",
                    f"{base_rec.peak_allocated_mb:.1f}/{rec.peak_allocated_mb:.1f} ({vram_d:+.1f})",
                    parity.max_rel_diff,
                    "PASS" if parity.passed else "FAIL",
                ])
                for ev in rec.fallback_events:
                    if "compile.auto" in str(ev.get("pass_name", "")):
                        event_notes.append(
                            f"[{ev.get('status')}] {label} ({prof}) -> {ev.get('pass_name')}: {ev.get('reason')}"
                        )

    gpu_name = fp.primary_gpu.name if fp.primary_gpu else "CPU-only"
    report = TextReportBuilder("PHASE 5.1 REPORT: TINY DYNAMIC-SHAPE COMPILE BYPASS", REPO_ROOT)
    report.add_header({
        "Google Colab": fp.is_colab,
        "GPU / Backend": f"{gpu_name} | {device} (sm_{fp.sm_int})",
        "Unit Test (5.1)": f"{ran_line} ({unit_dt:.2f}s) -> {'PASS' if unit_ok else 'FAIL'}",
        "Overall Phase 5.1": "PASS" if all_ok else "FAIL",
    })
    report.add_table(
        "Dynamic-Shape Bypass & Regression Check",
        ["Case", "Profile", "Base ms", "Wrap ms", "Speedup", "VRAM MB (B/W/d)", "MaxRelDiff", "Status"],
        rows,
    )
    report.add_lines_section(
        "Compile Decision Audit (`compile.auto`)",
        event_notes if event_notes else ["(no compile.auto events recorded on this backend)"],
    )
    if not unit_ok:
        report.add_lines_section("Unit Failure Trace", unit_out[-2000:].splitlines())
    report.add_verification_footer(
        status="PENDING_USER_REVIEW" if all_ok else "NEEDS_FIX",
        note="Verify `edge_cases` speedup recovery on Google Colab GPU before advancing to Plan 5.2",
    )

    out_path = report.write(name="phase-5.1")
    print(report.render())
    print(f"Report written to: {out_path}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
