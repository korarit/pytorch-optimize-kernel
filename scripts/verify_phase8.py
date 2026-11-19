#!/usr/bin/env python3
"""Phase 8 Verification Suite + 7-Part Deep Debug Detailed Text Report generator."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
BENCH_DIR = REPO_ROOT / "pipline-for-test"
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

from phase5_report_helper import (  # noqa: E402
    extract_allocator_row,
    extract_phase_breakdown_row,
    extract_step0_and_p99,
    format_step_trace_block,
    format_torch_profiler_block,
    run_workload_variant,
)
from ptopt.common.text_report import TextReportBuilder  # noqa: E402
from ptopt.config import ConfigResolver, detect_runtime_cache_key  # noqa: E402
from ptopt.detect import collect_fingerprint  # noqa: E402
from telemetry import check_parity, compute_speedup, compute_vram_delta  # noqa: E402


def _run_unit_file(rel_path: str) -> tuple[bool, float, str, str]:
    t0 = time.perf_counter()
    env = os.environ.copy()
    old_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{old_pp}" if old_pp else str(REPO_ROOT)
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / rel_path), "-v"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode == 0, time.perf_counter() - t0, proc.stdout, proc.stderr


def main() -> int:
    fp = collect_fingerprint()
    has_cuda = bool(torch.cuda.is_available())
    cache_key = detect_runtime_cache_key(torch).cache_key
    resolver = ConfigResolver()
    bal_cfg = resolver.resolve(fp, profile="balanced", env={})
    boost_cfg = resolver.resolve(fp, profile="balanced", overrides={"runtime.bandwidth_boost": True}, env={})
    agg_cfg = resolver.resolve(fp, profile="aggressive", overrides={"runtime.bandwidth_boost": True}, env={})

    # Part 1/2: Unit & Functional Verification
    unit_ok, unit_dt, unit_out, unit_err = _run_unit_file("tests/unit/test_phase8_bandwidth_booster.py")
    cfg_ok, cfg_dt, cfg_out, cfg_err = _run_unit_file("tests/unit/test_config_resolver.py")

    audit_rows = [
        [
            "8.1",
            "Adaptive multi-batch GPU prefetch + Event sync + back-off",
            "Unit",
            "PASS" if unit_ok else "FAIL",
            f"Ran test_phase8_bandwidth_booster.py in {unit_dt:.2f}s",
        ],
        [
            "8.2",
            "Opt-in vram_bandwidth_boost rule (--bandwidth-boost, default off)",
            "Config",
            "PASS" if cfg_ok else "FAIL",
            f"default={bal_cfg.matched_rules} | boost={boost_cfg.matched_rules} (dt={cfg_dt:.2f}s)",
        ],
        [
            "8.3",
            "Inductor coordinate_descent_tuning opt-in VRAM gate",
            "Unit",
            "PASS" if unit_ok else "FAIL",
            "Tested default skip vs boost max-autotune apply & >=75% skip",
        ],
        [
            "8.4",
            "FP8 dual-resident vs boost/auto-drop & M>=128 prewarm",
            "Unit",
            "PASS" if unit_ok else "FAIL",
            "Tested dual_resident_default vs boost_auto_drop (>=85%) & auto (>=80%)",
        ],
        [
            "8.5",
            "bandwidth_vram_booster telemetry & real_bench table",
            "Unit",
            "PASS" if unit_ok else "FAIL",
            "TelemetryCollector + real_bench summary verified",
        ],
    ]

    # Part 2-6: GPU Deep Debug Ablation (or CPU smoke + WAIT FOR USER on CPU CI)
    ablation_rows: list[list[object]] = []
    phase_rows: list[list[object]] = []
    alloc_rows: list[list[object]] = []
    prof_blocks: list[str] = []
    step_traces: list[str] = []
    diag_lines: list[str] = []

    if has_cuda:
        with tempfile.TemporaryDirectory() as cache_root:
            specs = [
                (
                    "cnn_vision_224",
                    "cnn_vision",
                    12,
                    32,
                    ["--image-size", "224", "--width", "64"],
                    "baseline",
                    True,
                    (),
                    {},
                ),
                (
                    "cnn_vision_224",
                    "cnn_vision",
                    12,
                    32,
                    ["--image-size", "224", "--width", "64"],
                    "balanced_default",
                    False,
                    (),
                    {},
                ),
                (
                    "cnn_vision_224",
                    "cnn_vision",
                    12,
                    32,
                    ["--image-size", "224", "--width", "64"],
                    "balanced_phase8_boost",
                    False,
                    ("runtime.bandwidth_boost=true",),
                    {},
                ),
                (
                    "transformer_lm_med",
                    "transformer_lm",
                    12,
                    16,
                    ["--d-model", "768", "--n-heads", "12", "--n-layers", "6", "--seq-len", "256"],
                    "baseline",
                    True,
                    (),
                    {},
                ),
                (
                    "transformer_lm_med",
                    "transformer_lm",
                    12,
                    16,
                    ["--d-model", "768", "--n-heads", "12", "--n-layers", "6", "--seq-len", "256"],
                    "aggressive_phase8_bw",
                    False,
                    ("runtime.bandwidth_boost=true",),
                    {},
                ),
            ]
            baselines: dict[str, Any] = {}
            for case_lbl, wl_name, steps, bs, extra, var_lbl, is_base, ovr, xenv in specs:
                var_profile = "aggressive" if "aggressive" in var_lbl else "balanced"
                rec, summary, tel = run_workload_variant(
                    case_label=case_lbl,
                    wl_name=wl_name,
                    steps=steps,
                    device="cuda",
                    profile=var_profile,
                    variant_name=var_lbl,
                    extra_args=extra,
                    batch_size=bs,
                    is_baseline=is_base,
                    overrides=ovr,
                    cache_dir=Path(cache_root) / case_lbl / var_lbl,
                    debug_profile=not is_base,
                    torch_profile=not is_base,
                    prof_start=2,
                    prof_steps=4,
                    extra_env=xenv,
                )
                if is_base:
                    baselines[case_lbl] = rec
                base_rec = baselines.get(case_lbl, rec)
                step0, p99 = extract_step0_and_p99(summary, tel)
                spd = 1.0 if is_base else compute_speedup(base_rec.step_stats.median, rec.step_stats.median)
                parity_ok = True if is_base else check_parity(base_rec.losses, rec.losses, profile=var_profile).passed
                status = "PASS" if (rec.exit_code == 0 and parity_ok) else "FAIL"
                vram_d = compute_vram_delta(base_rec.peak_allocated_mb, rec.peak_allocated_mb)
                ablation_rows.append([
                    case_lbl,
                    var_lbl,
                    f"{step0:.1f}",
                    f"{rec.step_stats.median:.2f}",
                    f"{rec.step_stats.p95:.2f}",
                    f"{p99:.2f}",
                    f"{spd:.2f}x",
                    f"{rec.end_to_end_sec:.2f}s",
                    f"{rec.peak_allocated_mb:.1f} MB ({vram_d:+.1f})",
                    status,
                ])
                step_traces.extend(format_step_trace_block(case_lbl, var_lbl, summary, tel))
                if tel is not None:
                    pb_row = extract_phase_breakdown_row(case_lbl, var_lbl, tel)
                    if pb_row is not None:
                        phase_rows.append(pb_row)
                    al_row = extract_allocator_row(case_lbl, var_lbl, tel)
                    if al_row is not None:
                        alloc_rows.append(al_row)
                    prof_blocks.extend(format_torch_profiler_block(case_lbl, var_lbl, tel))
                    bw = (tel.get("debug_profile") or {}).get("bandwidth_vram_booster")
                    if bw:
                        diag_lines.append(f"[BW_BOOSTER] {case_lbl} ({var_lbl}): {bw}")
    else:
        ablation_rows.append([
            "gpu_ablation",
            "cuda_required",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "WAIT FOR USER (Colab GPU)",
        ])
        diag_lines.append("CUDA unavailable locally; run `python3 scripts/verify_phase8.py` on Colab L4/A100 for GPU ablation.")

    all_ok = unit_ok and cfg_ok and all(str(r[-1]).startswith("PASS") or str(r[-1]).startswith("WAIT") for r in ablation_rows)
    builder = TextReportBuilder("Phase 8: VRAM-Headroom Bandwidth Booster & Detailed Telemetry", REPO_ROOT)
    builder.add_header({
        "Git Commit": cache_key,
        "Backend / GPU": (
            f"{fp.effective_backend} / {fp.primary_gpu.name} ({fp.primary_gpu.sm_name}, {fp.vram_mb:.1f} MB)"
            if fp.primary_gpu
            else fp.effective_backend
        ),
        "PyTorch / Triton": f"{fp.host.torch_version} / {fp.host.triton_version or 'N/A'}",
        "Balanced Default Rules": ", ".join(bal_cfg.matched_rules),
        "Balanced Boost Rules": ", ".join(boost_cfg.matched_rules),
        "Aggressive Boost Rules": ", ".join(agg_cfg.matched_rules),
    })
    builder.add_table("PHASE 8 AUDIT & FUNCTIONAL MATRIX", ["ID", "Workstream", "Scope", "Status", "Notes"], audit_rows)
    builder.add_table(
        "ABLATION MATRIX + P95 / P99 / E2E LATENCY SUMMARY",
        ["Case", "Mode", "Step0(ms)", "SteadyP50", "P95(ms)", "P99(ms)", "Speedup", "E2E", "PeakVRAM", "Status"],
        ablation_rows,
    )
    if phase_rows:
        builder.add_table(
            "PHASE BREAKDOWN (DEBUG PROFILE)",
            ["Case", "Mode", "Step0(ms)", "Steady(ms)", "Data/Other", "Forward", "Loss/Clip", "Backward", "Optim"],
            phase_rows,
        )
    if alloc_rows:
        builder.add_table(
            "CUDA ALLOCATOR & MEMORY HEADROOM",
            ["Case", "Mode", "Alloc(MB)", "Reserved(MB)", "Overhead(MB)", "Headroom(MB)", "Util(%)", "Retries"],
            alloc_rows,
        )
    if prof_blocks:
        builder.add_lines_section("TOP PYTORCH PROFILER OPS (STEADY WINDOW)", prof_blocks)
    if step_traces:
        builder.add_lines_section("WARMUP VS STEADY STEP TRACE", step_traces)

    if not unit_ok:
        diag_lines.append(f"[UNIT FAIL] test_phase8_bandwidth_booster.py:\n{(unit_err or unit_out)[-1200:]}")
    if not cfg_ok:
        diag_lines.append(f"[UNIT FAIL] test_config_resolver.py:\n{(cfg_err or cfg_out)[-1200:]}")
    builder.add_lines_section("AUTOMATED DIAGNOSIS & BANDWIDTH BOOSTER TELEMETRY", diag_lines)
    builder.add_verification_footer(
        status="PENDING_USER_REVIEW" if all_ok else "FAILED_NEEDS_ATTENTION",
        note="Run on Google Colab L4/A100 GPU & inspect bandwidth_vram_booster logs before closing Phase 8",
    )

    out_path = builder.write(name="phase-8")
    text = out_path.read_text(encoding="utf-8")
    print(text)
    print(f"[SAVED] Phase 8 report written to: {out_path}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
