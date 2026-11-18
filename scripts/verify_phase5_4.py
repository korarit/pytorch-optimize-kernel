#!/usr/bin/env python3
"""Phase 5.4 Verification (Phase 3-Grade Deep Debug): CUDA Graphs for Static-Shape Training Loops."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

BENCH_DIR = Path(__file__).resolve().parent.parent / "pipline-for-test"
REPO_ROOT = BENCH_DIR.parent
for _p in (str(REPO_ROOT), str(BENCH_DIR), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from phase5_report_helper import (
    extract_allocator_row,
    extract_phase_breakdown_row,
    extract_step0_and_p99,
    format_step_trace_block,
    format_torch_profiler_block,
    run_workload_variant,
)
from src.common.text_report import TextReportBuilder
from src.config import detect_runtime_cache_key
from src.detect import collect_fingerprint
from telemetry import check_parity, compute_speedup, compute_vram_delta


def _run_unit_file(rel_path: str) -> tuple[bool, float, str, str]:
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / rel_path), "-v"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    dt = time.perf_counter() - t0
    out = (proc.stdout + "\n" + proc.stderr).strip()
    ran = next((ln.strip() for ln in reversed(out.splitlines()) if ln.strip().startswith("Ran ")), f"exit={proc.returncode}")
    return proc.returncode == 0, dt, ran, out


def main() -> int:
    fp = collect_fingerprint()
    device = "cuda" if fp.effective_backend == "cuda" else "cpu"
    is_cuda = device == "cuda"
    sm_int = fp.sm_int
    cache_key = detect_runtime_cache_key().cache_key

    # 1. Unit & Migrate tests
    ok_cmp, dt_cmp, ran_cmp, out_cmp = _run_unit_file("tests/unit/test_pass_compile.py")
    ok_mig, dt_mig, ran_mig, out_mig = _run_unit_file("tests/migrate/test_phase5_4_train_cudagraphs.py")
    all_ok = ok_cmp and ok_mig

    # 2. Launch-bound static training workloads: deep_mlp, cnn_vision:small, transformer_lm:small
    if is_cuda:
        cases: list[tuple[str, str, int, list[str], int]] = [
            ("deep_mlp", "deep_mlp", 64, ["--width", "256", "--depth", "24"], 30),
            ("cnn_vision_small", "cnn_vision", 32, ["--image-size", "64", "--width", "64"], 25),
            (
                "transformer_lm_small",
                "transformer_lm",
                16,
                ["--vocab-size", "2048", "--d-model", "256", "--n-heads", "4", "--n-layers", "4", "--seq-len", "128"],
                25,
            ),
        ]
    else:
        cases = [
            ("deep_mlp_cpu", "deep_mlp", 16, ["--width", "64", "--depth", "6"], 6),
            ("cnn_vision_cpu", "cnn_vision", 8, ["--image-size", "32", "--width", "32"], 6),
            (
                "transformer_lm_cpu",
                "transformer_lm",
                4,
                ["--vocab-size", "512", "--d-model", "64", "--n-heads", "2", "--n-layers", "2", "--seq-len", "32"],
                5,
            ),
        ]

    ab_rows: list[list[object]] = []
    phase_rows: list[list[object]] = []
    alloc_rows: list[list[object]] = []
    prof_lines: list[str] = []
    step_traces: list[str] = []
    diag_lines: list[str] = []
    audit_lines: list[str] = []

    with tempfile.TemporaryDirectory() as tmp_cache:
        for case_lbl, wl_name, bs, extra, steps in cases:
            case_cache = Path(tmp_cache) / case_lbl
            case_cache.mkdir(parents=True, exist_ok=True)

            base_rec, base_sum, _ = run_workload_variant(
                case_label=case_lbl,
                wl_name=wl_name,
                steps=steps,
                device=device,
                profile="safe",
                variant_name="baseline",
                extra_args=extra,
                batch_size=bs,
                is_baseline=True,
            )
            step_traces.extend(format_step_trace_block(case_lbl, "baseline", base_sum, None))

            variants: list[tuple[str, str, list[str], Path | None]] = [
                ("compile_default", "balanced", ["compile.mode=default", "compile.train_cuda_graphs=false"], case_cache),
                ("train_cuda_graphs", "balanced", ["compile.mode=default", "compile.train_cuda_graphs=true"], case_cache),
            ]
            comp_med_ref: float | None = None
            for v_name, prof, ovr, v_cache in variants:
                rec, summ, tel = run_workload_variant(
                    case_label=case_lbl,
                    wl_name=wl_name,
                    steps=steps,
                    device=device,
                    profile=prof,
                    variant_name=v_name,
                    extra_args=extra,
                    batch_size=bs,
                    overrides=ovr,
                    cache_dir=v_cache,
                )
                if base_rec.exit_code != 0 or rec.exit_code != 0:
                    all_ok = False
                    ab_rows.append([case_lbl, v_name, "-", f"{base_rec.step_stats.median:.2f}", "-", "-", "-", "-", "0.00x", "-", "N/A", "N/A", "FAIL"])
                    audit_lines.append(f"[ERROR] {case_lbl} ({v_name}): {rec.error or base_rec.error}")
                    continue

                step0_w, p99_w = extract_step0_and_p99(summ, tel)
                parity = check_parity(base_rec.losses, rec.losses, profile=prof)
                sp_base = compute_speedup(base_rec.step_stats.median, rec.step_stats.median)
                if v_name == "compile_default":
                    comp_med_ref = rec.step_stats.median
                    vs_comp_str = "1.00x"
                else:
                    sp_comp = compute_speedup(comp_med_ref or rec.step_stats.median, rec.step_stats.median)
                    vs_comp_str = f"{sp_comp:.2f}x"
                vd = compute_vram_delta(base_rec.peak_allocated_mb, rec.peak_allocated_mb)
                if not parity.passed:
                    all_ok = False
                ab_rows.append([
                    case_lbl,
                    v_name,
                    f"{step0_w:.1f}",
                    f"{base_rec.step_stats.median:.2f}",
                    f"{rec.step_stats.median:.2f}",
                    f"{rec.step_stats.p95:.2f}",
                    f"{p99_w:.2f}",
                    f"{base_rec.end_to_end_sec:.2f}/{rec.end_to_end_sec:.2f}s",
                    f"{sp_base:.2f}x",
                    vs_comp_str,
                    f"{base_rec.peak_allocated_mb:.1f}/{rec.peak_allocated_mb:.1f} ({vd:+.1f})",
                    parity.max_rel_diff,
                    "PASS" if parity.passed else "FAIL",
                ])
                step_traces.extend(format_step_trace_block(case_lbl, v_name, summ, tel))
                for ev in rec.fallback_events:
                    audit_lines.append(f"[{ev.get('status')}] {case_lbl} ({v_name}) -> {ev.get('pass_name')}: {ev.get('reason')}")
                diag_lines.append(
                    f"- {case_lbl} ({v_name}): med={rec.step_stats.median:.2f}ms ({sp_base:.2f}x vs base, {vs_comp_str} vs compile=default), "
                    f"VRAM {base_rec.peak_allocated_mb:.1f} -> {rec.peak_allocated_mb:.1f} MB ({vd:+.1f} MB)"
                )

            # Deep Debug + PyTorch Profiler pass (`baseline_dbg` vs `compile_default_dbg` vs `train_cg_dbg`)
            dbg_specs = [
                ("baseline_dbg", "safe", True, [], None),
                ("compile_default_dbg", "balanced", False, ["compile.mode=default", "compile.train_cuda_graphs=false"], case_cache),
                ("train_cg_dbg", "balanced", False, ["compile.mode=default", "compile.train_cuda_graphs=true"], case_cache),
            ]
            dbg_steps = 10 if is_cuda else 4
            prof_start = 4 if is_cuda else 1
            for d_name, prof, dis_p, ovr, d_cache in dbg_specs:
                rec_d, _, tel_d = run_workload_variant(
                    case_label=case_lbl,
                    wl_name=wl_name,
                    steps=dbg_steps,
                    device=device,
                    profile=prof,
                    variant_name=d_name,
                    extra_args=extra,
                    batch_size=bs,
                    disable_builtin_passes=dis_p,
                    overrides=ovr,
                    cache_dir=d_cache,
                    debug_profile=True,
                    torch_profile=True,
                    prof_start=prof_start,
                    prof_steps=3 if is_cuda else 2,
                )
                if rec_d.exit_code != 0 or not tel_d:
                    audit_lines.append(f"[ERROR] dbg {case_lbl} ({d_name}): {rec_d.error}")
                    continue
                pb_row = extract_phase_breakdown_row(case_lbl, d_name, tel_d)
                if pb_row:
                    phase_rows.append(pb_row)
                mem_row = extract_allocator_row(case_lbl, d_name, tel_d)
                if mem_row:
                    alloc_rows.append(mem_row)
                prof_lines.extend(format_torch_profiler_block(case_lbl, d_name, tel_d))

    gpu_name = fp.primary_gpu.name if fp.primary_gpu else "CPU-only"
    report = TextReportBuilder("PHASE 5.4 REPORT: TRAINING CUDA GRAPHS (DEEP DEBUG)", REPO_ROOT)
    report.add_header({
        "Google Colab": fp.is_colab,
        "GPU / Backend": f"{gpu_name} | {device} (sm_{sm_int})",
        "Python / PyTorch": f"{fp.host.python_version} | {fp.host.torch_version}",
        "CUDA / Triton": f"{fp.host.cuda_version or 'N/A'} | {fp.host.triton_version or 'N/A'}",
        "Runtime Cache Key": cache_key,
        "Unit (compile)": f"{ran_cmp} ({dt_cmp:.2f}s) -> {'PASS' if ok_cmp else 'FAIL'}",
        "Migrate (phase5.4)": f"{ran_mig} ({dt_mig:.2f}s) -> {'PASS' if ok_mig else 'FAIL'}",
        "Overall Phase 5.4": "PASS" if all_ok else "FAIL",
    })
    report.add_table(
        "TRAINING CUDA GRAPHS ABLATION (BASELINE VS COMPILE=DEFAULT VS TRAIN_CUDA_GRAPHS)",
        ["Case", "Variant", "Step0 ms", "Base ms", "Wrap ms", "Wrap P95", "Wrap P99", "E2E s (B/W)", "vs Base", "vs Comp", "VRAM MB (B/W/d)", "MaxRelDiff", "Status"],
        ab_rows,
    )
    report.add_table(
        "DEEP DEBUG: STEP PHASE BREAKDOWN (STEADY MEAN MS & SHARE %)",
        ["Case", "Mode", "Step0 ms", "Steady ms", "Data/Other", "Forward", "Loss/Clip", "Backward", "Optim"],
        phase_rows,
    )
    if alloc_rows:
        report.add_table(
            "DEEP DEBUG: CUDA ALLOCATOR & VRAM HEADROOM",
            ["Case", "Mode", "Alloc MB", "Resv MB", "Ovhead MB", "Headroom", "Util %", "Retries"],
            alloc_rows,
        )
    report.add_lines_section(
        "TOP PYTORCH PROFILER OPS (CHECK self_cpu_ms & cudaGraphLaunch)",
        prof_lines if prof_lines else ["(no profiler ops captured)"],
    )
    report.add_lines_section("WARMUP VS STEADY STEP TRACE", step_traces)
    report.add_lines_section("AUTOMATED DIAGNOSIS", diag_lines)
    report.add_lines_section("KERNEL & FALLBACK AUDIT", audit_lines)
    if not all_ok:
        err_lines: list[str] = []
        if not ok_cmp:
            err_lines.extend(out_cmp[-1500:].splitlines())
        if not ok_mig:
            err_lines.extend(out_mig[-1500:].splitlines())
        if err_lines:
            report.add_lines_section("UNIT / MIGRATE FAILURE TRACE", err_lines)

    report.add_verification_footer(
        status="PENDING_USER_REVIEW" if all_ok else "NEEDS_FIX",
        note="Verify Training CUDA Graphs speedup (15%-40% vs compile=default on deep_mlp/transformer_lm_small) on Colab L4 before Plan 5.5",
    )

    out_path = report.write(name="phase-5.4")
    print(report.render())
    print(f"Report written to: {out_path}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
