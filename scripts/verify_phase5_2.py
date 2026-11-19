#!/usr/bin/env python3
"""Phase 5.2 Verification (Phase 3-Grade Deep Debug): Fused Chunked Linear + CrossEntropy."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

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
from ptopt.common.text_report import TextReportBuilder
from ptopt.config import detect_runtime_cache_key
from ptopt.detect import collect_fingerprint
from ptopt.kernels import ref_linear_cross_entropy, triton_fused_linear_cross_entropy as fused_linear_cross_entropy
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


def _bench_linear_ce_micro(is_cuda: bool) -> list[list[object]]:
    """Micro-benchmark Peak VRAM & Step Time: Eager F.linear+CE vs FusedLinearCrossEntropy across vocab sizes."""
    device = torch.device("cuda" if is_cuda else "cpu")
    dtype = torch.bfloat16 if is_cuda else torch.float32
    specs = (
        [(1024, 512, 8192), (2048, 768, 32768), (4096, 1024, 65536)]
        if is_cuda
        else [(128, 64, 4096), (256, 128, 8192)]
    )
    rows: list[list[object]] = []
    iters = 10 if is_cuda else 4

    for n_rows, h_dim, vocab in specs:
        h0 = torch.randn(n_rows, h_dim, device=device, dtype=dtype, requires_grad=True)
        w0 = torch.randn(vocab, h_dim, device=device, dtype=dtype, requires_grad=True)
        tgt = torch.randint(0, vocab, (n_rows,), device=device)

        def _measure(use_fused: bool) -> tuple[float, float, float]:
            for _ in range(2):
                h_in = h0.detach().clone().requires_grad_(True)
                w_in = w0.detach().clone().requires_grad_(True)
                loss = (
                    fused_linear_cross_entropy(h_in, w_in, tgt)
                    if use_fused
                    else ref_linear_cross_entropy(h_in, w_in, None, tgt)
                )
                loss.backward()
            if is_cuda:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            last_loss = 0.0
            for _ in range(iters):
                h_in = h0.detach().clone().requires_grad_(True)
                w_in = w0.detach().clone().requires_grad_(True)
                loss = (
                    fused_linear_cross_entropy(h_in, w_in, tgt)
                    if use_fused
                    else ref_linear_cross_entropy(h_in, w_in, None, tgt)
                )
                loss.backward()
                last_loss = float(loss.detach().item())
            if is_cuda:
                torch.cuda.synchronize()
                peak_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            else:
                peak_mb = 0.0
            return ((time.perf_counter() - t0) * 1000.0) / iters, peak_mb, last_loss

        e_ms, e_vram, e_loss = _measure(False)
        f_ms, f_vram, f_loss = _measure(True)
        diff = abs(e_loss - f_loss)
        vram_delta = f_vram - e_vram
        vram_pct = ((e_vram - f_vram) / e_vram * 100.0) if e_vram > 0 else 0.0
        rows.append([
            f"{n_rows}x{h_dim} -> V={vocab}",
            f"{e_ms:.2f}",
            f"{f_ms:.2f}",
            f"{compute_speedup(e_ms, f_ms):.2f}x",
            f"{e_vram:.1f}",
            f"{f_vram:.1f}",
            f"{vram_delta:+.1f} ({vram_pct:.1f}% saved)",
            f"{diff:.2e}",
        ])
    return rows


def main() -> int:
    fp = collect_fingerprint()
    device = "cuda" if fp.effective_backend == "cuda" else "cpu"
    is_cuda = device == "cuda"
    cache_key = detect_runtime_cache_key().cache_key

    # 1. Unit tests
    ok_lce, dt_lce, ran_lce, out_lce = _run_unit_file("tests/unit/test_fused_linear_ce.py")
    ok_kp, dt_kp, ran_kp, out_kp = _run_unit_file("tests/unit/test_kernels_parity.py")
    all_ok = ok_lce and ok_kp

    # 2. Micro-benchmark scaling across large vocabularies
    micro_rows = _bench_linear_ce_micro(is_cuda)

    # 3. End-to-End large-vocab `transformer_lm` scaling (Baseline vs Compile vs FusedLinearCE eager)
    e2e_scales = (
        [
            ("vocab8k", 16, ["--vocab-size", "8192", "--d-model", "512", "--n-heads", "8", "--n-layers", "4", "--seq-len", "256"], 12),
            ("vocab32k", 16, ["--vocab-size", "32768", "--d-model", "512", "--n-heads", "8", "--n-layers", "4", "--seq-len", "256"], 12),
        ]
        if is_cuda
        else [
            ("vocab4k_cpu", 4, ["--vocab-size", "4096", "--d-model", "128", "--n-heads", "4", "--n-layers", "2", "--seq-len", "64"], 4),
        ]
    )

    e2e_rows: list[list[object]] = []
    phase_rows: list[list[object]] = []
    alloc_rows: list[list[object]] = []
    prof_lines: list[str] = []
    step_traces: list[str] = []
    audit_lines: list[str] = []
    diag_lines: list[str] = []

    with tempfile.TemporaryDirectory() as cache_root:
        for sc_label, bs, extra, steps in e2e_scales:
            sc_cache = Path(cache_root) / sc_label
            sc_cache.mkdir(parents=True, exist_ok=True)
            base_rec, base_sum, base_tel = run_workload_variant(
                case_label=sc_label,
                wl_name="transformer_lm",
                steps=steps,
                device=device,
                profile="safe",
                variant_name="baseline",
                extra_args=extra,
                batch_size=bs,
                is_baseline=True,
            )
            step_traces.extend(format_step_trace_block(sc_label, "baseline", base_sum, base_tel))

            variants = [
                ("balanced_compile", "balanced", [], sc_cache),
                (
                    "fused_lce_eager",
                    "aggressive",
                    ["compile.enabled=false", "kernels.enabled=true", "kernels.allow_list=[\"fused_linear_cross_entropy\"]"],
                    None,
                ),
            ]
            # Wait: allow_list override is CLI comma/list parsed via python, use PTOPT env for clean allow_list!
            for v_name, prof, ovr, v_cache in variants:
                extra_env = {}
                actual_ovr = list(ovr)
                if v_name == "fused_lce_eager":
                    actual_ovr = ["compile.enabled=false", "kernels.enabled=true"]
                    extra_env = {"PTOPT_FUSED_LCE_MIN_VOCAB": "4096"}

                rec, summ, tel = run_workload_variant(
                    case_label=sc_label,
                    wl_name="transformer_lm",
                    steps=steps,
                    device=device,
                    profile=prof,
                    variant_name=v_name,
                    extra_args=extra,
                    batch_size=bs,
                    overrides=actual_ovr,
                    cache_dir=v_cache,
                    extra_env=extra_env,
                )
                if base_rec.exit_code != 0 or rec.exit_code != 0:
                    all_ok = False
                    e2e_rows.append([sc_label, v_name, "-", f"{base_rec.step_stats.median:.2f}", "-", "-", "-", "0.00x", "N/A", "N/A", "FAIL"])
                    audit_lines.append(f"[ERROR] {sc_label} ({v_name}): {rec.error or base_rec.error}")
                    continue

                step0_w, p99_w = extract_step0_and_p99(summ, tel)
                parity = check_parity(base_rec.losses, rec.losses, profile=prof)
                sp = compute_speedup(base_rec.step_stats.median, rec.step_stats.median)
                vd = compute_vram_delta(base_rec.peak_allocated_mb, rec.peak_allocated_mb)
                if not parity.passed:
                    all_ok = False
                e2e_rows.append([
                    sc_label,
                    v_name,
                    f"{step0_w:.1f}",
                    f"{base_rec.step_stats.median:.2f}",
                    f"{rec.step_stats.median:.2f}",
                    f"{rec.step_stats.p95:.2f}",
                    f"{p99_w:.2f}",
                    f"{base_rec.end_to_end_sec:.2f}/{rec.end_to_end_sec:.2f}s",
                    f"{sp:.2f}x",
                    f"{base_rec.peak_allocated_mb:.1f}/{rec.peak_allocated_mb:.1f} ({vd:+.1f})",
                    parity.max_rel_diff,
                    "PASS" if parity.passed else "FAIL",
                ])
                step_traces.extend(format_step_trace_block(sc_label, v_name, summ, tel))
                for ev in rec.fallback_events:
                    audit_lines.append(f"[{ev.get('status')}] {sc_label} ({v_name}) -> {ev.get('pass_name')}: {ev.get('reason')}")
                diag_lines.append(
                    f"- {sc_label} ({v_name}): med={rec.step_stats.median:.2f}ms ({sp:.2f}x), VRAM {base_rec.peak_allocated_mb:.1f} -> {rec.peak_allocated_mb:.1f} MB ({vd:+.1f} MB)"
                )

            # Deep Debug pass on this scale
            dbg_specs = [
                ("baseline_dbg", "safe", True, [], None, {}),
                ("compile_dbg", "balanced", False, [], sc_cache, {}),
                ("fused_lce_dbg", "aggressive", False, ["compile.enabled=false", "kernels.enabled=true"], None, {"PTOPT_FUSED_LCE_MIN_VOCAB": "4096"}),
            ]
            dbg_steps = 7 if is_cuda else 4
            for d_name, prof, dis_p, ovr, d_cache, d_env in dbg_specs:
                rec_d, summ_d, tel_d = run_workload_variant(
                    case_label=sc_label,
                    wl_name="transformer_lm",
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
                    prof_start=2 if is_cuda else 1,
                    prof_steps=2,
                    extra_env=d_env,
                )
                if rec_d.exit_code != 0 or not tel_d:
                    audit_lines.append(f"[ERROR] dbg {sc_label} ({d_name}): {rec_d.error}")
                    continue
                pb_row = extract_phase_breakdown_row(sc_label, d_name, tel_d)
                if pb_row:
                    phase_rows.append(pb_row)
                mem_row = extract_allocator_row(sc_label, d_name, tel_d)
                if mem_row:
                    alloc_rows.append(mem_row)
                prof_lines.extend(format_torch_profiler_block(sc_label, d_name, tel_d))

    gpu_name = fp.primary_gpu.name if fp.primary_gpu else "CPU-only"
    report = TextReportBuilder("PHASE 5.2 REPORT: FUSED CHUNKED LINEAR + CROSS-ENTROPY (DEEP DEBUG)", REPO_ROOT)
    report.add_header({
        "Google Colab": fp.is_colab,
        "GPU / Backend": f"{gpu_name} | {device} (sm_{fp.sm_int})",
        "Python / PyTorch": f"{fp.host.python_version} | {fp.host.torch_version}",
        "CUDA / Triton": f"{fp.host.cuda_version or 'N/A'} | {fp.host.triton_version or 'N/A'}",
        "Runtime Cache Key": cache_key,
        "Unit (fused_lce)": f"{ran_lce} ({dt_lce:.2f}s) -> {'PASS' if ok_lce else 'FAIL'}",
        "Unit (registry)": f"{ran_kp} ({dt_kp:.2f}s) -> {'PASS' if ok_kp else 'FAIL'}",
        "Overall Phase 5.2": "PASS" if all_ok else "FAIL",
    })
    report.add_table(
        "MICRO-BENCHMARK: EAGER LINEAR+CE VS FUSED CHUNKED LINEAR+CE",
        ["Shape (N x H -> V)", "Eager ms", "Fused ms", "Speedup", "Eager VRAM MB", "Fused VRAM MB", "VRAM Delta", "LossAbsDiff"],
        micro_rows,
    )
    report.add_table(
        "END-TO-END LARGE-VOCAB TRANSFORMER_LM (BASELINE VS COMPILE VS FUSED_LCE)",
        ["Scale", "Variant", "Step0 ms", "Base ms", "Wrap ms", "Wrap P95", "Wrap P99", "E2E s (B/W)", "Speedup", "VRAM MB (B/W/d)", "MaxRelDiff", "Status"],
        e2e_rows,
    )
    report.add_table(
        "DEEP DEBUG: STEP PHASE BREAKDOWN (STEADY MEAN MS & SHARE %)",
        ["Scale", "Mode", "Step0 ms", "Steady ms", "Data/Other", "Forward", "Loss/Clip", "Backward", "Optim"],
        phase_rows,
    )
    if alloc_rows:
        report.add_table(
            "DEEP DEBUG: CUDA ALLOCATOR & VRAM HEADROOM",
            ["Scale", "Mode", "Alloc MB", "Resv MB", "Ovhead MB", "Headroom", "Util %", "Retries"],
            alloc_rows,
        )
    report.add_lines_section(
        "TOP PYTORCH PROFILER OPS (BASELINE VS COMPILE VS FUSED_LCE)",
        prof_lines if prof_lines else ["(no profiler ops captured)"],
    )
    report.add_lines_section("WARMUP VS STEADY STEP TRACE", step_traces)
    report.add_lines_section("AUTOMATED DIAGNOSIS", diag_lines)
    report.add_lines_section("KERNEL & FALLBACK AUDIT", audit_lines)
    if not all_ok:
        err_lines: list[str] = []
        if not ok_lce:
            err_lines.extend(out_lce[-1500:].splitlines())
        if not ok_kp:
            err_lines.extend(out_kp[-1500:].splitlines())
        if err_lines:
            report.add_lines_section("UNIT FAILURE TRACE", err_lines)

    report.add_verification_footer(
        status="PENDING_USER_REVIEW" if all_ok else "NEEDS_FIX",
        note="Verify Peak VRAM savings (40%-75% on large vocab) & Deep Debug traces on Colab L4 before Plan 5.3",
    )

    out_path = report.write(name="phase-5.2")
    print(report.render())
    print(f"Report written to: {out_path}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
