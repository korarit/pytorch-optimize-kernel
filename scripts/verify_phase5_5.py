#!/usr/bin/env python3
"""Phase 5.5 Verification (Phase 3-Grade Deep Debug): Auto Activation Checkpointing & Async CUDA Prefetch."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path

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

    # 1. Unit tests
    ok_55, dt_55, ran_55, out_55 = _run_unit_file("tests/unit/test_pass_actckpt_prefetch.py")
    ok_dl, dt_dl, ran_dl, out_dl = _run_unit_file("tests/unit/test_pass_dataloader.py")
    all_ok = ok_55 and ok_dl

    ckpt_rows: list[list[object]] = []
    prefetch_rows: list[list[object]] = []
    phase_rows: list[list[object]] = []
    alloc_rows: list[list[object]] = []
    prof_lines: list[str] = []
    step_traces: list[str] = []
    diag_lines: list[str] = []
    audit_lines: list[str] = []

    with tempfile.TemporaryDirectory() as tmp_cache:
        # 2. Part A: Auto Activation Checkpointing (`transformer_lm` activation-heavy scale)
        if is_cuda:
            lm_cases: list[tuple[str, int, list[str], int]] = [
                (
                    "lm_act_med",
                    16,
                    ["--vocab-size", "2048", "--d-model", "512", "--n-heads", "8", "--n-layers", "8", "--seq-len", "512"],
                    10,
                ),
                (
                    "lm_act_large",
                    24,
                    ["--vocab-size", "2048", "--d-model", "512", "--n-heads", "8", "--n-layers", "10", "--seq-len", "512"],
                    10,
                ),
            ]
        else:
            lm_cases = [
                (
                    "lm_act_cpu",
                    4,
                    ["--vocab-size", "512", "--d-model", "128", "--n-heads", "4", "--n-layers", "3", "--seq-len", "64"],
                    5,
                ),
            ]

        for sc_lbl, bs, extra, steps in lm_cases:
            sc_cache = Path(tmp_cache) / sc_lbl
            sc_cache.mkdir(parents=True, exist_ok=True)
            base_rec, base_sum, _ = run_workload_variant(
                case_label=sc_lbl,
                wl_name="transformer_lm",
                steps=steps,
                device=device,
                profile="safe",
                variant_name="baseline",
                extra_args=extra,
                batch_size=bs,
                is_baseline=True,
            )
            step_traces.extend(format_step_trace_block(sc_lbl, "baseline", base_sum, None))

            ckpt_variants = [
                ("balanced_no_ckpt", "balanced", ["runtime.activation_checkpointing=off"], sc_cache, {}),
                (
                    "balanced_act_ckpt",
                    "balanced",
                    ["runtime.activation_checkpointing=always"],
                    sc_cache,
                    {"PTOPT_ACT_CKPT_MIN_PARAMS": "60000"} if not is_cuda else {},
                ),
            ]
            no_ckpt_vram: float | None = None
            for v_name, prof, ovr, v_cache, v_env in ckpt_variants:
                rec, summ, tel = run_workload_variant(
                    case_label=sc_lbl,
                    wl_name="transformer_lm",
                    steps=steps,
                    device=device,
                    profile=prof,
                    variant_name=v_name,
                    extra_args=extra,
                    batch_size=bs,
                    overrides=ovr,
                    cache_dir=v_cache,
                    extra_env=v_env,
                )
                if base_rec.exit_code != 0 or rec.exit_code != 0:
                    all_ok = False
                    ckpt_rows.append([sc_lbl, v_name, "-", "-", "-", "-", "N/A", "FAIL"])
                    audit_lines.append(f"[ERROR] {sc_lbl} ({v_name}): {rec.error or base_rec.error}")
                    continue

                step0_w, p99_w = extract_step0_and_p99(summ, tel)
                parity = check_parity(base_rec.losses, rec.losses, profile=prof)
                sp = compute_speedup(base_rec.step_stats.median, rec.step_stats.median)
                vd_base = compute_vram_delta(base_rec.peak_allocated_mb, rec.peak_allocated_mb)
                if v_name == "balanced_no_ckpt":
                    no_ckpt_vram = rec.peak_allocated_mb
                    saved_vs_nockpt = "0.0%"
                else:
                    pct = ((no_ckpt_vram - rec.peak_allocated_mb) / no_ckpt_vram * 100.0) if (no_ckpt_vram and no_ckpt_vram > 0) else 0.0
                    saved_vs_nockpt = f"{pct:.1f}% saved"
                if not parity.passed:
                    all_ok = False
                ckpt_rows.append([
                    sc_lbl,
                    v_name,
                    f"{step0_w:.1f}",
                    f"{rec.step_stats.median:.2f}",
                    f"{p99_w:.2f}",
                    f"{sp:.2f}x",
                    f"{base_rec.peak_allocated_mb:.1f}/{rec.peak_allocated_mb:.1f} ({vd_base:+.1f} | {saved_vs_nockpt})",
                    parity.max_rel_diff,
                    "PASS" if parity.passed else "FAIL",
                ])
                step_traces.extend(format_step_trace_block(sc_lbl, v_name, summ, tel))
                for ev in rec.fallback_events:
                    audit_lines.append(f"[{ev.get('status')}] {sc_lbl} ({v_name}) -> {ev.get('pass_name')}: {ev.get('reason')}")
                diag_lines.append(
                    f"- {sc_lbl} ({v_name}): med={rec.step_stats.median:.2f}ms ({sp:.2f}x), "
                    f"VRAM {rec.peak_allocated_mb:.1f} MB ({saved_vs_nockpt} vs no_ckpt)"
                )

            # Deep Debug for Part A
            dbg_a = [
                ("no_ckpt_dbg", "balanced", ["runtime.activation_checkpointing=off"], sc_cache, {}),
                (
                    "act_ckpt_dbg",
                    "balanced",
                    ["runtime.activation_checkpointing=always"],
                    sc_cache,
                    {"PTOPT_ACT_CKPT_MIN_PARAMS": "60000"} if not is_cuda else {},
                ),
            ]
            for d_name, prof, ovr, d_cache, d_env in dbg_a:
                rec_d, _, tel_d = run_workload_variant(
                    case_label=sc_lbl,
                    wl_name="transformer_lm",
                    steps=6 if is_cuda else 4,
                    device=device,
                    profile=prof,
                    variant_name=d_name,
                    extra_args=extra,
                    batch_size=bs,
                    overrides=ovr,
                    cache_dir=d_cache,
                    debug_profile=True,
                    torch_profile=True,
                    prof_start=2 if is_cuda else 1,
                    prof_steps=2,
                    extra_env=d_env,
                )
                if rec_d.exit_code == 0 and tel_d:
                    pb = extract_phase_breakdown_row(sc_lbl, d_name, tel_d)
                    if pb:
                        phase_rows.append(pb)
                    ar = extract_allocator_row(sc_lbl, d_name, tel_d)
                    if ar:
                        alloc_rows.append(ar)
                    prof_lines.extend(format_torch_profiler_block(sc_lbl, d_name, tel_d))

        # 3. Part B: Dedicated CUDA Copy Stream Prefetch (`cnn_vision` heavy H2D payload)
        if is_cuda:
            vis_label = "cnn_vision_224"
            vis_bs = 64
            vis_extra = ["--image-size", "224", "--width", "64"]
            vis_steps = 25
        else:
            vis_label = "cnn_vision_cpu"
            vis_bs = 8
            vis_extra = ["--image-size", "32", "--width", "32"]
            vis_steps = 5

        vis_cache = Path(tmp_cache) / vis_label
        vis_cache.mkdir(parents=True, exist_ok=True)
        vbase_rec, vbase_sum, _ = run_workload_variant(
            case_label=vis_label,
            wl_name="cnn_vision",
            steps=vis_steps,
            device=device,
            profile="safe",
            variant_name="baseline",
            extra_args=vis_extra,
            batch_size=vis_bs,
            is_baseline=True,
        )
        step_traces.extend(format_step_trace_block(vis_label, "baseline", vbase_sum, None))

        pf_variants = [("dl_no_prefetch", "balanced", ["dataloader.cuda_prefetch=false"], vis_cache)]
        if is_cuda:
            pf_variants.append(("dl_cuda_prefetch", "balanced", ["dataloader.cuda_prefetch=true"], vis_cache))
        for v_name, prof, ovr, v_cache in pf_variants:
            rec, summ, tel = run_workload_variant(
                case_label=vis_label,
                wl_name="cnn_vision",
                steps=vis_steps,
                device=device,
                profile=prof,
                variant_name=v_name,
                extra_args=vis_extra,
                batch_size=vis_bs,
                overrides=ovr,
                cache_dir=v_cache,
            )
            if vbase_rec.exit_code != 0 or rec.exit_code != 0:
                all_ok = False
                prefetch_rows.append([vis_label, v_name, "-", "-", "-", "0.00x", "N/A", "FAIL"])
                audit_lines.append(f"[ERROR] {vis_label} ({v_name}): {rec.error or vbase_rec.error}")
                continue
            step0_w, p99_w = extract_step0_and_p99(summ, tel)
            parity = check_parity(vbase_rec.losses, rec.losses, profile=prof)
            sp = compute_speedup(vbase_rec.step_stats.median, rec.step_stats.median)
            vd = compute_vram_delta(vbase_rec.peak_allocated_mb, rec.peak_allocated_mb)
            if not parity.passed:
                all_ok = False
            prefetch_rows.append([
                vis_label,
                v_name,
                f"{step0_w:.1f}",
                f"{rec.step_stats.median:.2f}",
                f"{p99_w:.2f}",
                f"{vbase_rec.end_to_end_sec:.2f}/{rec.end_to_end_sec:.2f}s",
                f"{sp:.2f}x",
                f"{vbase_rec.peak_allocated_mb:.1f}/{rec.peak_allocated_mb:.1f} ({vd:+.1f})",
                parity.max_rel_diff,
                "PASS" if parity.passed else "FAIL",
            ])
            step_traces.extend(format_step_trace_block(vis_label, v_name, summ, tel))
            for ev in rec.fallback_events:
                audit_lines.append(f"[{ev.get('status')}] {vis_label} ({v_name}) -> {ev.get('pass_name')}: {ev.get('reason')}")
            diag_lines.append(
                f"- {vis_label} ({v_name}): med={rec.step_stats.median:.2f}ms ({sp:.2f}x), E2E={rec.end_to_end_sec:.2f}s, "
                f"VRAM {rec.peak_allocated_mb:.1f} MB ({vd:+.1f} MB)"
            )

        # Deep Debug for Part B (`Data/Other %` vs `cudaMemcpyAsync`)
        dbg_b = [("dl_no_prefetch_dbg", ["dataloader.cuda_prefetch=false"])]
        if is_cuda:
            dbg_b.append(("dl_cuda_prefetch_dbg", ["dataloader.cuda_prefetch=true"]))
        for d_name, ovr in dbg_b:
            rec_d, _, tel_d = run_workload_variant(
                case_label=vis_label,
                wl_name="cnn_vision",
                steps=8 if is_cuda else 4,
                device=device,
                profile="balanced",
                variant_name=d_name,
                extra_args=vis_extra,
                batch_size=vis_bs,
                overrides=ovr,
                cache_dir=vis_cache,
                debug_profile=True,
                torch_profile=True,
                prof_start=3 if is_cuda else 1,
                prof_steps=3 if is_cuda else 2,
            )
            if rec_d.exit_code == 0 and tel_d:
                pb = extract_phase_breakdown_row(vis_label, d_name, tel_d)
                if pb:
                    phase_rows.append(pb)
                ar = extract_allocator_row(vis_label, d_name, tel_d)
                if ar:
                    alloc_rows.append(ar)
                prof_lines.extend(format_torch_profiler_block(vis_label, d_name, tel_d))

    gpu_name = fp.primary_gpu.name if fp.primary_gpu else "CPU-only"
    report = TextReportBuilder("PHASE 5.5 REPORT: AUTO ACTIVATION CHECKPOINTING & ASYNC CUDA PREFETCH (DEEP DEBUG)", REPO_ROOT)
    report.add_header({
        "Google Colab": fp.is_colab,
        "GPU / Backend": f"{gpu_name} | {device} (sm_{sm_int})",
        "Python / PyTorch": f"{fp.host.python_version} | {fp.host.torch_version}",
        "CUDA / Triton": f"{fp.host.cuda_version or 'N/A'} | {fp.host.triton_version or 'N/A'}",
        "Runtime Cache Key": cache_key,
        "Unit (actckpt+pf)": f"{ran_55} ({dt_55:.2f}s) -> {'PASS' if ok_55 else 'FAIL'}",
        "Unit (dataloader)": f"{ran_dl} ({dt_dl:.2f}s) -> {'PASS' if ok_dl else 'FAIL'}",
        "Overall Phase 5.5": "PASS" if all_ok else "FAIL",
    })
    report.add_table(
        "PART A: AUTO ACTIVATION CHECKPOINTING (VRAM REDUCTION & PARITY)",
        ["Case", "Variant", "Step0 ms", "Med ms", "Wrap P99", "Speedup", "VRAM MB (B/W/d | Saved)", "MaxRelDiff", "Status"],
        ckpt_rows,
    )
    report.add_table(
        "PART B: DEDICATED CUDA COPY STREAM PREFETCH (HEAVY H2D THROUGHPUT)",
        ["Case", "Variant", "Step0 ms", "Med ms", "Wrap P99", "E2E s (B/W)", "Speedup", "VRAM MB (B/W/d)", "MaxRelDiff", "Status"],
        prefetch_rows,
    )
    report.add_table(
        "DEEP DEBUG: STEP PHASE BREAKDOWN (DATA/OTHER VS FORWARD/BACKWARD)",
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
        "TOP PYTORCH PROFILER OPS (RECOMPUTE & cudaMemcpyAsync OVERLAP)",
        prof_lines if prof_lines else ["(no profiler ops captured)"],
    )
    report.add_lines_section("WARMUP VS STEADY STEP TRACE", step_traces)
    report.add_lines_section("AUTOMATED DIAGNOSIS", diag_lines)
    report.add_lines_section("KERNEL & FALLBACK AUDIT", audit_lines)
    if not all_ok:
        err_lines: list[str] = []
        if not ok_55:
            err_lines.extend(out_55[-1500:].splitlines())
        if not ok_dl:
            err_lines.extend(out_dl[-1500:].splitlines())
        if err_lines:
            report.add_lines_section("UNIT FAILURE TRACE", err_lines)

    report.add_verification_footer(
        status="PENDING_USER_REVIEW" if all_ok else "NEEDS_FIX",
        note="Verify 30%-60% Activation VRAM reduction and H2D Data/Other overlap on Colab L4",
    )

    out_path = report.write(name="phase-5.5")
    print(report.render())
    print(f"Report written to: {out_path}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
