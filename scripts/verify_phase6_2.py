"""Phase 6.2 Verification: Hybrid Compile + FusedLinearCrossEntropy & Triton/Strided Activation Checkpointing."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import time

BENCH_DIR = Path(__file__).resolve().parent.parent / "pipline-for-test"
REPO_ROOT = BENCH_DIR.parent
for _p in (str(REPO_ROOT), str(BENCH_DIR), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from phase5_report_helper import (  # noqa: E402
    extract_allocator_row,
    extract_phase_breakdown_row,
    extract_step0_and_p99,
    format_step_trace_block,
    format_torch_profiler_block,
    run_workload_variant,
)
from src.common.text_report import TextReportBuilder  # noqa: E402
from src.config import detect_runtime_cache_key  # noqa: E402
from src.detect import collect_fingerprint  # noqa: E402
from src.kernels.fused_linear_cross_entropy import fused_linear_cross_entropy  # noqa: E402
from telemetry import check_parity, compute_speedup, compute_vram_delta  # noqa: E402


def _run_unit_file(rel_path: str) -> tuple[bool, float, str, str]:
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-m", "unittest", rel_path, "-v"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    dt = time.perf_counter() - t0
    out = proc.stdout + "\n" + proc.stderr
    ran_line = next((ln.strip() for ln in reversed(out.splitlines()) if ln.strip().startswith("Ran ")), "Ran ? tests")
    return proc.returncode == 0, dt, ran_line, out


def _bench_linear_ce_micro(is_cuda: bool) -> list[list[object]]:
    import torch
    import torch.nn.functional as F

    device = "cuda" if is_cuda else "cpu"
    shapes = (
        [
            (4096, 512, 8192, torch.bfloat16),
            (4096, 512, 32768, torch.bfloat16),
            (4096, 512, 32768, torch.float16),
            (4096, 1024, 50257, torch.bfloat16),
        ]
        if is_cuda
        else [
            (256, 128, 4096, torch.float32),
            (256, 128, 8192, torch.float32),
        ]
    )
    rows: list[list[object]] = []
    for n_rows, h_dim, vocab, dtype in shapes:
        torch.manual_seed(0)
        x0 = torch.randn(n_rows, h_dim, device=device, dtype=dtype)
        w0 = torch.randn(vocab, h_dim, device=device, dtype=dtype)
        b0 = torch.randn(vocab, device=device, dtype=dtype)
        y = torch.randint(0, vocab, (n_rows,), device=device)

        def _measure(fused: bool) -> tuple[float, float, float]:
            x = x0.clone().detach().requires_grad_(True)
            w = w0.clone().detach().requires_grad_(True)
            b = b0.clone().detach().requires_grad_(True)
            warmup, iters = (3, 6) if is_cuda else (1, 2)
            for _ in range(warmup):
                x.grad = w.grad = b.grad = None
                if fused:
                    loss = fused_linear_cross_entropy(x, w, y, bias=b)
                else:
                    loss = F.cross_entropy(F.linear(x, w, b).float(), y)
                loss.backward()
            if is_cuda:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            last_val = 0.0
            for _ in range(iters):
                x.grad = w.grad = b.grad = None
                if fused:
                    loss = fused_linear_cross_entropy(x, w, y, bias=b)
                else:
                    loss = F.cross_entropy(F.linear(x, w, b).float(), y)
                loss.backward()
                last_val = float(loss.detach().item())
            if is_cuda:
                torch.cuda.synchronize()
                peak_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            else:
                peak_mb = 0.0
            dt_ms = (time.perf_counter() - t0) * 1000.0 / iters
            return dt_ms, peak_mb, last_val

        e_ms, e_vram, e_loss = _measure(False)
        f_ms, f_vram, f_loss = _measure(True)
        diff = abs(e_loss - f_loss)
        vram_delta = f_vram - e_vram
        vram_pct = ((e_vram - f_vram) / e_vram * 100.0) if e_vram > 0 else 0.0
        dt_tag = str(dtype).replace("torch.", "")
        rows.append([
            f"{n_rows}x{h_dim} -> V={vocab} ({dt_tag})",
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
    ok_ac, dt_ac, ran_ac, out_ac = _run_unit_file("tests/unit/test_pass_actckpt_prefetch.py")
    all_ok = ok_lce and ok_kp and ok_ac

    # 2. Part A Micro-benchmark: FusedLinearCE (including fp16 dw_buf headroom)
    micro_rows = _bench_linear_ce_micro(is_cuda)

    e2e_lce_rows: list[list[object]] = []
    ckpt_rows: list[list[object]] = []
    phase_rows: list[list[object]] = []
    alloc_rows: list[list[object]] = []
    prof_lines: list[str] = []
    step_traces: list[str] = []
    audit_lines: list[str] = []
    diag_lines: list[str] = []

    with tempfile.TemporaryDirectory() as cache_root:
        # ------------------------------------------------------------------
        # Part A: Hybrid torch.compile Backbone + FusedLinearCrossEntropy Head
        # ------------------------------------------------------------------
        lce_scales = (
            [
                ("vocab8k", 16, ["--vocab-size", "8192", "--d-model", "512", "--n-heads", "8", "--n-layers", "4", "--seq-len", "256"], 12),
                ("vocab32k", 16, ["--vocab-size", "32768", "--d-model", "512", "--n-heads", "8", "--n-layers", "4", "--seq-len", "256"], 12),
            ]
            if is_cuda
            else [
                ("vocab4k_cpu", 4, ["--vocab-size", "4096", "--d-model", "128", "--n-heads", "4", "--n-layers", "2", "--seq-len", "64"], 4),
            ]
        )

        for sc_label, bs, extra, steps in lce_scales:
            sc_cache = Path(cache_root) / f"lce_{sc_label}"
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

            lce_variants = [
                ("balanced_compile", "balanced", [], sc_cache, {"PTOPT_HYBRID_LCE": "0"}),
                (
                    "fused_lce_eager",
                    "aggressive",
                    ["compile.enabled=false", "kernels.enabled=true"],
                    None,
                    {"PTOPT_FUSED_LCE_MIN_VOCAB": "4096"},
                ),
                (
                    "hybrid_compile_lce",
                    "balanced",
                    [],
                    Path(cache_root) / f"lce_hybrid_{sc_label}",
                    {"PTOPT_HYBRID_LCE": "1", "PTOPT_FUSED_LCE_MIN_VOCAB": "4096" if not is_cuda else "8192"},
                ),
            ]
            for v_name, prof, ovr, v_cache, v_env in lce_variants:
                if v_cache is not None:
                    v_cache.mkdir(parents=True, exist_ok=True)
                rec, summ, tel = run_workload_variant(
                    case_label=sc_label,
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
                    e2e_lce_rows.append([sc_label, v_name, "-", f"{base_rec.step_stats.median:.2f}", "-", "-", "-", "N/A", "0.00x", "N/A", "N/A", "FAIL"])
                    audit_lines.append(f"[ERROR] {sc_label} ({v_name}): {rec.error or base_rec.error}")
                    continue

                step0_w, p99_w = extract_step0_and_p99(summ, tel)
                parity = check_parity(base_rec.losses, rec.losses, profile=prof)
                sp = compute_speedup(base_rec.step_stats.median, rec.step_stats.median)
                vd = compute_vram_delta(base_rec.peak_allocated_mb, rec.peak_allocated_mb)
                if not parity.passed:
                    all_ok = False
                e2e_lce_rows.append([
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
                    f"- [Part A] {sc_label} ({v_name}): med={rec.step_stats.median:.2f}ms ({sp:.2f}x), "
                    f"VRAM {base_rec.peak_allocated_mb:.1f} -> {rec.peak_allocated_mb:.1f} MB ({vd:+.1f} MB)"
                )

            # Deep Debug on the primary CUDA scale (or CPU scale)
            if sc_label in ("vocab8k", "vocab32k", "vocab4k_cpu"):
                dbg_specs = [
                    ("baseline_dbg", "safe", True, [], None, {}),
                    ("balanced_compile_dbg", "balanced", False, [], sc_cache, {"PTOPT_HYBRID_LCE": "0"}),
                    (
                        "hybrid_compile_lce_dbg",
                        "balanced",
                        False,
                        [],
                        Path(cache_root) / f"lce_hybrid_{sc_label}",
                        {"PTOPT_HYBRID_LCE": "1", "PTOPT_FUSED_LCE_MIN_VOCAB": "4096" if not is_cuda else "8192"},
                    ),
                ]
                for d_name, prof, dis_p, ovr, d_cache, d_env in dbg_specs:
                    rec_d, _, tel_d = run_workload_variant(
                        case_label=sc_label,
                        wl_name="transformer_lm",
                        steps=7 if is_cuda else 4,
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

        # ------------------------------------------------------------------
        # Part B: Triton-Backed & Smarter Activation Checkpointing
        # ------------------------------------------------------------------
        ckpt_scales = (
            [
                ("lm_act_med", 16, ["--vocab-size", "8192", "--d-model", "512", "--n-heads", "8", "--n-layers", "6", "--seq-len", "512"], 12),
                ("lm_act_large", 24, ["--vocab-size", "8192", "--d-model", "768", "--n-heads", "12", "--n-layers", "8", "--seq-len", "512"], 10),
            ]
            if is_cuda
            else [
                ("lm_act_cpu", 4, ["--vocab-size", "1024", "--d-model", "128", "--n-heads", "4", "--n-layers", "4", "--seq-len", "64"], 5),
            ]
        )

        for sc_lbl, bs, extra, steps in ckpt_scales:
            sc_cache = Path(cache_root) / f"ckpt_{sc_lbl}"
            sc_cache.mkdir(parents=True, exist_ok=True)
            base_rec, base_sum, base_tel = run_workload_variant(
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
            step_traces.extend(format_step_trace_block(sc_lbl, "baseline", base_sum, base_tel))

            cpu_param_env = {"PTOPT_ACT_CKPT_MIN_PARAMS": "10000"} if not is_cuda else {}
            ckpt_variants = [
                ("no_ckpt", "balanced", ["runtime.activation_checkpointing=off"], sc_cache, {}),
                (
                    "act_ckpt_full_triton",
                    "balanced",
                    ["runtime.activation_checkpointing=always"],
                    sc_cache,
                    {**cpu_param_env, "PTOPT_ACT_CKPT_RATIO": "1.0", "PTOPT_ACT_CKPT_TRITON": "1"},
                ),
                (
                    "act_ckpt_strided_50pct",
                    "balanced",
                    ["runtime.activation_checkpointing=always"],
                    sc_cache,
                    {**cpu_param_env, "PTOPT_ACT_CKPT_RATIO": "0.5", "PTOPT_ACT_CKPT_TRITON": "1"},
                ),
                (
                    "act_ckpt_auto",
                    "balanced",
                    ["runtime.activation_checkpointing=auto"],
                    sc_cache,
                    {**cpu_param_env, "PTOPT_ACT_CKPT_RATIO": "0.5", "PTOPT_ACT_CKPT_TRITON": "1"},
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
                    ckpt_rows.append([sc_lbl, v_name, "-", "-", "-", "0.00x", "N/A", "N/A", "FAIL"])
                    audit_lines.append(f"[ERROR] {sc_lbl} ({v_name}): {rec.error or base_rec.error}")
                    continue

                step0_w, p99_w = extract_step0_and_p99(summ, tel)
                parity = check_parity(base_rec.losses, rec.losses, profile=prof)
                sp = compute_speedup(base_rec.step_stats.median, rec.step_stats.median)
                vd_base = compute_vram_delta(base_rec.peak_allocated_mb, rec.peak_allocated_mb)
                if v_name == "no_ckpt":
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
                    f"- [Part B] {sc_lbl} ({v_name}): med={rec.step_stats.median:.2f}ms ({sp:.2f}x vs safe), "
                    f"VRAM {rec.peak_allocated_mb:.1f} MB ({saved_vs_nockpt} vs no_ckpt)"
                )

            if sc_lbl in ("lm_act_med", "lm_act_cpu"):
                dbg_ckpt = [
                    ("no_ckpt_dbg", "balanced", ["runtime.activation_checkpointing=off"], sc_cache, {}),
                    (
                        "act_ckpt_full_triton_dbg",
                        "balanced",
                        ["runtime.activation_checkpointing=always"],
                        sc_cache,
                        {**cpu_param_env, "PTOPT_ACT_CKPT_RATIO": "1.0", "PTOPT_ACT_CKPT_TRITON": "1"},
                    ),
                ]
                for d_name, prof, ovr, d_cache, d_env in dbg_ckpt:
                    rec_d, _, tel_d = run_workload_variant(
                        case_label=sc_lbl,
                        wl_name="transformer_lm",
                        steps=7 if is_cuda else 4,
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
                    if rec_d.exit_code != 0 or not tel_d:
                        audit_lines.append(f"[ERROR] dbg {sc_lbl} ({d_name}): {rec_d.error}")
                        continue
                    pb_row = extract_phase_breakdown_row(sc_lbl, d_name, tel_d)
                    if pb_row:
                        phase_rows.append(pb_row)
                    mem_row = extract_allocator_row(sc_lbl, d_name, tel_d)
                    if mem_row:
                        alloc_rows.append(mem_row)
                    prof_lines.extend(format_torch_profiler_block(sc_lbl, d_name, tel_d))

    gpu_name = fp.primary_gpu.name if fp.primary_gpu else "CPU-only"
    report = TextReportBuilder("PHASE 6.2 REPORT: HYBRID COMPILE+FUSED_LCE & TRITON/STRIDED ACT_CKPT (DEEP DEBUG)", REPO_ROOT)
    report.add_header({
        "Google Colab": fp.is_colab,
        "GPU / Backend": f"{gpu_name} | {device} (sm_{fp.sm_int})",
        "Python / PyTorch": f"{fp.host.python_version} | {fp.host.torch_version}",
        "CUDA / Triton": f"{fp.host.cuda_version or 'N/A'} | {fp.host.triton_version or 'N/A'}",
        "Runtime Cache Key": cache_key,
        "Unit (fused_lce)": f"{ran_lce} ({dt_lce:.2f}s) -> {'PASS' if ok_lce else 'FAIL'}",
        "Unit (kernels_parity)": f"{ran_kp} ({dt_kp:.2f}s) -> {'PASS' if ok_kp else 'FAIL'}",
        "Unit (act_ckpt_prefetch)": f"{ran_ac} ({dt_ac:.2f}s) -> {'PASS' if ok_ac else 'FAIL'}",
        "Overall Phase 6.2": "PASS" if all_ok else "FAIL",
    })
    report.add_table(
        "PART A1 - MICRO-BENCHMARK: EAGER LINEAR+CE VS FUSED CHUNKED LINEAR+CE (BF16/FP16)",
        ["Shape (N x H -> V)", "Eager ms", "Fused ms", "Speedup", "Eager VRAM MB", "Fused VRAM MB", "VRAM Delta", "LossAbsDiff"],
        micro_rows,
    )
    report.add_table(
        "PART A2 - HYBRID COMPILE + FUSED_LCE E2E (BASELINE VS COMPILE VS FUSED_LCE VS HYBRID)",
        ["Scale", "Variant", "Step0 ms", "Base ms", "Wrap ms", "Wrap P95", "Wrap P99", "E2E s (B/W)", "Speedup", "VRAM MB (B/W/d)", "MaxRelDiff", "Status"],
        e2e_lce_rows,
    )
    report.add_table(
        "PART B - ACTIVATION CHECKPOINTING SPEED RECOVERY (RECOMPUTE AMP FIX + AUTO-TRITON + STRIDED)",
        ["Scale", "Variant", "Step0 ms", "Med ms", "P99 ms", "Speedup vs Base", "VRAM MB (B/W/d | vs no_ckpt)", "MaxRelDiff", "Status"],
        ckpt_rows,
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
        "TOP PYTORCH PROFILER OPS (PART A HYBRID LCE & PART B RECOMPUTE AMP+TRITON)",
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
        if not ok_ac:
            err_lines.extend(out_ac[-1500:].splitlines())
        if err_lines:
            report.add_lines_section("UNIT FAILURE TRACE", err_lines)

    report.add_verification_footer(
        status="PENDING_USER_REVIEW" if all_ok else "NEEDS_FIX",
        note="Verify Hybrid Compile+FusedLCE speedup+VRAM on vocab32k & lm_act_med BF16+AutoTriton speed recovery on Colab L4 before Plan 6.3",
    )

    out_path = report.write(name="phase-6.2")
    print(report.render())
    print(f"Report written to: {out_path}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
