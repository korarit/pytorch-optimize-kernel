#!/usr/bin/env python3
"""Phase 6.3 Verification (Deep Debug): LLM Kernels (RoPE & Zero-Copy Strides), Static FP8 & Startup Polish."""

from __future__ import annotations

import os
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
from src.kernels import (  # noqa: E402
    ref_cross_entropy,
    ref_rmsnorm,
    ref_rope,
    ref_swiglu,
    triton_cross_entropy,
    triton_rmsnorm,
    triton_rope,
    triton_swiglu,
)
from telemetry import check_parity, compute_speedup, compute_vram_delta  # noqa: E402


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


def _bench_kernels_micro(is_cuda: bool) -> list[list[object]]:
    import torch

    device = torch.device("cuda" if is_cuda else "cpu")
    dtype = torch.bfloat16 if is_cuda else torch.float32
    warmup, iters = (5, 15) if is_cuda else (1, 3)
    rows: list[list[object]] = []

    def _run_pair(name: str, bytes_moved: int, fn_ref: object, fn_tri: object) -> None:
        def _meas(use_tri: bool) -> tuple[float, float, float]:
            fn = fn_tri if use_tri else fn_ref
            for _ in range(warmup):
                val = fn()  # type: ignore[operator]
            if is_cuda:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            last_v = 0.0
            for _ in range(iters):
                last_v = float(fn())  # type: ignore[operator]
            if is_cuda:
                torch.cuda.synchronize()
                vram = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            else:
                vram = 0.0
            ms = ((time.perf_counter() - t0) * 1000.0) / iters
            return ms, vram, last_v

        r_ms, r_vram, r_val = _meas(False)
        t_ms, t_vram, t_val = _meas(True)
        gbps = (bytes_moved / (t_ms * 1e-3)) / 1e9 if t_ms > 0 else 0.0
        dv = t_vram - r_vram
        rows.append([
            name,
            f"{r_ms:.3f}",
            f"{t_ms:.3f}",
            f"{compute_speedup(r_ms, t_ms):.2f}x",
            f"{gbps:.1f}",
            f"{r_vram:.1f}/{t_vram:.1f} ({dv:+.1f})",
            f"{abs(r_val - t_val):.2e}",
        ])

    torch.manual_seed(0)
    # 1. Fused RoPE (Fwd + Bwd)
    b, h_q, h_k, s, d = (4, 8, 8, 512, 128) if is_cuda else (2, 4, 4, 32, 64)
    q0 = torch.randn(b, h_q, s, d, device=device, dtype=dtype)
    k0 = torch.randn(b, h_k, s, d, device=device, dtype=dtype)
    ang = torch.randn(1, 1, s, d // 2, device=device, dtype=dtype)
    cos0 = torch.cat([ang.cos(), ang.cos()], dim=-1)
    sin0 = torch.cat([ang.sin(), ang.sin()], dim=-1)

    def _rope_step(use_t: bool) -> float:
        q = q0.clone().requires_grad_(True)
        k = k0.clone().requires_grad_(True)
        qr, kr = triton_rope(q, k, cos0, sin0) if use_t else ref_rope(q, k, cos0, sin0)
        loss = (qr.sum() + kr.sum()) * 1e-4
        loss.backward()
        return float(loss.detach().item())

    elem_sz = 2 if dtype == torch.bfloat16 else 4
    rope_bytes = 4 * (q0.numel() + k0.numel()) * elem_sz  # fwd read+write + bwd read+write
    _run_pair(f"RoPE Fwd+Bwd ({b}x{h_q}x{s}x{d})", rope_bytes, lambda: _rope_step(False), lambda: _rope_step(True))

    # 2. Zero-Copy Row-Strided CrossEntropy (slice [:, :V] from padded [:, V+128])
    m_ce, v_ce = (4096, 8192) if is_cuda else (128, 1024)
    ce_buf = torch.randn(m_ce, v_ce + 128, device=device, dtype=dtype)
    ce_tgt = torch.randint(0, v_ce, (m_ce,), device=device)

    def _ce_step(use_t: bool) -> float:
        logits = ce_buf[:, :v_ce].detach().requires_grad_(True)
        loss = (
            triton_cross_entropy(logits, ce_tgt, reduction="mean")
            if use_t
            else ref_cross_entropy(logits, ce_tgt, reduction="mean")
        )
        loss.backward()
        return float(loss.detach().item())

    ce_bytes = m_ce * v_ce * elem_sz * 3
    _run_pair(f"CrossEntropy Strided ({m_ce}x{v_ce})", ce_bytes, lambda: _ce_step(False), lambda: _ce_step(True))

    # 3. Packed Multi-row RMSNorm (small N=512 & 768)
    m_rms, n_rms = (4096, 512) if is_cuda else (256, 256)
    x_rms0 = torch.randn(m_rms, n_rms, device=device, dtype=dtype)
    w_rms0 = torch.ones(n_rms, device=device, dtype=dtype)

    def _rms_step(use_t: bool) -> float:
        x = x_rms0.clone().requires_grad_(True)
        y = triton_rmsnorm(x, w_rms0) if use_t else ref_rmsnorm(x, w_rms0)
        loss = y.sum() * 1e-4
        loss.backward()
        return float(loss.detach().item())

    rms_bytes = m_rms * n_rms * elem_sz * 5
    _run_pair(f"RMSNorm Packed ({m_rms}x{n_rms})", rms_bytes, lambda: _rms_step(False), lambda: _rms_step(True))

    # 4. Zero-Copy Strided SwiGLU (`gate_up.chunk(2, dim=-1)`)
    m_swi, n_swi = (4096, 2048) if is_cuda else (256, 256)
    gate_up = torch.randn(m_swi, 2 * n_swi, device=device, dtype=dtype)

    def _swi_step(use_t: bool) -> float:
        gu = gate_up.detach().clone().requires_grad_(True)
        g, u = gu.chunk(2, dim=-1)
        y = triton_swiglu(g, u) if use_t else ref_swiglu(g, u)
        loss = y.sum() * 1e-4
        loss.backward()
        return float(loss.detach().item())

    swi_bytes = m_swi * n_swi * elem_sz * 8
    _run_pair(f"SwiGLU ZeroCopy Chunk ({m_swi}x{n_swi})", swi_bytes, lambda: _swi_step(False), lambda: _swi_step(True))
    return rows


def _bench_fingerprint_startup() -> list[list[object]]:
    """Measure cold Python subprocess `collect_fingerprint()` latency with vs without disk cache."""
    rows: list[list[object]] = []
    with tempfile.TemporaryDirectory() as cache_dir:
        cmd = [sys.executable, "-c", "from src.detect.fingerprint import collect_fingerprint; fp = collect_fingerprint(); print(fp.effective_backend, fp.host.shm_total_mb)"]
        base_env = dict(os.environ)
        base_env["PTOPT_FP_CACHE_DIR"] = cache_dir

        def _time_subproc(no_cache: str, runs: int = 3) -> tuple[float, int]:
            env = dict(base_env)
            env["PTOPT_NO_FP_CACHE"] = no_cache
            dts: list[float] = []
            rc = 0
            for _ in range(runs):
                t0 = time.perf_counter()
                proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=False)
                dts.append((time.perf_counter() - t0) * 1000.0)
                if proc.returncode != 0:
                    rc = proc.returncode
            return sorted(dts)[len(dts) // 2], rc

        ms_nocache, rc1 = _time_subproc("1", runs=3)
        ms_miss, rc2 = _time_subproc("0", runs=1)  # populate disk cache
        ms_hit, rc3 = _time_subproc("0", runs=3)  # warm disk cache hit + live /dev/shm
        rows.append([
            "collect_fingerprint() subprocess",
            f"{ms_nocache:.1f}",
            f"{ms_miss:.1f}",
            f"{ms_hit:.1f}",
            f"{compute_speedup(ms_nocache, ms_hit):.2f}x",
            "PASS" if (rc1 == 0 and rc2 == 0 and rc3 == 0) else "FAIL",
        ])
    return rows


def main() -> int:
    fp = collect_fingerprint()
    device = "cuda" if fp.effective_backend == "cuda" else "cpu"
    is_cuda = device == "cuda"
    sm_int = fp.sm_int
    cache_key = detect_runtime_cache_key().cache_key

    unit_files = [
        "tests/unit/test_phase6_3_features.py",
        "tests/unit/test_kernels_parity.py",
        "tests/unit/test_pass_inference.py",
        "tests/unit/test_detect.py",
        "tests/unit/test_pass_optimizer_amp.py",
        "tests/unit/test_pass_dataloader.py",
    ]
    unit_summary: dict[str, str] = {}
    err_traces: list[str] = []
    all_ok = True
    for rel in unit_files:
        ok, dt, ran, out = _run_unit_file(rel)
        unit_summary[Path(rel).name] = f"{ran} ({dt:.2f}s) -> {'PASS' if ok else 'FAIL'}"
        if not ok:
            all_ok = False
            err_traces.extend(out[-1500:].splitlines())

    micro_rows = _bench_kernels_micro(is_cuda)
    startup_rows = _bench_fingerprint_startup()

    # E2E FP8 Dynamic vs Static Calibration on infer_bs64_large (CUDA) / infer_bs32_medium (CPU)
    if is_cuda:
        case_label = "infer_bs64_large"
        bs = 64
        steps = 20
        extra_args = ["--d-model", "768", "--n-layers", "12", "--seq-len", "256"]
        prof_start, prof_steps = 4, 5
    else:
        case_label = "infer_bs32_medium"
        bs = 32
        steps = 6
        extra_args = ["--d-model", "256", "--n-layers", "4", "--seq-len", "64"]
        prof_start, prof_steps = 1, 2

    ab_rows: list[list[object]] = []
    phase_rows: list[list[object]] = []
    alloc_rows: list[list[object]] = []
    prof_lines: list[str] = []
    step_traces: list[str] = []
    diag_lines: list[str] = []
    audit_lines: list[str] = []

    with tempfile.TemporaryDirectory() as tmp_cache:
        base_rec, base_sum, _ = run_workload_variant(
            case_label=case_label,
            wl_name="infer_runner_bs32",
            steps=steps,
            device=device,
            profile="safe",
            variant_name="baseline",
            batch_size=bs,
            extra_args=extra_args,
            is_baseline=True,
        )
        base_s0, base_p99 = extract_step0_and_p99(base_sum, None)
        step_traces.extend(format_step_trace_block(case_label, "baseline", base_sum, None))
        ab_rows.append([
            case_label,
            "baseline",
            f"{base_s0:.1f}",
            f"{base_rec.step_stats.median:.2f}",
            f"{base_rec.step_stats.p95:.2f}",
            f"{base_p99:.2f}",
            f"{(bs * 1000.0 / base_rec.step_stats.median):.1f}" if base_rec.step_stats.median > 0 else "0.0",
            f"{base_rec.end_to_end_sec:.2f}s",
            "1.00x",
            f"{base_rec.peak_allocated_mb:.1f} (+0.0)",
            "0.0",
            "BASE",
        ])

        variants: list[tuple[str, str, list[str], dict[str, str], Path]] = [
            (
                "bf16_compile",
                "balanced",
                ["compile.enabled=true", "amp.enabled=true", "amp.dtype=bf16"],
                {},
                Path(tmp_cache) / "bf16_compile",
            ),
        ]
        if is_cuda and sm_int >= 89:
            variants.extend([
                (
                    "fp8_compile_dynamic",
                    "balanced",
                    ["compile.enabled=true", "amp.enabled=true", "amp.dtype=bf16"],
                    {"PTOPT_INFERENCE_FP8": "1", "PTOPT_FP8_STATIC_SCALE": "0"},
                    Path(tmp_cache) / "fp8_dyn",
                ),
                (
                    "fp8_compile_static",
                    "balanced",
                    ["compile.enabled=true", "amp.enabled=true", "amp.dtype=bf16"],
                    {"PTOPT_INFERENCE_FP8": "1", "PTOPT_FP8_STATIC_SCALE": "1"},
                    Path(tmp_cache) / "fp8_static",
                ),
            ])

        for var_name, prof, ovr, extra_env, cdir in variants:
            cdir.mkdir(parents=True, exist_ok=True)
            rec, w_sum, w_tel = run_workload_variant(
                case_label=case_label,
                wl_name="infer_runner_bs32",
                steps=steps,
                device=device,
                profile=prof,
                variant_name=var_name,
                batch_size=bs,
                extra_args=extra_args,
                overrides=ovr,
                cache_dir=cdir,
                extra_env=extra_env,
            )
            if base_rec.exit_code != 0 or rec.exit_code != 0:
                all_ok = False
                ab_rows.append([case_label, var_name, "-", "-", "-", "-", "-", "-", "0.00x", "N/A", "N/A", "FAIL"])
                audit_lines.append(f"[ERROR] {case_label} ({var_name}): {rec.error or base_rec.error}")
                continue

            w_s0, w_p99 = extract_step0_and_p99(w_sum, w_tel)
            step_traces.extend(format_step_trace_block(case_label, var_name, w_sum, w_tel))
            tol_prof = "aggressive" if "fp8" in var_name else "balanced"
            parity = check_parity(base_rec.losses, rec.losses, profile=tol_prof)
            sp = compute_speedup(base_rec.step_stats.median, rec.step_stats.median)
            d_mb = compute_vram_delta(base_rec.peak_allocated_mb, rec.peak_allocated_mb)
            samp_s = (bs * 1000.0 / rec.step_stats.median) if rec.step_stats.median > 0 else 0.0
            ok_v = parity.passed
            if not ok_v:
                all_ok = False
            ab_rows.append([
                case_label,
                var_name,
                f"{w_s0:.1f}",
                f"{rec.step_stats.median:.2f}",
                f"{rec.step_stats.p95:.2f}",
                f"{w_p99:.2f}",
                f"{samp_s:.1f}",
                f"{rec.end_to_end_sec:.2f}s",
                f"{sp:.2f}x",
                f"{rec.peak_allocated_mb:.1f} ({d_mb:+.1f})",
                parity.max_rel_diff,
                "PASS" if ok_v else "FAIL",
            ])
            for ev in rec.fallback_events:
                audit_lines.append(f"[{ev.get('status')}] {case_label} ({var_name}) -> {ev.get('pass_name')}: {ev.get('reason')}")
            diag_lines.append(
                f"- {case_label} ({var_name}): med={rec.step_stats.median:.2f}ms ({sp:.2f}x), "
                f"{samp_s:.1f} samp/s, VRAM={rec.peak_allocated_mb:.1f} MB ({d_mb:+.1f} MB)"
            )

        # Deep Debug + Torch Profiler comparison (dynamic vs static FP8)
        for dbg_name, dbg_prof, ovr, d_env, d_cache in variants:
            rec_d, _, tel_d = run_workload_variant(
                case_label=case_label,
                wl_name="infer_runner_bs32",
                steps=12 if is_cuda else 5,
                device=device,
                profile=dbg_prof,
                variant_name=f"{dbg_name}_dbg",
                batch_size=bs,
                extra_args=extra_args,
                overrides=ovr,
                cache_dir=d_cache,
                debug_profile=True,
                torch_profile=True,
                prof_start=prof_start,
                prof_steps=prof_steps,
                extra_env=d_env,
            )
            if rec_d.exit_code == 0 and tel_d:
                pb_row = extract_phase_breakdown_row(case_label, f"{dbg_name}_dbg", tel_d)
                if pb_row:
                    phase_rows.append(pb_row)
                mem_row = extract_allocator_row(case_label, f"{dbg_name}_dbg", tel_d)
                if mem_row:
                    alloc_rows.append(mem_row)
                prof_lines.extend(format_torch_profiler_block(case_label, f"{dbg_name}_dbg", tel_d))

    # Optional quick fast-gate check
    gate_status = "SKIPPED"
    if os.environ.get("PTOPT_SKIP_FAST_GATE", "0") != "1":
        t0_g = time.perf_counter()
        proc_g = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts/run_gate.py"), "--mode", "fast"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        dt_g = time.perf_counter() - t0_g
        gate_status = f"{'PASS' if proc_g.returncode == 0 else 'FAIL'} ({dt_g:.1f}s)"
        if proc_g.returncode != 0:
            all_ok = False
            err_traces.extend((proc_g.stdout + "\n" + proc_g.stderr)[-1500:].splitlines())

    gpu_name = fp.primary_gpu.name if fp.primary_gpu else "CPU-only"
    report = TextReportBuilder("PHASE 6.3 REPORT: LLM KERNELS (ROPE/STRIDED), STATIC FP8 & STARTUP POLISH", REPO_ROOT)
    hdr: dict[str, object] = {
        "Google Colab": fp.is_colab,
        "GPU / Backend": f"{gpu_name} | {device} (sm_{sm_int})",
        "Python / PyTorch": f"{fp.host.python_version} | {fp.host.torch_version}",
        "CUDA / Triton": f"{fp.host.cuda_version or 'N/A'} | {fp.host.triton_version or 'N/A'}",
        "Runtime Cache Key": cache_key,
    }
    for k, v in unit_summary.items():
        hdr[f"Unit ({k})"] = v
    hdr["Fast Gate (--mode fast)"] = gate_status
    hdr["Overall Phase 6.3"] = "PASS" if all_ok else "FAIL"
    report.add_header(hdr)

    report.add_table(
        "MICRO-BENCHMARK: FUSED ROPE, ZERO-COPY STRIDED CE/SWIGLU & PACKED RMSNORM",
        ["Kernel (Shape)", "Ref ms", "Triton ms", "Speedup", "GB/s", "VRAM MB (R/T/d)", "AbsDiff"],
        micro_rows,
    )
    report.add_table(
        "COLD LAUNCH: FINGERPRINT DISK CACHE LATENCY (NO CACHE VS CACHE HIT)",
        ["Probe Path", "NoCache ms", "Miss ms", "Hit ms", "Speedup", "Status"],
        startup_rows,
    )
    report.add_table(
        f"E2E INFERENCE ABLATION ({case_label}: DYNAMIC VS STATIC FP8 + COMPILE)",
        ["Case", "Variant", "Step0 ms", "Med ms", "P95 ms", "P99 ms", "Samp/s", "E2E s", "Speedup", "VRAM MB", "MaxRelDiff", "Status"],
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
        "TOP PYTORCH PROFILER OPS (DYNAMIC VS STATIC FP8 QUANTIZATION)",
        prof_lines if prof_lines else ["(no profiler ops captured)"],
    )
    report.add_lines_section("WARMUP VS STEADY STEP TRACE", step_traces)
    report.add_lines_section("AUTOMATED DIAGNOSIS", diag_lines)
    report.add_lines_section("KERNEL & FALLBACK AUDIT", audit_lines)
    if err_traces:
        report.add_lines_section("UNIT / GATE FAILURE TRACE", err_traces)

    report.add_verification_footer(
        status="PENDING_USER_REVIEW" if all_ok else "NEEDS_FIX",
        note="Verify Fused RoPE, zero-copy strided kernels, static FP8 reduction drop, and cold CLI startup on Colab L4",
    )

    out_path = report.write(name="phase-6.3")
    print(report.render())
    print(f"Report written to: {out_path}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
