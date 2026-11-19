#!/usr/bin/env python3
"""Phase 5.3 Verification (Phase 3.6-Grade Deep Debug): Compilable FP8 Inference Fusion (`sm>=89`)."""

from __future__ import annotations

import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

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
    extract_step0_and_p99,
    format_step_trace_block,
    format_torch_profiler_block,
    run_workload_variant,
)
from ptopt.common.text_report import TextReportBuilder
from ptopt.config import detect_runtime_cache_key
from ptopt.detect import collect_fingerprint
from ptopt.passes.inference_opt import apply_fp8_linear_eval_hooks, prewarm_fp8_weight_caches
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


def _bench_fp8_scaling_micro(is_cuda: bool, sm_int: int) -> list[list[object]]:
    """Calibration micro-benchmark: BF16+Compile vs FP8 Eager vs FP8+Compile across (M, K, N)."""
    if not is_cuda or sm_int < 89 or not hasattr(torch, "float8_e4m3fn"):
        class _CpuSmokeBlock(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin = nn.Linear(512, 512)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.lin(F.gelu(x))

        mod = _CpuSmokeBlock().to(torch.bfloat16).eval()
        ref_mod = _CpuSmokeBlock().to(torch.bfloat16).eval()
        ref_mod.load_state_dict(mod.state_dict())
        x = torch.randn(512, 512, dtype=torch.bfloat16)

        def _fake_scaled_mm(
            a: torch.Tensor,
            b: torch.Tensor,
            *,
            scale_a: torch.Tensor,
            scale_b: torch.Tensor,
            out_dtype: torch.dtype,
            bias: torch.Tensor | None = None,
        ) -> torch.Tensor:
            out = (a.float() @ b.float()) * (scale_a.float() * scale_b.float())
            if bias is not None:
                out = out + bias.float()
            return out.to(out_dtype)

        with torch.inference_mode(), patch("ptopt.passes.inference_opt._is_cuda_tensor", new=lambda _t: True), patch.object(
            torch, "_scaled_mm", new=_fake_scaled_mm, create=True
        ):
            apply_fp8_linear_eval_hooks(mod, torch, min_work=512 * 512)
            prewarm_fp8_weight_caches(mod, torch, min_work=512 * 512)
            compiled = torch.compile(mod, fullgraph=True)
            t0 = time.perf_counter()
            y_ref = ref_mod(x)
            ms_ref = (time.perf_counter() - t0) * 1000.0
            _ = compiled(x)
            t1 = time.perf_counter()
            y_c = compiled(x)
            ms_fp8 = (time.perf_counter() - t1) * 1000.0
            rel = float((y_c.float() - y_ref.float()).abs().max().item() / y_ref.float().abs().max().clamp(min=1e-6).item())
        return [[
            "512x512 -> 512 (CPU mock)",
            f"{ms_ref:.2f}",
            "N/A",
            f"{ms_fp8:.2f}",
            f"{compute_speedup(ms_ref, ms_fp8):.2f}x",
            "N/A",
            "0.0 / 0.0",
            f"{rel:.4f}",
        ]]

    class _MlpStage(nn.Module):
        def __init__(self, k_dim: int, n_dim: int) -> None:
            super().__init__()
            self.fc1 = nn.Linear(k_dim, n_dim)
            self.fc2 = nn.Linear(n_dim, k_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc2(F.gelu(self.fc1(x)))

    specs = [
        (2048, 2048, 2048),
        (2048, 4096, 4096),
        (4096, 4096, 4096),
    ]
    rows: list[list[object]] = []
    iters = 25

    for m_rows, k_dim, n_dim in specs:
        base_m = _MlpStage(k_dim, n_dim).cuda().to(torch.bfloat16).eval()
        x = torch.randn(m_rows, k_dim, device="cuda", dtype=torch.bfloat16)

        with torch.inference_mode():
            y_ref = base_m(x)

        def _measure(mode: str) -> tuple[float, float, float]:
            mod = _MlpStage(k_dim, n_dim).cuda().to(torch.bfloat16).eval()
            mod.load_state_dict(base_m.state_dict())
            if mode in ("fp8_eager", "fp8_compile"):
                apply_fp8_linear_eval_hooks(mod, torch, min_work=512 * 512)
                prewarm_fp8_weight_caches(mod, torch, min_work=512 * 512)
            fn = torch.compile(mod, fullgraph=True) if mode in ("bf16_compile", "fp8_compile") else mod

            with torch.inference_mode():
                for _ in range(4):
                    y_last = fn(x)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                for _ in range(iters):
                    y_last = fn(x)
                torch.cuda.synchronize()
                ms = ((time.perf_counter() - t0) * 1000.0) / iters
                vram = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                rel = float(
                    (y_last.float() - y_ref.float()).abs().max().item()
                    / y_ref.float().abs().max().clamp(min=1e-6).item()
                )
            return ms, vram, rel

        ms_bf16, v_bf16, _ = _measure("bf16_compile")
        ms_fp8_e, _, _ = _measure("fp8_eager")
        ms_fp8_c, v_fp8_c, rel_fp8 = _measure("fp8_compile")
        rows.append([
            f"{m_rows}x{k_dim} -> {n_dim}",
            f"{ms_bf16:.2f}",
            f"{ms_fp8_e:.2f}",
            f"{ms_fp8_c:.2f}",
            f"{compute_speedup(ms_bf16, ms_fp8_c):.2f}x",
            f"{compute_speedup(ms_fp8_e, ms_fp8_c):.2f}x",
            f"{v_bf16:.1f} / {v_fp8_c:.1f}",
            f"{rel_fp8:.4f}",
        ])
    return rows


def _extract_infer_phase_row(
    case_label: str,
    mode_label: str,
    summary: dict[str, object] | None,
    tel_data: dict[str, object] | None,
) -> list[object] | None:
    raw = (tel_data or {}).get("step_times_ms") or (summary or {}).get("step_times_ms") or []
    vals = [float(v) for v in raw]
    if not vals:
        return None
    step0 = vals[0]
    steady = vals[1:] if len(vals) > 1 else vals
    st_mean = statistics.fmean(steady)
    st_min = min(steady)
    st_max = max(steady)
    st_std = statistics.pstdev(steady) if len(steady) > 1 else 0.0
    pb = ((tel_data or {}).get("debug_profile") or {}).get("phase_breakdown") or {}
    st_pb = pb.get("steady_mean_ms") or {}
    sh_pb = pb.get("steady_shares_pct") or {}
    host_str = f"{st_pb.get('data_other_ms', 0.0):.2f} ({sh_pb.get('data_other', 0.0)}%)" if st_pb else "N/A"
    fwd_str = f"{st_pb.get('forward_ms', st_mean):.2f} ({sh_pb.get('forward', 100.0)}%)" if st_pb else "N/A"
    return [
        case_label,
        mode_label,
        f"{step0:.2f}",
        f"{st_mean:.2f}",
        f"{st_min:.2f}/{st_max:.2f}/{st_std:.2f}",
        host_str,
        fwd_str,
    ]


def main() -> int:
    fp = collect_fingerprint()
    device = "cuda" if fp.effective_backend == "cuda" else "cpu"
    is_cuda = device == "cuda"
    sm_int = fp.sm_int
    cache_key = detect_runtime_cache_key().cache_key

    # 1. Unit & Migrate tests
    ok_inf, dt_inf, ran_inf, out_inf = _run_unit_file("tests/unit/test_pass_inference.py")
    ok_mig, dt_mig, ran_mig, out_mig = _run_unit_file("tests/migrate/test_phase3_6_inference.py")
    all_ok = ok_inf and ok_mig

    # 2. Micro-benchmark calibration across compute-intensity scales
    micro_rows = _bench_fp8_scaling_micro(is_cuda, sm_int)

    # 3. End-to-End Inference ablation (`infer_bs64_large` on CUDA / `infer_bs32_medium` on CPU)
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
        cache_dir = Path(tmp_cache) / case_label
        cache_dir.mkdir(parents=True, exist_ok=True)

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

        variants: list[tuple[str, str, list[str], dict[str, str], Path | None]] = [
            (
                "bf16_compile",
                "balanced",
                ["compile.enabled=true", "amp.enabled=true", "amp.dtype=bf16"],
                {},
                cache_dir,
            ),
            (
                "fp8_eager",
                "balanced",
                ["compile.enabled=false", "amp.enabled=true", "amp.dtype=bf16"],
                {"PTOPT_INFERENCE_FP8": "1"},
                None,
            ),
            (
                "fp8_compile",
                "balanced",
                ["compile.enabled=true", "amp.enabled=true", "amp.dtype=bf16"],
                {"PTOPT_INFERENCE_FP8": "1"},
                cache_dir,
            ),
        ]
        if is_cuda and sm_int >= 89:
            variants.append((
                "fp8_compile_drop_w",
                "balanced",
                ["compile.enabled=true", "amp.enabled=true", "amp.dtype=bf16"],
                {"PTOPT_INFERENCE_FP8": "1", "PTOPT_FP8_DROP_BF16_EVAL": "1"},
                cache_dir,
            ))

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

        for var_name, prof, ovr, extra_env, cdir in variants:
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
            ok_v = rec.exit_code == 0 and parity.passed
            all_ok = all_ok and ok_v
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
                f"{parity.max_rel_diff:.6g}",
                "PASS" if ok_v else "FAIL",
            ])
            for ev in rec.fallback_events:
                audit_lines.append(f"[{ev.get('status')}] {case_label} ({var_name}) -> {ev.get('pass_name')}: {ev.get('reason')}")
            diag_lines.append(
                f"- {case_label} ({var_name}): med={rec.step_stats.median:.2f}ms ({sp:.2f}x), "
                f"samp/s={samp_s:.1f}, VRAM={rec.peak_allocated_mb:.1f} MB ({d_mb:+.1f} MB)"
            )

        # 4. Deep Debug & PyTorch Profiler pass
        dbg_configs = [
            ("baseline_dbg", "safe", True, [], {}, None),
            ("bf16_compile_dbg", "balanced", False, ["compile.enabled=true", "amp.enabled=true", "amp.dtype=bf16"], {}, cache_dir),
            ("fp8_eager_dbg", "balanced", False, ["compile.enabled=false", "amp.enabled=true", "amp.dtype=bf16"], {"PTOPT_INFERENCE_FP8": "1"}, None),
            ("fp8_compile_dbg", "balanced", False, ["compile.enabled=true", "amp.enabled=true", "amp.dtype=bf16"], {"PTOPT_INFERENCE_FP8": "1"}, cache_dir),
        ]
        for mode_lbl, prof, dis_passes, ovr, e_env, cdir in dbg_configs:
            rec_d, d_sum, d_tel = run_workload_variant(
                case_label=case_label,
                wl_name="infer_runner_bs32",
                steps=steps,
                device=device,
                profile=prof,
                variant_name=mode_lbl,
                batch_size=bs,
                extra_args=extra_args,
                disable_builtin_passes=dis_passes,
                overrides=ovr,
                cache_dir=cdir,
                debug_profile=True,
                torch_profile=True,
                prof_start=prof_start,
                prof_steps=prof_steps,
                extra_env=e_env,
            )
            if rec_d.exit_code != 0:
                audit_lines.append(f"[ERROR] dbg {case_label} ({mode_lbl}): {rec_d.error}")
                continue
            prow = _extract_infer_phase_row(case_label, mode_lbl, d_sum, d_tel)
            if prow:
                phase_rows.append(prow)
            arow = extract_allocator_row(case_label, mode_lbl, d_tel)
            if arow:
                alloc_rows.append(arow)
            prof_lines.extend(format_torch_profiler_block(case_label, mode_lbl, d_tel))

    gpu_name = fp.primary_gpu.name if fp.primary_gpu else "CPU-only"
    report = TextReportBuilder("PHASE 5.3 REPORT: COMPILABLE FP8 INFERENCE FUSION (DEEP DEBUG)", REPO_ROOT)
    report.add_header({
        "Google Colab": fp.is_colab,
        "GPU / Backend": f"{gpu_name} | {device} (sm_{sm_int})",
        "Python / PyTorch": f"{fp.host.python_version} | {fp.host.torch_version}",
        "CUDA / Triton": f"{fp.host.cuda_version or 'N/A'} | {fp.host.triton_version or 'N/A'}",
        "Runtime Cache Key": cache_key,
        "Unit (inference)": f"{ran_inf} ({dt_inf:.2f}s) -> {'PASS' if ok_inf else 'FAIL'}",
        "Migrate (phase3.6)": f"{ran_mig} ({dt_mig:.2f}s) -> {'PASS' if ok_mig else 'FAIL'}",
        "Overall Phase 5.3": "PASS" if all_ok else "FAIL",
    })
    report.add_table(
        "MICRO-BENCHMARK CALIBRATION: BF16+COMPILE VS FP8 EAGER VS FP8+COMPILE",
        ["Shape (MxK -> N)", "BF16+Comp ms", "FP8 Eager ms", "FP8+Comp ms", "vs BF16", "vs FP8 Eager", "VRAM (BF16/FP8) MB", "MaxRel"],
        micro_rows,
    )
    report.add_table(
        "END-TO-END INFERENCE ABLATION (BASELINE VS BF16+COMPILE VS FP8+COMPILE)",
        ["Case", "Variant", "Step0 ms", "Med ms", "P95 ms", "P99 ms", "Steady Samp/s", "E2E s", "Speedup", "VRAM MB (d)", "MaxRelDiff", "Status"],
        ab_rows,
    )
    report.add_table(
        "DEEP DEBUG: INFERENCE PHASE BREAKDOWN & LATENCY JITTER",
        ["Case", "Mode", "Step0 ms", "Steady ms", "Min/Max/Std", "Host/Data (%)", "Model Fwd (%)"],
        phase_rows if phase_rows else [["-", "-", "-", "-", "-", "-", "-"]],
    )
    if alloc_rows:
        report.add_table(
            "DEEP DEBUG: CUDA ALLOCATOR & VRAM HEADROOM",
            ["Case", "Mode", "Alloc MB", "Resv MB", "Ovhead MB", "Headroom", "Util %", "Retries"],
            alloc_rows,
        )
    report.add_lines_section(
        "TOP PYTORCH PROFILER OPS (BASELINE VS BF16+COMPILE VS FP8 EAGER VS FP8+COMPILE)",
        prof_lines if prof_lines else ["(no profiler ops captured)"],
    )
    report.add_lines_section("WARMUP VS STEADY STEP TRACE", step_traces)
    report.add_lines_section("AUTOMATED DIAGNOSIS", diag_lines)
    report.add_lines_section("KERNEL & FALLBACK AUDIT", audit_lines)
    if not all_ok:
        err_lines: list[str] = []
        if not ok_inf:
            err_lines.extend(out_inf[-1500:].splitlines())
        if not ok_mig:
            err_lines.extend(out_mig[-1500:].splitlines())
        if err_lines:
            report.add_lines_section("UNIT FAILURE TRACE", err_lines)

    report.add_verification_footer(
        status="PENDING_USER_REVIEW" if all_ok else "NEEDS_FIX",
        note="Verify FP8+Compile speedup & profiler kernel fusion on Colab L4 before Plan 5.4",
    )

    out_path = report.write(name="phase-5.3")
    print(report.render())
    print(f"Report written to: {out_path}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
