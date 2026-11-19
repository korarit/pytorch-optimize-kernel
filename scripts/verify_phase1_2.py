#!/usr/bin/env python3
"""Phase 1.2 verification script: resolves configs across mock & live hardware and writes `report/HHMMSS_YYYYMMDD-phase-1.2.txt`."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ptopt.common.text_report import TextReportBuilder
from ptopt.config import ConfigResolver
from ptopt.detect import GPUDeviceInfo, GPUProbeResult, HardwareFingerprint, HostInfo, collect_fingerprint


def _mock_target(name: str, *, backend: str, sm_major: int = 0, sm_minor: int = 0, vram_mb: float = 0.0, shm_mb: float | None = 16384.0) -> HardwareFingerprint:
    host = HostInfo(
        hostname=f"mock-{name}",
        os_name="Darwin" if backend == "mps" else "Linux",
        os_release="6.6.0",
        cpu_arch="arm64" if backend == "mps" else "x86_64",
        cpu_model="Mock CPU",
        physical_cores=6,
        logical_cores=12,
        isa_flags=["avx2"],
        total_ram_mb=32768.0,
        available_ram_mb=24000.0,
        shm_total_mb=shm_mb,
        shm_free_mb=shm_mb,
        is_colab=(backend == "cuda"),
        python_version="3.13.0",
        torch_version="2.14.1+cu130" if backend == "cuda" else "2.15.0",
        cuda_version="13.0" if backend == "cuda" else None,
        triton_version="3.8.0" if backend == "cuda" else None,
    )
    devices = []
    if backend == "cuda":
        sm_int = sm_major * 10 + sm_minor
        devices.append(
            GPUDeviceInfo(
                index=0,
                name=name,
                compute_capability=f"{sm_major}.{sm_minor}",
                major=sm_major,
                minor=sm_minor,
                sm_int=sm_int,
                sm_name=f"sm_{sm_int}",
                total_vram_mb=vram_mb,
                sm_count=40,
                bf16_supported=sm_int >= 80,
                tf32_supported=sm_int >= 80,
                fp8_supported=sm_int >= 89,
                flash_sdp_supported=sm_int >= 80,
                mem_efficient_sdp_supported=sm_int >= 70,
            )
        )
    gpu = GPUProbeResult(backend=backend, device_count=len(devices), selected_index=0, cuda_visible_devices=None, devices=devices)
    return HardwareFingerprint(host=host, gpu=gpu)


def _run_unit_tests(repo_root: Path) -> tuple[str, int, str]:
    test_file = repo_root / "tests" / "unit" / "test_config_resolver.py"
    if importlib.util.find_spec("pytest") is not None:
        cmd = [sys.executable, "-m", "pytest", str(test_file), "-v"]
        runner = "pytest"
    else:
        cmd = [sys.executable, "-m", "unittest", "tests/unit/test_config_resolver.py", "-v"]
        runner = "unittest"
    res = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, check=False)
    return runner, res.returncode, (res.stdout + "\n" + res.stderr).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Phase 1.2 Config Rule Engine and generate text report.")
    parser.add_argument("--report-dir", type=Path, default=None)
    args = parser.parse_args()

    live_fp = collect_fingerprint()
    resolver = ConfigResolver()
    runner, rc, test_output = _run_unit_tests(REPO_ROOT)

    live_balanced = resolver.resolve(live_fp, profile="balanced", env={})

    targets = [
        ("Live Host", live_fp),
        ("Mock Colab T4 (sm_75)", _mock_target("Tesla T4", backend="cuda", sm_major=7, sm_minor=5, vram_mb=15360.0)),
        ("Mock Colab L4 (sm_89)", _mock_target("NVIDIA L4", backend="cuda", sm_major=8, sm_minor=9, vram_mb=22563.1)),
        ("Mock Colab A100 (sm_80)", _mock_target("NVIDIA A100", backend="cuda", sm_major=8, sm_minor=0, vram_mb=40960.0)),
        ("Mock Low-SHM T4", _mock_target("Tesla T4", backend="cuda", sm_major=7, sm_minor=5, vram_mb=15360.0, shm_mb=512.0)),
        ("Mock CPU-Only", _mock_target("CPU", backend="cpu")),
    ]

    matrix_rows = []
    for label, fp in targets:
        for prof in ("safe", "balanced", "aggressive"):
            cfg = resolver.resolve(fp, profile=prof, env={}).config
            matrix_rows.append([
                label,
                prof,
                cfg.runtime.tf32,
                f"{cfg.amp.enabled}/{cfg.amp.dtype}",
                cfg.amp.grad_scaler,
                f"{cfg.compile.enabled}/{cfg.compile.mode}",
                cfg.optimizer.fused,
                cfg.dataloader.auto_tune,
                cfg.kernels.enabled,
            ])

    trace_lines = [
        f"[{entry.layer}] {entry.rule_name}: {entry.field_path} ({entry.old_value!r} -> {entry.new_value!r}) | {entry.reason}"
        for entry in live_balanced.trace
    ]

    builder = TextReportBuilder(
        phase_title="PHASE 1.2: CONFIG SCHEMA V1 & LAYERED RULE ENGINE",
        repo_root=REPO_ROOT,
    )
    builder.add_header({
        "Live Backend": live_fp.effective_backend,
        "Live GPU": live_fp.primary_gpu.name if live_fp.primary_gpu else "None",
        "Test Runner": f"{runner} (exit_code={rc})",
    })
    builder.add_lines_section("LIVE HOST MATCHED RULES (balanced)", live_balanced.matched_rules)
    builder.add_lines_section("LIVE HOST RESOLUTION TRACE (balanced)", trace_lines)
    builder.add_table(
        "ARCHITECTURE x PROFILE RESOLUTION MATRIX",
        ["Target", "Profile", "TF32", "AMP", "Scaler", "Compile", "FusedOpt", "DL-Tune", "Kernels"],
        matrix_rows,
    )
    builder.add_lines_section(f"UNIT TEST OUTPUT ({runner})", test_output.splitlines())
    builder.add_verification_footer(
        status="PENDING_USER_REVIEW" if rc == 0 else "FAILED_NEEDS_ATTENTION",
        note="Inspect resolved config matrix on Google Colab before starting Phase 1.3",
    )

    out_path = builder.write("phase-1.2", report_dir=args.report_dir)
    print(builder.render())
    print(f"[SAVED] Text report written to: {out_path}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
