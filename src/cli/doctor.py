"""Environment diagnostics (`ptopt --doctor`), readiness checks, and profile recommendation."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from src.config import OptimizeConfig, detect_runtime_cache_key
from src.detect import HardwareFingerprint, collect_fingerprint


@dataclass(frozen=True)
class DoctorCheck:
    """Single hardware/software readiness check outcome."""

    name: str
    status: str  # OK | WARN | MISSING
    detail: str
    remediation: str | None = None


@dataclass(frozen=True)
class DoctorReport:
    """Complete environment diagnostic report produced by `run_doctor`."""

    fingerprint: HardwareFingerprint
    checks: tuple[DoctorCheck, ...]
    recommended_profile: str
    recommendation_reason: str
    cache_dir: Path
    report_dir: Path
    runtime_cache_key: str

    @property
    def ok_count(self) -> int:
        return sum(1 for c in self.checks if c.status == "OK")

    @property
    def warn_count(self) -> int:
        return sum(1 for c in self.checks if c.status == "WARN")

    @property
    def missing_count(self) -> int:
        return sum(1 for c in self.checks if c.status == "MISSING")


def recommend_profile(fp: HardwareFingerprint) -> tuple[str, str]:
    """Recommend the best default `ptopt` profile for the detected hardware."""
    backend = fp.effective_backend
    if backend == "cpu":
        return (
            "supersafe",
            "CPU-only environment detected -> recommended: supersafe "
            "(strict FP32 parity; CUDA passes skipped safely)",
        )
    if backend == "mps":
        return (
            "safe",
            "Apple Silicon MPS detected -> recommended: safe "
            "(inference_mode enabled without CUDA-only passes)",
        )

    dev = fp.primary_gpu
    gpu_name = dev.name if dev is not None else "CUDA GPU"
    vram_gb = (fp.vram_mb or 0.0) / 1024.0
    shm_str = f"{int(fp.host.shm_total_mb)}MB /dev/shm" if fp.host.shm_total_mb is not None else "unknown /dev/shm"

    if fp.sm_int < 80:
        extra = " + low-/dev/shm worker cap" if fp.is_low_shm else ""
        return (
            "balanced",
            f"Detected {gpu_name} (sm_{fp.sm_int}, {vram_gb:.1f}GB VRAM, {shm_str}) "
            f"-> recommended: balanced (fp16 AMP + GradScaler, TF32 off{extra})",
        )

    return (
        "balanced",
        f"Detected {gpu_name} (sm_{fp.sm_int}, {vram_gb:.1f}GB VRAM, {shm_str}) "
        "-> recommended: balanced (bf16 AMP, TF32 on, torch.compile default; "
        "use --profile aggressive for CUDA Graphs / max throughput)",
    )


def _check_cuda(fp: HardwareFingerprint) -> DoctorCheck:
    if fp.effective_backend == "cuda" and fp.primary_gpu is not None:
        dev = fp.primary_gpu
        drv = fp.gpu.driver_version or "unknown"
        cu = fp.host.cuda_version or "unknown"
        return DoctorCheck(
            name="CUDA Driver & GPU",
            status="OK",
            detail=(
                f"{dev.name} (sm_{dev.sm_int}, {dev.total_vram_mb / 1024.0:.1f} GB VRAM, "
                f"CUDA={cu}, driver={drv})"
            ),
        )
    if fp.effective_backend == "mps":
        return DoctorCheck(
            name="CUDA Driver & GPU",
            status="WARN",
            detail="Apple MPS active; CUDA-specific passes (TF32/AMP/CUDA Graphs/Triton) unavailable.",
        )
    err_note = f" ({fp.gpu.probe_errors[0]})" if fp.gpu.probe_errors else ""
    return DoctorCheck(
        name="CUDA Driver & GPU",
        status="MISSING",
        detail=f"No CUDA GPU detected (backend={fp.effective_backend}){err_note}.",
        remediation=(
            "In Google Colab: Runtime -> Change runtime type -> T4/L4/A100 GPU; "
            "or install CUDA-enabled PyTorch."
        ),
    )


def _check_pytorch(fp: HardwareFingerprint) -> DoctorCheck:
    ver = fp.host.torch_version
    if not ver:
        return DoctorCheck(
            name="PyTorch & torch.compile",
            status="MISSING",
            detail="PyTorch is not installed in the active environment.",
            remediation="Install PyTorch >= 2.2 (e.g. `pip install torch`).",
        )
    parts = ver.split("+", 1)[0].split(".")
    major = int(parts[0]) if parts and parts[0].isdigit() else 0
    if major < 2:
        return DoctorCheck(
            name="PyTorch & torch.compile",
            status="WARN",
            detail=f"PyTorch {ver} (< 2.0): torch.compile & SDPA accelerations unavailable.",
            remediation="Upgrade to PyTorch >= 2.2: `pip install -U torch`.",
        )
    return DoctorCheck(
        name="PyTorch & torch.compile",
        status="OK",
        detail=f"PyTorch {ver} (Python {fp.host.python_version}; Inductor/SDPA supported).",
    )


def _check_triton(fp: HardwareFingerprint) -> DoctorCheck:
    tr = fp.host.triton_version
    if tr:
        return DoctorCheck(
            name="Triton Compiler",
            status="OK",
            detail=f"triton {tr} installed (Inductor fusion & custom Triton kernels available).",
        )
    if fp.effective_backend == "cuda":
        return DoctorCheck(
            name="Triton Compiler",
            status="MISSING",
            detail="Triton is not installed; Inductor compilation and custom Triton kernels disabled.",
            remediation="Install Triton: `pip install triton` (Linux CUDA).",
        )
    return DoctorCheck(
        name="Triton Compiler",
        status="WARN",
        detail="Triton not installed (optional outside Linux CUDA).",
    )


def _check_shm(fp: HardwareFingerprint) -> DoctorCheck:
    total = fp.host.shm_total_mb
    free = fp.host.shm_free_mb
    if total is None:
        return DoctorCheck(
            name="/dev/shm Shared Memory",
            status="WARN",
            detail="/dev/shm size unavailable on this OS; DataLoader workers will stay conservative.",
        )
    free_txt = f", free={free:.0f} MB" if free is not None else ""
    if total < 1024.0:
        return DoctorCheck(
            name="/dev/shm Shared Memory",
            status="WARN",
            detail=f"Low /dev/shm ({total:.0f} MB{free_txt}); multi-worker DataLoader disabled automatically.",
            remediation="In Docker launch with `--shm-size=4g` (or rely on ptopt auto num_workers=0 guard).",
        )
    if total < 2048.0:
        return DoctorCheck(
            name="/dev/shm Shared Memory",
            status="WARN",
            detail=f"Moderate /dev/shm ({total:.0f} MB{free_txt}); DataLoader workers capped at 2.",
            remediation="Increase `/dev/shm` (>= 2 GB) for full multi-worker DataLoader throughput.",
        )
    return DoctorCheck(
        name="/dev/shm Shared Memory",
        status="OK",
        detail=f"Healthy /dev/shm (total={total:.0f} MB{free_txt}).",
    )


def _check_host_allocators(fp: HardwareFingerprint) -> DoctorCheck:
    jem = fp.host.jemalloc_path
    tcm = fp.host.tcmalloc_path
    if jem or tcm:
        active = f"jemalloc={jem}" if jem else f"tcmalloc={tcm}"
        return DoctorCheck(name="Host Memory Allocator", status="OK", detail=f"Found {active}.")
    if fp.host.os_name.lower() == "linux":
        return DoctorCheck(
            name="Host Memory Allocator",
            status="MISSING",
            detail="Neither libjemalloc nor libtcmalloc found (optional CPU malloc speedup).",
            remediation="Optional on Ubuntu/Colab: `apt-get update && apt-get install -y libjemalloc2`.",
        )
    return DoctorCheck(
        name="Host Memory Allocator",
        status="WARN",
        detail="jemalloc/tcmalloc preload check skipped on non-Linux OS.",
    )


def _check_dir_writable(name: str, target_path: Path, remediation: str) -> DoctorCheck:
    """Probe write permissions on `target_path` (or its closest existing parent) safely."""
    resolved = target_path.expanduser().resolve()
    probe_dir = resolved if resolved.is_dir() else None
    if probe_dir is None:
        for parent in resolved.parents:
            if parent.exists():
                probe_dir = parent if parent.is_dir() else None
                break
    if probe_dir is None:
        return DoctorCheck(
            name=name,
            status="MISSING",
            detail=f"No existing parent directory for '{resolved}'.",
            remediation=remediation,
        )
    try:
        with tempfile.NamedTemporaryFile(dir=probe_dir, prefix=".ptopt_probe_", delete=True):
            pass
        state = "exists & writable" if resolved.is_dir() else f"parent '{probe_dir}' writable"
        return DoctorCheck(name=name, status="OK", detail=f"{resolved} ({state}).")
    except OSError as exc:
        return DoctorCheck(
            name=name,
            status="MISSING",
            detail=f"{resolved} is not writable ({exc}).",
            remediation=remediation,
        )


def run_doctor(
    *,
    fp: HardwareFingerprint | None = None,
    env: Mapping[str, str] | None = None,
    cache_dir: str | Path | None = None,
    report_dir: str | Path | None = None,
) -> DoctorReport:
    """Run all hardware, runtime, and filesystem checks without mutating environment."""
    env_map = os.environ if env is None else env
    active_fp = fp if fp is not None else collect_fingerprint(env=env_map)

    default_cache = env_map.get("TORCHINDUCTOR_CACHE_DIR") or OptimizeConfig().compile.cache_dir
    resolved_cache = Path(cache_dir if cache_dir is not None else default_cache)
    resolved_report = Path(report_dir if report_dir is not None else "report")

    key_info = detect_runtime_cache_key(
        gpu_arch=f"sm_{active_fp.sm_int}" if active_fp.sm_int > 0 else active_fp.effective_backend,
        torch_version=active_fp.host.torch_version or "unknown",
        triton_version=active_fp.host.triton_version or "none",
    )

    checks = (
        _check_cuda(active_fp),
        _check_pytorch(active_fp),
        _check_triton(active_fp),
        _check_shm(active_fp),
        _check_host_allocators(active_fp),
        _check_dir_writable(
            "TORCHINDUCTOR_CACHE_DIR",
            resolved_cache,
            "Set a writable cache directory via `--override compile.cache_dir=/tmp/ptopt_cache`.",
        ),
        _check_dir_writable(
            "Report Output Directory",
            resolved_report,
            "Pass a writable `--save-report /tmp/ptopt_reports` directory.",
        ),
    )

    rec_prof, rec_reason = recommend_profile(active_fp)
    return DoctorReport(
        fingerprint=active_fp,
        checks=checks,
        recommended_profile=rec_prof,
        recommendation_reason=rec_reason,
        cache_dir=resolved_cache.expanduser().resolve(),
        report_dir=resolved_report.expanduser().resolve(),
        runtime_cache_key=key_info.cache_key,
    )


def format_doctor_text(report: DoctorReport) -> str:
    """Render DoctorReport as structured plain text for terminal or report files."""
    sep = "=" * 78
    lines = [
        sep,
        "ptopt --doctor | ENVIRONMENT DIAGNOSTICS & PROFILE RECOMMENDATION",
        sep,
        f"Backend            : {report.fingerprint.effective_backend}",
        f"Host               : {report.fingerprint.host.os_name} {report.fingerprint.host.cpu_arch} "
        f"(Colab={report.fingerprint.is_colab})",
        f"Runtime Cache Key  : {report.runtime_cache_key}",
        "",
        "[Diagnostic Checks]",
        "-" * 78,
    ]
    for chk in report.checks:
        lines.append(f"[{chk.status:<7}] {chk.name}: {chk.detail}")
        if chk.remediation:
            lines.append(f"          Fix -> {chk.remediation}")

    lines.extend(
        [
            "",
            "[Recommended Profile]",
            "-" * 78,
            f"Profile : {report.recommended_profile}",
            f"Advice  : {report.recommendation_reason}",
            "",
            "[Summary]",
            "-" * 78,
            f"OK: {report.ok_count} | WARN: {report.warn_count} | MISSING: {report.missing_count}",
            sep,
        ]
    )
    return "\n".join(lines) + "\n"
