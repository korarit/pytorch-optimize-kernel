#!/usr/bin/env python3
"""Phase 0 environment and hardware discovery script for Google Colab and local hosts.

Generates a detailed plain-text report in `report/HHMMSS_YYYYMMDD-phase-0-env.txt`
and optionally writes `ai-docs-dev/env-baseline.md`.
"""

from __future__ import annotations

import argparse
import ctypes.util
import datetime
import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _run_cmd(cmd: list[str], timeout: float = 5.0) -> str | None:
    """Run a command safely and return stripped stdout, or None on failure."""
    try:
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return None
    return None


def _get_pkg_version(name: str) -> str | None:
    """Return installed package version via importlib.metadata, or None."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _detect_colab() -> dict[str, Any]:
    """Detect whether execution is inside Google Colab."""
    colab_tag = os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_BACKEND_VERSION")
    colab_gpu = os.environ.get("COLAB_GPU")
    has_content_dir = Path("/content").is_dir()
    has_colab_pkg = _get_pkg_version("google-colab") is not None
    is_colab = bool(colab_tag or colab_gpu or has_colab_pkg)
    return {
        "is_colab": is_colab,
        "colab_release_tag": colab_tag,
        "colab_gpu_env": colab_gpu,
        "has_content_dir": has_content_dir,
        "google_colab_pkg": has_colab_pkg,
    }


def _parse_proc_cpuinfo() -> dict[str, Any]:
    """Parse /proc/cpuinfo on Linux for model name, physical cores, and ISA flags."""
    cpuinfo_path = Path("/proc/cpuinfo")
    if not cpuinfo_path.is_file():
        return {"model_name": platform.processor() or "unknown", "physical_cores": None, "isa_flags": []}

    model_name = "unknown"
    flags: set[str] = set()
    physical_pairs: set[tuple[str, str]] = set()
    phys_id = "0"
    core_id: str | None = None

    for line in cpuinfo_path.read_text(errors="replace").splitlines():
        if not line.strip():
            if core_id is not None:
                physical_pairs.add((phys_id, core_id))
            phys_id, core_id = "0", None
            continue
        if ":" not in line:
            continue
        key, val = (part.strip() for part in line.split(":", 1))
        if key in ("model name", "Model") and model_name == "unknown":
            model_name = val
        elif key == "physical id":
            phys_id = val
        elif key == "core id":
            core_id = val
        elif key in ("flags", "Features"):
            flags.update(val.split())

    if core_id is not None:
        physical_pairs.add((phys_id, core_id))

    interesting = [f for f in ("avx2", "avx512f", "avx512_bf16", "amx_bf16", "amx_tile", "asimd", "neon") if f in flags]
    return {
        "model_name": model_name,
        "physical_cores": len(physical_pairs) if physical_pairs else None,
        "isa_flags": interesting,
    }


def _read_meminfo_mb() -> dict[str, float | None]:
    """Read MemTotal and MemAvailable from /proc/meminfo in MB."""
    meminfo_path = Path("/proc/meminfo")
    if not meminfo_path.is_file():
        return {"total_ram_mb": None, "available_ram_mb": None}

    values_kb: dict[str, int] = {}
    for line in meminfo_path.read_text(errors="replace").splitlines():
        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        parts = rest.strip().split()
        if parts and parts[0].isdigit():
            values_kb[key] = int(parts[0])

    total = values_kb.get("MemTotal")
    avail = values_kb.get("MemAvailable")
    return {
        "total_ram_mb": round(total / 1024.0, 1) if total is not None else None,
        "available_ram_mb": round(avail / 1024.0, 1) if avail is not None else None,
    }


def _read_shm_mb() -> dict[str, float | None]:
    """Return /dev/shm total and free size in MB."""
    shm_path = Path("/dev/shm")
    if not shm_path.exists():
        return {"shm_total_mb": None, "shm_free_mb": None}
    try:
        st = os.statvfs(shm_path)
        total_mb = (st.f_frsize * st.f_blocks) / (1024.0 * 1024.0)
        free_mb = (st.f_frsize * st.f_bavail) / (1024.0 * 1024.0)
        return {"shm_total_mb": round(total_mb, 1), "shm_free_mb": round(free_mb, 1)}
    except OSError:
        return {"shm_total_mb": None, "shm_free_mb": None}


def _find_allocator_libs() -> dict[str, str | None]:
    """Check for jemalloc and tcmalloc shared libraries."""
    found: dict[str, str | None] = {"jemalloc": None, "tcmalloc": None}
    candidates = {
        "jemalloc": ["jemalloc", "/usr/lib/x86_64-linux-gnu/libjemalloc.so.2", "/usr/lib/aarch64-linux-gnu/libjemalloc.so.2"],
        "tcmalloc": ["tcmalloc", "tcmalloc_minimal", "/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4"],
    }
    for key, probes in candidates.items():
        for probe in probes:
            if probe.startswith("/"):
                if Path(probe).exists():
                    found[key] = probe
                    break
            else:
                lib = ctypes.util.find_library(probe)
                if lib:
                    found[key] = lib
                    break
    return found


def _probe_host() -> dict[str, Any]:
    """Collect host OS, CPU, RAM, /dev/shm, and allocator information."""
    cpu_parsed = _parse_proc_cpuinfo()
    mem_info = _read_meminfo_mb()
    shm_info = _read_shm_mb()
    logical_cores = os.cpu_count()
    return {
        "hostname": platform.node(),
        "os": f"{platform.system()} {platform.release()}",
        "cpu_arch": platform.machine(),
        "cpu_model": cpu_parsed["model_name"],
        "physical_cores": cpu_parsed["physical_cores"],
        "logical_cores": logical_cores,
        "isa_flags": cpu_parsed["isa_flags"],
        **mem_info,
        **shm_info,
        "allocators": _find_allocator_libs(),
    }


def _probe_nvidia_smi() -> dict[str, Any]:
    """Query nvidia-smi if present on PATH."""
    if not shutil.which("nvidia-smi"):
        return {"available": False, "driver_version": None, "gpus": []}
    out = _run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )
    if not out:
        return {"available": True, "driver_version": None, "gpus": []}
    gpus = []
    driver_ver = None
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5:
            driver_ver = parts[2]
            gpus.append(
                {
                    "index": parts[0],
                    "name": parts[1],
                    "memory_total_mb": parts[3],
                    "compute_cap": parts[4],
                }
            )
    return {"available": True, "driver_version": driver_ver, "gpus": gpus}


def _probe_torch_and_gpus() -> dict[str, Any]:
    """Probe Python stack, PyTorch, Triton, and GPU capabilities."""
    stack: dict[str, Any] = {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "torch_installed": False,
        "torch_version": None,
        "torch_cuda_version": None,
        "torch_hip_version": None,
        "triton_version": _get_pkg_version("triton"),
        "pytest_version": _get_pkg_version("pytest"),
        "backend": "cpu",
        "cuda_available": False,
        "mps_available": False,
        "device_count": 0,
        "devices": [],
        "sdpa_flags": {},
        "nvidia_smi": _probe_nvidia_smi(),
    }

    try:
        import torch  # noqa: PLC0415
    except ImportError as exc:
        stack["torch_import_error"] = str(exc)
        return stack

    stack["torch_installed"] = True
    stack["torch_version"] = torch.__version__
    stack["torch_cuda_version"] = getattr(torch.version, "cuda", None)
    stack["torch_hip_version"] = getattr(torch.version, "hip", None)

    cuda_avail = bool(torch.cuda.is_available())
    mps_avail = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    stack["cuda_available"] = cuda_avail
    stack["mps_available"] = mps_avail

    if cuda_avail:
        stack["backend"] = "rocm" if stack["torch_hip_version"] else "cuda"
        count = torch.cuda.device_count()
        stack["device_count"] = count
        bf16_ok = bool(torch.cuda.is_bf16_supported()) if hasattr(torch.cuda, "is_bf16_supported") else False
        for idx in range(count):
            props = torch.cuda.get_device_properties(idx)
            sm_int = props.major * 10 + props.minor
            stack["devices"].append(
                {
                    "index": idx,
                    "name": props.name,
                    "compute_capability": f"{props.major}.{props.minor}",
                    "sm": f"sm_{sm_int}",
                    "sm_int": sm_int,
                    "total_vram_mb": round(props.total_memory / (1024.0 * 1024.0), 1),
                    "sm_count": props.multi_processor_count,
                    "bf16_supported": bf16_ok and sm_int >= 80,
                    "tf32_supported": sm_int >= 80,
                    "fp8_supported": sm_int >= 89,
                }
            )
        if hasattr(torch.backends.cuda, "flash_sdp_enabled"):
            stack["sdpa_flags"] = {
                "flash_sdp_enabled": bool(torch.backends.cuda.flash_sdp_enabled()),
                "mem_efficient_sdp_enabled": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
                "math_sdp_enabled": bool(torch.backends.cuda.math_sdp_enabled()),
            }
    elif mps_avail:
        stack["backend"] = "mps"
        stack["device_count"] = 1

    return stack


def _probe_git(repo_root: Path) -> dict[str, str | None]:
    """Read current git branch and short commit hash."""
    if not (repo_root / ".git").exists():
        return {"branch": None, "commit": None}
    branch = _run_cmd(["git", "-C", str(repo_root), "rev-parse", "--abbrev-ref", "HEAD"])
    commit = _run_cmd(["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"])
    return {"branch": branch, "commit": commit}


def _build_recommendations(host: dict[str, Any], stack: dict[str, Any]) -> list[str]:
    """Generate actionable Phase 0 tuning notes for this environment."""
    notes: list[str] = []
    shm = host.get("shm_total_mb")
    if shm is not None and shm < 2048:
        notes.append(f"[WARN] /dev/shm is small ({shm} MB < 2048 MB): DataLoader auto-tuning should cap num_workers.")

    backend = stack.get("backend")
    devices = stack.get("devices", [])
    if backend == "cpu":
        notes.append("[INFO] CPU-only environment detected: GPU CUDA passes will be skipped; CPU thread/allocator rules apply.")
    elif devices:
        dev = devices[0]
        sm_int = dev["sm_int"]
        if sm_int < 80:
            notes.append(
                f"[INFO] GPU {dev['name']} ({dev['sm']}): TF32/BF16 hardware unavailable -> recommend FP16 AMP + GradScaler, Mem-Efficient SDPA."
            )
        elif sm_int < 89:
            notes.append(
                f"[INFO] GPU {dev['name']} ({dev['sm']}): Ampere architecture -> enable TF32, BF16 AMP, Flash SDPA, Fused AdamW."
            )
        else:
            notes.append(
                f"[INFO] GPU {dev['name']} ({dev['sm']}): Ada/Hopper+ architecture -> full TF32, BF16, FP8-capable, Triton Kernels & max-autotune eligible."
            )
        if dev["total_vram_mb"] < 12288:
            notes.append(f"[WARN] VRAM < 12 GB ({dev['total_vram_mb']} MB): prefer expandable_segments:True and conservative compile.")

    if not stack.get("triton_version"):
        notes.append("[WARN] Triton not installed: Phase 3.6 custom Triton kernels will fallback to PyTorch native ops.")
    if not stack.get("pytest_version"):
        notes.append("[WARN] pytest not installed: install pytest before running Phase 1+ test suites.")
    return notes


def format_text_report(
    now: datetime.datetime,
    repo_root: Path,
    colab: dict[str, Any],
    host: dict[str, Any],
    stack: dict[str, Any],
    git_info: dict[str, str | None],
    notes: list[str],
) -> str:
    """Format the collected environment discovery into a readable detailed text report."""
    lines: list[str] = []
    sep = "=" * 78
    sub = "-" * 78

    lines.extend([
        sep,
        "KERNEL-PYTOUCH-OPTIMIZE | PHASE 0: ENVIRONMENT & HARDWARE DISCOVERY REPORT",
        sep,
        f"Timestamp          : {now.strftime('%Y-%m-%d %H:%M:%S %Z') or now.isoformat()}",
        f"Repo Root          : {repo_root}",
        f"Git Branch/Commit  : {git_info.get('branch') or 'N/A'} @ {git_info.get('commit') or 'N/A'}",
        f"Google Colab       : {colab['is_colab']} (tag={colab['colab_release_tag']}, COLAB_GPU={colab['colab_gpu_env']})",
        "",
        "[HOST & MACHINE ARCHITECTURE]",
        sub,
        f"Hostname           : {host['hostname']}",
        f"OS / Kernel        : {host['os']}",
        f"CPU Arch           : {host['cpu_arch']}",
        f"CPU Model          : {host['cpu_model']}",
        f"CPU Cores          : physical={host['physical_cores']}, logical={host['logical_cores']}",
        f"CPU ISA Flags      : {', '.join(host['isa_flags']) if host['isa_flags'] else 'none detected'}",
        f"System RAM (MB)    : total={host['total_ram_mb']}, available={host['available_ram_mb']}",
        f"/dev/shm (MB)      : total={host['shm_total_mb']}, free={host['shm_free_mb']}",
        f"Allocator Libs     : jemalloc={host['allocators']['jemalloc']}, tcmalloc={host['allocators']['tcmalloc']}",
        "",
        "[PYTHON & PYTORCH STACK]",
        sub,
        f"Python             : {stack['python_version']} ({stack['python_executable']})",
        f"PyTorch            : {stack['torch_version'] or 'NOT INSTALLED'}",
        f"Torch CUDA / HIP   : cuda={stack['torch_cuda_version']}, hip={stack['torch_hip_version']}",
        f"Triton             : {stack['triton_version'] or 'NOT INSTALLED'}",
        f"pytest             : {stack['pytest_version'] or 'NOT INSTALLED'}",
        f"Selected Backend   : {stack['backend']} (cuda_avail={stack['cuda_available']}, mps_avail={stack['mps_available']})",
        "",
        "[GPU ARCHITECTURE & CAPABILITIES]",
        sub,
    ])

    if stack["devices"]:
        for dev in stack["devices"]:
            lines.extend([
                f"GPU[{dev['index']}]             : {dev['name']}",
                f"  Compute Cap      : {dev['compute_capability']} ({dev['sm']})",
                f"  VRAM / SMs       : {dev['total_vram_mb']} MB / {dev['sm_count']} SMs",
                f"  Features         : bf16={dev['bf16_supported']}, tf32={dev['tf32_supported']}, fp8={dev['fp8_supported']}",
            ])
        if stack["sdpa_flags"]:
            lines.append(f"SDPA Defaults      : {stack['sdpa_flags']}")
    else:
        lines.append("PyTorch CUDA GPUs  : None detected")

    smi = stack["nvidia_smi"]
    lines.append(f"nvidia-smi         : available={smi['available']}, driver={smi['driver_version']}")
    for gpu in smi.get("gpus", []):
        lines.append(f"  smi GPU[{gpu['index']}]      : {gpu['name']} | {gpu['memory_total_mb']} MB | cc={gpu['compute_cap']}")

    lines.extend([
        "",
        "[PHASE 0 OBSERVATIONS & ARCHITECTURE NOTES]",
        sub,
    ])
    lines.extend(notes if notes else ["[INFO] No special warnings detected."])

    lines.extend([
        "",
        "[VERIFICATION STATUS]",
        sub,
        "STATUS             : PENDING_USER_REVIEW (Run on Google Colab & verify manually)",
        sep,
    ])
    return "\n".join(lines) + "\n"


def format_baseline_markdown(now: datetime.datetime, colab: dict[str, Any], host: dict[str, Any], stack: dict[str, Any], notes: list[str]) -> str:
    """Render a concise Markdown snapshot for ai-docs-dev/env-baseline.md."""
    gpu_summary = (
        ", ".join(f"{d['name']} ({d['sm']}, {d['total_vram_mb']} MB)" for d in stack["devices"])
        if stack["devices"]
        else "None (CPU-only)"
    )
    lines = [
        "# Environment Baseline Snapshot (Phase 0)",
        "",
        f"- **Captured At**: `{now.strftime('%Y-%m-%d %H:%M:%S')}`",
        f"- **Google Colab**: `{colab['is_colab']}`",
        f"- **OS / CPU Arch**: `{host['os']}` / `{host['cpu_arch']}` (`{host['cpu_model']}`)",
        f"- **CPU Cores**: physical=`{host['physical_cores']}`, logical=`{host['logical_cores']}`",
        f"- **RAM / `/dev/shm`**: `{host['total_ram_mb']} MB` / `{host['shm_total_mb']} MB`",
        f"- **Python / PyTorch / Triton**: `{stack['python_version']}` / `{stack['torch_version']}` / `{stack['triton_version']}`",
        f"- **Backend / GPU**: `{stack['backend']}` / `{gpu_summary}`",
        "",
        "## Observations",
        *[f"- {note}" for note in notes],
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 0 Environment Discovery & Text Report Generator")
    parser.add_argument("--report-dir", type=Path, default=None, help="Directory to store HHMMSS_YYYYMMDD-phase-0-env.txt")
    parser.add_argument("--save-baseline", action="store_true", help="Also write snapshot to ai-docs-dev/env-baseline.md")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    report_dir = args.report_dir or (repo_root / "report")
    report_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.datetime.now().astimezone()
    filename = f"{now.strftime('%H%M%S_%Y%m%d')}-phase-0-env.txt"
    report_path = report_dir / filename

    colab = _detect_colab()
    host = _probe_host()
    stack = _probe_torch_and_gpus()
    git_info = _probe_git(repo_root)
    notes = _build_recommendations(host, stack)

    report_text = format_text_report(now, repo_root, colab, host, stack, git_info, notes)
    report_path.write_text(report_text, encoding="utf-8")

    if args.save_baseline:
        baseline_path = repo_root / "ai-docs-dev" / "env-baseline.md"
        baseline_path.write_text(format_baseline_markdown(now, colab, host, stack, notes), encoding="utf-8")

    print(report_text)
    print(f"[SAVED] Text report written to: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
