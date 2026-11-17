"""Explainability formatter and CLI (`--explain`, `--show-config`, `--dump-config`, `--target-arch`)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from src.config import ConfigResolver, ConfigValidationError, ResolutionResult
from src.detect import GPUDeviceInfo, GPUProbeResult, HardwareFingerprint, HostInfo, collect_fingerprint

# Unified preset table shared by --target-arch and --mock-profile.
_MOCK_PRESETS: dict[str, dict[str, Any]] = {
    "t4": {"label": "Colab T4", "backend": "cuda", "name": "Tesla T4", "major": 7, "minor": 5, "vram_mb": 15360.0, "sm_count": 40, "shm_mb": 16384.0, "cores": 2},
    "sm_75": {"label": "Turing sm_75", "backend": "cuda", "name": "Tesla T4 (Mock)", "major": 7, "minor": 5, "vram_mb": 15360.0, "sm_count": 40, "shm_mb": 16384.0, "cores": 2},
    "low_shm_t4": {"label": "Colab T4 Low-SHM", "backend": "cuda", "name": "Tesla T4", "major": 7, "minor": 5, "vram_mb": 15360.0, "sm_count": 40, "shm_mb": 512.0, "cores": 2},
    "a100": {"label": "Colab A100", "backend": "cuda", "name": "NVIDIA A100-SXM4-40GB", "major": 8, "minor": 0, "vram_mb": 40960.0, "sm_count": 108, "shm_mb": 26155.0, "cores": 6},
    "sm_80": {"label": "Ampere sm_80", "backend": "cuda", "name": "NVIDIA A100 (Mock)", "major": 8, "minor": 0, "vram_mb": 40960.0, "sm_count": 108, "shm_mb": 26155.0, "cores": 6},
    "sm_86": {"label": "Ampere sm_86", "backend": "cuda", "name": "NVIDIA RTX 3090 (Mock)", "major": 8, "minor": 6, "vram_mb": 24576.0, "sm_count": 82, "shm_mb": 16384.0, "cores": 8},
    "l4": {"label": "Colab L4", "backend": "cuda", "name": "NVIDIA L4", "major": 8, "minor": 9, "vram_mb": 22563.1, "sm_count": 58, "shm_mb": 26155.0, "cores": 6},
    "sm_89": {"label": "Ada sm_89", "backend": "cuda", "name": "NVIDIA L4 (Mock)", "major": 8, "minor": 9, "vram_mb": 22563.1, "sm_count": 58, "shm_mb": 26155.0, "cores": 6},
    "h100": {"label": "Hopper H100", "backend": "cuda", "name": "NVIDIA H100 80GB HBM3", "major": 9, "minor": 0, "vram_mb": 81920.0, "sm_count": 132, "shm_mb": 32768.0, "cores": 16},
    "sm_90": {"label": "Hopper sm_90", "backend": "cuda", "name": "NVIDIA H100 (Mock)", "major": 9, "minor": 0, "vram_mb": 81920.0, "sm_count": 132, "shm_mb": 32768.0, "cores": 16},
    "cpu": {"label": "CPU-Only", "backend": "cpu", "name": "CPU", "major": 0, "minor": 0, "vram_mb": 0.0, "sm_count": 0, "shm_mb": 16384.0, "cores": 6},
    "mps": {"label": "Apple Silicon MPS", "backend": "mps", "name": "Apple MPS", "major": 0, "minor": 0, "vram_mb": 0.0, "sm_count": 0, "shm_mb": None, "cores": 8},
}


def available_mock_targets() -> list[str]:
    """Return sorted list of supported --target-arch / --mock-profile names."""
    return sorted(_MOCK_PRESETS.keys())


def build_mock_fingerprint(target: str) -> HardwareFingerprint:
    """Construct a HardwareFingerprint from a named GPU arch or preset."""
    key = target.strip().lower()
    if key not in _MOCK_PRESETS:
        raise ConfigValidationError(
            f"Unknown mock target '{target}'. Supported: {available_mock_targets()}"
        )
    spec = _MOCK_PRESETS[key]
    backend = spec["backend"]
    is_mps = backend == "mps"
    host = HostInfo(
        hostname=f"mock-{key}",
        os_name="Darwin" if is_mps else "Linux",
        os_release="6.6.0",
        cpu_arch="arm64" if is_mps else "x86_64",
        cpu_model="Mock CPU",
        physical_cores=spec["cores"],
        logical_cores=spec["cores"] * 2,
        isa_flags=["neon"] if is_mps else ["avx2", "avx512f"],
        total_ram_mb=32768.0,
        available_ram_mb=24000.0,
        shm_total_mb=spec["shm_mb"],
        shm_free_mb=spec["shm_mb"],
        is_colab=(backend == "cuda"),
        python_version="3.13.0",
        torch_version="2.14.1+cu130" if backend == "cuda" else "2.15.0",
        cuda_version="13.0" if backend == "cuda" else None,
        triton_version="3.8.0" if backend == "cuda" else None,
    )
    devices: list[GPUDeviceInfo] = []
    if backend == "cuda":
        sm_int = spec["major"] * 10 + spec["minor"]
        devices.append(
            GPUDeviceInfo(
                index=0,
                name=spec["name"],
                compute_capability=f"{spec['major']}.{spec['minor']}",
                major=spec["major"],
                minor=spec["minor"],
                sm_int=sm_int,
                sm_name=f"sm_{sm_int}",
                total_vram_mb=spec["vram_mb"],
                sm_count=spec["sm_count"],
                bf16_supported=sm_int >= 80,
                tf32_supported=sm_int >= 80,
                fp8_supported=sm_int >= 89,
                flash_sdp_supported=sm_int >= 80,
                mem_efficient_sdp_supported=sm_int >= 70,
            )
        )
    gpu = GPUProbeResult(
        backend=backend,
        device_count=len(devices),
        selected_index=0,
        cuda_visible_devices=None,
        devices=devices,
    )
    return HardwareFingerprint(host=host, gpu=gpu)


def build_config_snapshot_dict(fp: HardwareFingerprint, result: ResolutionResult) -> dict[str, Any]:
    """Return a JSON-serializable snapshot with fingerprint, config, rules, and trace."""
    return {
        "schema_version": result.config.schema_version,
        "hardware_fingerprint": fp.to_dict(),
        "config": result.config.to_dict(),
        "matched_rules": list(result.matched_rules),
        "trace": [entry.to_dict() for entry in result.trace],
    }


def format_explain_text(fp: HardwareFingerprint, result: ResolutionResult) -> str:
    """Format a human-readable explanation of hardware detection, rule trace, and config."""
    sep = "=" * 78
    sub = "-" * 78
    gpu_desc = (
        f"{fp.primary_gpu.name} ({fp.primary_gpu.sm_name}, {fp.primary_gpu.total_vram_mb} MB VRAM)"
        if fp.primary_gpu is not None
        else "None"
    )
    lines = [
        sep,
        f"PTOPT CONFIG EXPLAIN (profile={result.config.profile})",
        sep,
        "[HARDWARE FINGERPRINT]",
        sub,
        f"Backend / Colab    : {fp.effective_backend} (is_colab={fp.is_colab})",
        f"Primary GPU        : {gpu_desc}",
        f"Host OS / Arch     : {fp.host.os_name} / {fp.host.cpu_arch} (cores={fp.host.physical_cores}/{fp.host.logical_cores})",
        f"RAM / /dev/shm     : {fp.host.total_ram_mb} MB / {fp.host.shm_total_mb} MB (low_shm={fp.is_low_shm})",
        f"PyTorch / Triton   : {fp.host.torch_version} / {fp.host.triton_version}",
        "",
        "[MATCHED RULES]",
        sub,
        *(result.matched_rules or ["(none)"]),
        "",
        "[RESOLUTION TRACE]",
        sub,
    ]
    if result.trace:
        lines.extend(
            f"[{t.layer}] {t.rule_name}: {t.field_path} ({t.old_value!r} -> {t.new_value!r}) | {t.reason}"
            for t in result.trace
        )
    else:
        lines.append("(no overrides from base)")

    lines.extend([
        "",
        "[FINAL RESOLVED CONFIG]",
        sub,
        json.dumps(result.config.to_dict(), indent=2),
        sep,
    ])
    return "\n".join(lines) + "\n"


def format_show_config_text(result: ResolutionResult) -> str:
    """Format only the final resolved OptimizeConfig as JSON."""
    return json.dumps(result.config.to_dict(), indent=2) + "\n"


def _parse_override_value(raw: str) -> Any:
    low = raw.strip().lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in ("null", "none"):
        return None
    try:
        return int(raw)
    except ValueError:
        return raw


def _parse_overrides(items: Sequence[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ConfigValidationError(f"Invalid --override '{item}'; expected key=value.")
        key, val = item.split("=", 1)
        parsed[key.strip()] = _parse_override_value(val)
    return parsed


def run_explain_cli(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for --explain, --show-config, and --dump-config."""
    parser = argparse.ArgumentParser(description="Inspect hardware-aware ptopt configuration.")
    parser.add_argument("--profile", default=None, choices=["supersafe", "safe", "balanced", "aggressive"])
    parser.add_argument("--explain", action="store_true", help="Print full hardware + rule trace + config")
    parser.add_argument("--show-config", action="store_true", help="Print only final resolved JSON config")
    parser.add_argument("--dump-config", type=Path, default=None, help="Write full snapshot JSON to path")
    parser.add_argument("--target-arch", default=None, help=f"Mock target arch ({', '.join(available_mock_targets())})")
    parser.add_argument("--mock-profile", default=None, help="Alias for --target-arch (e.g. t4, l4, a100, h100)")
    parser.add_argument("--override", action="append", default=[], help="Override field via section.field=value")
    args = parser.parse_args(argv)

    mock_key = args.mock_profile or args.target_arch
    fp = build_mock_fingerprint(mock_key) if mock_key else collect_fingerprint()
    overrides = _parse_overrides(args.override)
    result = ConfigResolver().resolve(fp, profile=args.profile, overrides=overrides)

    if args.dump_config is not None:
        args.dump_config.parent.mkdir(parents=True, exist_ok=True)
        args.dump_config.write_text(json.dumps(build_config_snapshot_dict(fp, result), indent=2) + "\n", encoding="utf-8")

    if args.show_config and not args.explain:
        print(format_show_config_text(result), end="")
    else:
        print(format_explain_text(fp, result), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_explain_cli(sys.argv[1:]))
