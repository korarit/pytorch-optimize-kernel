"""Config Schema v0 -> v1 auto-migrator and runtime cache-key validation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
import subprocess
import sys
import warnings as _warnings
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.config.schema import (
    SCHEMA_VERSION,
    VALID_AMP_DTYPES,
    VALID_COMPILE_MODES,
    VALID_PROFILES,
    ConfigValidationError,
    OptimizeConfig,
)

CACHE_META_FILENAME = ".ptopt_cache_meta.json"

_PROFILE_ALIASES: dict[str, str] = {
    "strict": "supersafe",
    "fp32": "supersafe",
    "exact": "supersafe",
    "conservative": "safe",
    "tf32": "safe",
    "default": "balanced",
    "auto": "balanced",
    "standard": "balanced",
    "fast": "aggressive",
    "max": "aggressive",
    "turbo": "aggressive",
    "performance": "aggressive",
}

_AMP_DTYPE_ALIASES: dict[str, str] = {
    "float32": "fp32",
    "f32": "fp32",
    "float16": "fp16",
    "f16": "fp16",
    "half": "fp16",
    "bfloat16": "bf16",
    "bf": "bf16",
}

_COMPILE_MODE_ALIASES: dict[str, str] = {
    "reduce_overhead": "reduce-overhead",
    "max_autotune": "max-autotune",
    "inductor": "default",
    "eager": "default",
}

# Flat legacy v0 keys -> Schema v1 (section, field) targets.
_LEGACY_FLAT_MAP: dict[str, tuple[str, str]] = {
    "tf32": ("runtime", "tf32"),
    "allow_tf32": ("runtime", "tf32"),
    "cudnn_benchmark": ("runtime", "cudnn_benchmark"),
    "alloc_conf": ("runtime", "alloc_conf"),
    "cuda_alloc_conf": ("runtime", "alloc_conf"),
    "omp_threads": ("runtime", "omp_threads"),
    "jemalloc": ("runtime", "jemalloc"),
    "use_amp": ("amp", "enabled"),
    "mixed_precision": ("amp", "enabled"),
    "amp_enabled": ("amp", "enabled"),
    "amp_dtype": ("amp", "dtype"),
    "precision": ("amp", "dtype"),
    "grad_scaler": ("amp", "grad_scaler"),
    "use_grad_scaler": ("amp", "grad_scaler"),
    "use_compile": ("compile", "enabled"),
    "torch_compile": ("compile", "enabled"),
    "compile_enabled": ("compile", "enabled"),
    "compile_mode": ("compile", "mode"),
    "dynamic_shapes": ("compile", "dynamic"),
    "compile_dynamic": ("compile", "dynamic"),
    "cache_dir": ("compile", "cache_dir"),
    "inductor_cache_dir": ("compile", "cache_dir"),
    "fused_optimizer": ("optimizer", "fused"),
    "fused_adam": ("optimizer", "fused"),
    "foreach_optimizer": ("optimizer", "foreach"),
    "flash_attention": ("attention", "flash_sdp"),
    "flash_sdp": ("attention", "flash_sdp"),
    "mem_efficient_sdp": ("attention", "mem_efficient_sdp"),
    "math_sdp": ("attention", "math_sdp"),
    "auto_tune_dataloader": ("dataloader", "auto_tune"),
    "dataloader_auto_tune": ("dataloader", "auto_tune"),
    "pin_memory": ("dataloader", "pin_memory"),
    "persistent_workers": ("dataloader", "persistent_workers"),
    "num_workers": ("dataloader", "num_workers"),
    "prefetch_factor": ("dataloader", "prefetch_factor"),
    "inference_mode": ("inference", "inference_mode"),
    "cuda_graphs": ("inference", "cuda_graphs"),
    "use_cuda_graphs": ("inference", "cuda_graphs"),
    "custom_kernels": ("kernels", "enabled"),
    "triton_kernels": ("kernels", "enabled"),
    "kernels_allow_list": ("kernels", "allow_list"),
}

_SECTION_FIELD_ALIASES: dict[str, dict[str, str]] = {
    "runtime": {
        "allow_tf32": "tf32",
        "cuda_alloc_conf": "alloc_conf",
    },
    "amp": {
        "use_amp": "enabled",
        "mixed_precision": "enabled",
        "precision": "dtype",
        "use_grad_scaler": "grad_scaler",
    },
    "compile": {
        "use_compile": "enabled",
        "torch_compile": "enabled",
        "dynamic_shapes": "dynamic",
        "inductor_cache_dir": "cache_dir",
    },
    "optimizer": {
        "fused_optimizer": "fused",
        "foreach_optimizer": "foreach",
    },
    "attention": {
        "flash_attention": "flash_sdp",
    },
    "inference": {
        "use_cuda_graphs": "cuda_graphs",
    },
    "kernels": {
        "triton": "enabled",
        "ops": "allow_list",
    },
}

_DEPRECATED_LEGACY_KEYS = frozenset({
    "version",
    "config_version",
    "backend",
    "quantization",
    "int8",
    "fp8_quantization",
    "experimental",
})


@dataclass(frozen=True)
class ConfigMigrationResult:
    """Result of migrating a legacy or v0 config mapping into Schema v1."""

    config: OptimizeConfig
    source_version: int | str | None
    target_version: int
    migrated: bool
    changes: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class CacheKeyInfo:
    """Deterministic cache identity bound to (gpu_arch, torch_version, triton_version)."""

    gpu_arch: str
    torch_version: str
    triton_version: str
    cache_key: str

    def to_dict(self) -> dict[str, str]:
        return {
            "gpu_arch": self.gpu_arch,
            "torch_version": self.torch_version,
            "triton_version": self.triton_version,
            "cache_key": self.cache_key,
        }


@dataclass(frozen=True)
class CacheValidationResult:
    """Validated Inductor/Triton cache directory paths and mismatch diagnostics."""

    cache_dir: Path
    triton_dir: Path
    key_info: CacheKeyInfo
    reused: bool
    rotated: bool
    warnings: tuple[str, ...]


def _parse_source_version(raw_ver: Any) -> int | None:
    if raw_ver is None:
        return None
    if isinstance(raw_ver, bool):
        raise ConfigValidationError(f"Invalid schema_version type: {raw_ver!r}.")
    if isinstance(raw_ver, int):
        return raw_ver
    if isinstance(raw_ver, str):
        cleaned = raw_ver.strip().lower().lstrip("v")
        if cleaned.isdigit():
            return int(cleaned)
    raise ConfigValidationError(f"Unsupported schema_version value: {raw_ver!r}.")


def _set_nested(
    target: dict[str, Any],
    section: str,
    field_name: str,
    value: Any,
    changes: list[str],
    note: str,
) -> None:
    sec_dict = target.setdefault(section, {})
    if field_name not in sec_dict:
        sec_dict[field_name] = deepcopy(value)
        changes.append(note)


def _normalize_value(section: str, field_name: str, value: Any, warnings: list[str], changes: list[str]) -> Any:
    if section == "amp" and field_name == "dtype" and isinstance(value, str):
        low = value.strip().lower()
        mapped = _AMP_DTYPE_ALIASES.get(low, low)
        if mapped != value:
            changes.append(f"Normalized amp.dtype '{value}' -> '{mapped}'.")
            warnings.append(f"Legacy amp.dtype '{value}' migrated to Schema v1 '{mapped}'.")
        if mapped not in VALID_AMP_DTYPES:
            raise ConfigValidationError(f"Invalid amp.dtype '{value}'.")
        return mapped

    if section == "compile" and field_name == "mode" and isinstance(value, str):
        low = value.strip().lower()
        mapped = _COMPILE_MODE_ALIASES.get(low, low)
        if mapped != value:
            changes.append(f"Normalized compile.mode '{value}' -> '{mapped}'.")
            warnings.append(f"Legacy compile.mode '{value}' migrated to Schema v1 '{mapped}'.")
        if mapped not in VALID_COMPILE_MODES:
            raise ConfigValidationError(f"Invalid compile.mode '{value}'.")
        return mapped

    if section == "runtime" and field_name == "alloc_conf" and isinstance(value, bool):
        mapped = "expandable_segments:True" if value else None
        warnings.append(f"Legacy bool runtime.alloc_conf={value} migrated to {mapped!r}.")
        changes.append(f"Converted bool runtime.alloc_conf -> {mapped!r}.")
        return mapped

    return deepcopy(value)


def migrate_config_dict(
    raw: Mapping[str, Any],
    *,
    backend: str | None = None,
    emit_warnings: bool = True,
) -> ConfigMigrationResult:
    """Migrate a v0 / legacy config mapping into a validated Schema v1 OptimizeConfig."""
    if not isinstance(raw, Mapping):
        raise ConfigValidationError("Config payload must be a mapping.")

    data = deepcopy(dict(raw))
    raw_ver = data.get("schema_version")
    if raw_ver is None and "version" in data:
        raw_ver = data["version"]
    if raw_ver is None and "config_version" in data:
        raw_ver = data["config_version"]

    parsed_ver = _parse_source_version(raw_ver)
    if parsed_ver is not None and parsed_ver > SCHEMA_VERSION:
        raise ConfigValidationError(
            f"Unsupported future schema_version={parsed_ver}; max supported is {SCHEMA_VERSION}."
        )

    changes: list[str] = []
    warnings: list[str] = []
    is_legacy = parsed_ver is None or parsed_ver < SCHEMA_VERSION

    normalized: dict[str, Any] = {"schema_version": SCHEMA_VERSION}
    if parsed_ver is None:
        warnings.append(f"Missing 'schema_version'; auto-migrating legacy config to schema_version={SCHEMA_VERSION}.")
        changes.append(f"Set schema_version={SCHEMA_VERSION}.")
    elif parsed_ver < SCHEMA_VERSION:
        warnings.append(f"Legacy schema_version={parsed_ver} auto-migrated to schema_version={SCHEMA_VERSION}.")
        changes.append(f"Upgraded schema_version {parsed_ver} -> {SCHEMA_VERSION}.")

    # Profile handling + legacy profile aliases.
    if "profile" in data:
        prof = str(data["profile"]).strip().lower()
        mapped_prof = _PROFILE_ALIASES.get(prof, prof)
        if mapped_prof != data["profile"]:
            warnings.append(f"Legacy profile '{data['profile']}' migrated to '{mapped_prof}'.")
            changes.append(f"Mapped profile '{data['profile']}' -> '{mapped_prof}'.")
        if mapped_prof not in VALID_PROFILES:
            raise ConfigValidationError(f"Invalid profile '{data['profile']}' after migration.")
        normalized["profile"] = mapped_prof

    known_sections = {"runtime", "amp", "compile", "optimizer", "attention", "dataloader", "inference", "kernels"}

    # Process nested sections first so explicit nested keys take precedence over legacy flat shims.
    for sec in sorted(known_sections):
        if sec not in data:
            continue
        sec_val = data[sec]
        if sec == "amp" and isinstance(sec_val, bool):
            is_legacy = True
            _set_nested(normalized, "amp", "enabled", sec_val, changes, f"Mapped legacy bool amp={sec_val} -> amp.enabled.")
            warnings.append(f"Legacy scalar 'amp={sec_val}' migrated to 'amp.enabled'.")
            continue
        if sec == "compile" and isinstance(sec_val, bool):
            is_legacy = True
            _set_nested(
                normalized,
                "compile",
                "enabled",
                sec_val,
                changes,
                f"Mapped legacy bool compile={sec_val} -> compile.enabled.",
            )
            warnings.append(f"Legacy scalar 'compile={sec_val}' migrated to 'compile.enabled'.")
            continue
        if sec == "optimizer" and isinstance(sec_val, (str, bool)):
            is_legacy = True
            is_fused = sec_val if isinstance(sec_val, bool) else str(sec_val).strip().lower() in {"fused", "adamw_fused"}
            _set_nested(
                normalized,
                "optimizer",
                "fused",
                is_fused,
                changes,
                f"Mapped legacy optimizer={sec_val!r} -> optimizer.fused={is_fused}.",
            )
            warnings.append(f"Legacy scalar 'optimizer={sec_val!r}' migrated to 'optimizer.fused={is_fused}'.")
            continue
        if not isinstance(sec_val, Mapping):
            raise ConfigValidationError(f"Section '{sec}' must be a mapping.")

        aliases = _SECTION_FIELD_ALIASES.get(sec, {})
        sec_out = normalized.setdefault(sec, {})
        for k, v in sec_val.items():
            if k in aliases:
                is_legacy = True
                target_field = aliases[k]
                if target_field not in sec_out:
                    sec_out[target_field] = _normalize_value(sec, target_field, v, warnings, changes)
                    warnings.append(f"Deprecated key '{sec}.{k}' migrated to '{sec}.{target_field}'.")
                    changes.append(f"Renamed {sec}.{k} -> {sec}.{target_field}.")
            else:
                sec_out[k] = _normalize_value(sec, k, v, warnings, changes)

    # Process remaining top-level keys (flat v0 shims or deprecated keys).
    for key, val in data.items():
        if key in {"schema_version", "profile"} or key in known_sections:
            continue
        if key in _LEGACY_FLAT_MAP:
            is_legacy = True
            sec, fld = _LEGACY_FLAT_MAP[key]
            conv = _normalize_value(sec, fld, val, warnings, changes)
            _set_nested(normalized, sec, fld, conv, changes, f"Moved legacy key '{key}' -> '{sec}.{fld}'.")
            warnings.append(f"Legacy top-level key '{key}' migrated to '{sec}.{fld}'.")
        elif key in _DEPRECATED_LEGACY_KEYS:
            is_legacy = True
            if key not in {"version", "config_version"}:
                warnings.append(f"Deprecated legacy key '{key}' ignored during Schema v1 migration.")
                changes.append(f"Dropped deprecated key '{key}'.")
        else:
            if is_legacy:
                warnings.append(f"Unknown legacy key '{key}' ignored during Schema v1 migration.")
                changes.append(f"Dropped unknown legacy key '{key}'.")
            else:
                raise ConfigValidationError(f"Unknown top-level config keys: [{key!r}]")

    # Remediate known incompatible v0 combinations with explicit warnings before strict validation.
    amp_sec = normalized.get("amp", {})
    if isinstance(amp_sec, dict):
        if amp_sec.get("dtype") == "bf16" and amp_sec.get("grad_scaler") is True:
            amp_sec["grad_scaler"] = False
            warnings.append("Auto-disabled amp.grad_scaler for amp.dtype='bf16' (unsupported in Schema v1).")
            changes.append("Set amp.grad_scaler=False for bf16.")
        elif amp_sec.get("enabled") is False and amp_sec.get("grad_scaler") is True:
            amp_sec["grad_scaler"] = False
            warnings.append("Auto-disabled amp.grad_scaler because amp.enabled=False.")
            changes.append("Set amp.grad_scaler=False when AMP disabled.")

    dl_sec = normalized.get("dataloader", {})
    if isinstance(dl_sec, dict) and dl_sec.get("num_workers") == 0:
        if dl_sec.get("persistent_workers") is True:
            dl_sec["persistent_workers"] = False
            warnings.append("Auto-disabled dataloader.persistent_workers because num_workers=0.")
            changes.append("Set dataloader.persistent_workers=False for num_workers=0.")
        if dl_sec.get("prefetch_factor") is not None:
            dl_sec["prefetch_factor"] = None
            warnings.append("Reset dataloader.prefetch_factor=None because num_workers=0.")
            changes.append("Set dataloader.prefetch_factor=None for num_workers=0.")

    cfg = OptimizeConfig.from_dict(normalized)
    if backend is not None:
        from src.config.schema import validate_config

        validate_config(cfg, backend=backend)

    if emit_warnings:
        for msg in warnings:
            _warnings.warn(msg, UserWarning, stacklevel=2)

    return ConfigMigrationResult(
        config=cfg,
        source_version=raw_ver,
        target_version=SCHEMA_VERSION,
        migrated=bool(changes),
        changes=tuple(changes),
        warnings=tuple(warnings),
    )


def load_and_migrate_config(
    path: str | Path,
    *,
    backend: str | None = None,
    emit_warnings: bool = True,
) -> ConfigMigrationResult:
    """Read a JSON config file and auto-migrate legacy v0 schemas to Schema v1."""
    raw_text = Path(path).read_text(encoding="utf-8")
    payload = json.loads(raw_text)
    if not isinstance(payload, Mapping):
        raise ConfigValidationError(f"Config file '{path}' must decode to a JSON object.")
    return migrate_config_dict(payload, backend=backend, emit_warnings=emit_warnings)


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("_") or "none"


def build_runtime_cache_key(
    gpu_arch: str,
    torch_version: str,
    triton_version: str | None = None,
) -> CacheKeyInfo:
    """Build deterministic cache key bound strictly to (gpu_arch, torch_version, triton_version)."""
    arch_norm = (gpu_arch or "cpu").strip().lower()
    torch_norm = (torch_version or "unknown").strip()
    triton_norm = (triton_version or "none").strip()

    canonical = json.dumps(
        {
            "gpu_arch": arch_norm,
            "torch_version": torch_norm,
            "triton_version": triton_norm,
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    key = f"{_slug(arch_norm)}-pt{_slug(torch_norm)}-tr{_slug(triton_norm)}-{digest}"
    return CacheKeyInfo(
        gpu_arch=arch_norm,
        torch_version=torch_norm,
        triton_version=triton_norm,
        cache_key=key,
    )


def _pkg_version_safe(pkg: str) -> str | None:
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return None


def _probe_arch_via_nvidia_smi() -> str | None:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=1.5,
            check=False,
        )
        if proc.returncode != 0:
            return None
        first_line = next((ln.strip() for ln in proc.stdout.splitlines() if ln.strip()), "")
        if not first_line or "." not in first_line:
            return None
        major_s, minor_s = first_line.split(".", 1)
        if major_s.isdigit() and minor_s.isdigit():
            return f"sm_{int(major_s)}{int(minor_s)}"
    except Exception:  # noqa: BLE001
        return None
    return None


def detect_runtime_cache_key(
    torch_mod: Any = None,
    *,
    gpu_arch: str | None = None,
    torch_version: str | None = None,
    triton_version: str | None = None,
) -> CacheKeyInfo:
    """Resolve active (gpu_arch, torch_version, triton_version) without eager torch import."""
    tmod = torch_mod if torch_mod is not None else sys.modules.get("torch")

    resolved_torch = torch_version
    if resolved_torch is None:
        if tmod is not None and getattr(tmod, "__version__", None):
            resolved_torch = str(tmod.__version__)
        else:
            resolved_torch = _pkg_version_safe("torch") or "unknown"

    resolved_triton = triton_version
    if resolved_triton is None:
        triton_mod = sys.modules.get("triton")
        if triton_mod is not None and getattr(triton_mod, "__version__", None):
            resolved_triton = str(triton_mod.__version__)
        else:
            resolved_triton = _pkg_version_safe("triton") or "none"

    resolved_arch = gpu_arch
    if resolved_arch is None:
        if tmod is not None:
            try:
                cuda_mod = getattr(tmod, "cuda", None)
                if cuda_mod is not None and cuda_mod.is_available():
                    major, minor = cuda_mod.get_device_capability(cuda_mod.current_device())
                    resolved_arch = f"sm_{int(major)}{int(minor)}"
                else:
                    mps_mod = getattr(getattr(tmod, "backends", None), "mps", None)
                    resolved_arch = "mps" if (mps_mod is not None and mps_mod.is_available()) else "cpu"
            except Exception:  # noqa: BLE001
                resolved_arch = "cpu"
        else:
            resolved_arch = _probe_arch_via_nvidia_smi() or "cpu"

    return build_runtime_cache_key(resolved_arch, resolved_torch, resolved_triton)


def _read_cache_meta(meta_path: Path) -> dict[str, Any] | None:
    if not meta_path.is_file():
        return None
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:  # noqa: BLE001
        return None


def validate_or_scope_cache_dir(
    base_dir: str | Path,
    key_info: CacheKeyInfo,
    *,
    write_meta: bool = True,
    emit_warnings: bool = True,
) -> CacheValidationResult:
    """Validate base_dir against key_info; isolate to key-scoped subdir on arch/version mismatch."""
    base_path = Path(base_dir).expanduser().resolve()
    meta_path = base_path / CACHE_META_FILENAME
    existing = _read_cache_meta(meta_path)

    warnings: list[str] = []
    reused = False
    rotated = False
    active_dir = base_path

    if existing is not None:
        prev_key = existing.get("cache_key")
        if prev_key == key_info.cache_key:
            reused = True
        else:
            rotated = True
            active_dir = base_path / key_info.cache_key
            msg = (
                f"Inductor/Triton cache metadata mismatch in '{base_path}': "
                f"cached=({existing.get('gpu_arch')}, torch={existing.get('torch_version')}, "
                f"triton={existing.get('triton_version')}) vs active=({key_info.gpu_arch}, "
                f"torch={key_info.torch_version}, triton={key_info.triton_version}). "
                f"Refusing stale cache reuse; scoping active cache to '{active_dir}'."
            )
            warnings.append(msg)
            if emit_warnings:
                _warnings.warn(msg, UserWarning, stacklevel=2)

            scoped_meta = _read_cache_meta(active_dir / CACHE_META_FILENAME)
            if scoped_meta is not None and scoped_meta.get("cache_key") == key_info.cache_key:
                reused = True

    triton_dir = active_dir / "triton"
    if write_meta:
        active_dir.mkdir(parents=True, exist_ok=True)
        triton_dir.mkdir(parents=True, exist_ok=True)
        active_meta_path = active_dir / CACHE_META_FILENAME
        if not active_meta_path.is_file() or (_read_cache_meta(active_meta_path) or {}).get("cache_key") != key_info.cache_key:
            active_meta_path.write_text(json.dumps(key_info.to_dict(), indent=2), encoding="utf-8")

    return CacheValidationResult(
        cache_dir=active_dir,
        triton_dir=triton_dir,
        key_info=key_info,
        reused=reused,
        rotated=rotated,
        warnings=tuple(warnings),
    )
