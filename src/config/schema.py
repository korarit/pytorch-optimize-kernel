"""Config Schema v1 dataclasses, defaults, and strict validation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

SCHEMA_VERSION = 1
VALID_PROFILES = frozenset({"supersafe", "safe", "balanced", "aggressive"})
VALID_AMP_DTYPES = frozenset({"fp32", "fp16", "bf16"})
VALID_COMPILE_MODES = frozenset({"default", "reduce-overhead", "max-autotune"})


class ConfigValidationError(ValueError):
    """Raised when an OptimizeConfig dict or object violates Schema v1 rules."""


@dataclass
class RuntimeConfig:
    tf32: bool = False
    cudnn_benchmark: bool = False
    alloc_conf: str | None = "expandable_segments:True"
    omp_threads: int = 0
    jemalloc: bool = False


@dataclass
class AmpConfig:
    enabled: bool = False
    dtype: str = "fp32"
    grad_scaler: bool = False


@dataclass
class CompileConfig:
    enabled: bool = False
    mode: str = "default"
    dynamic: bool | None = False
    cache_dir: str = ".torch_inductor_cache"
    train_cuda_graphs: bool = False


@dataclass
class OptimizerConfig:
    fused: bool = False
    foreach: bool = True


@dataclass
class AttentionConfig:
    flash_sdp: bool = False
    mem_efficient_sdp: bool = True
    math_sdp: bool = True


@dataclass
class DataLoaderConfig:
    auto_tune: bool = False
    pin_memory: bool = False
    persistent_workers: bool = False
    num_workers: int | None = None
    prefetch_factor: int | None = 2


@dataclass
class InferenceConfig:
    inference_mode: bool = False
    cuda_graphs: bool = False


@dataclass
class KernelsConfig:
    enabled: bool = False
    allow_list: list[str] = field(default_factory=list)
    fallback_on_error: bool = True


_SECTION_CLASSES: dict[str, type[Any]] = {
    "runtime": RuntimeConfig,
    "amp": AmpConfig,
    "compile": CompileConfig,
    "optimizer": OptimizerConfig,
    "attention": AttentionConfig,
    "dataloader": DataLoaderConfig,
    "inference": InferenceConfig,
    "kernels": KernelsConfig,
}


@dataclass
class OptimizeConfig:
    """Top-level Schema v1 configuration object."""

    schema_version: int = SCHEMA_VERSION
    profile: str = "balanced"
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    amp: AmpConfig = field(default_factory=AmpConfig)
    compile: CompileConfig = field(default_factory=CompileConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    kernels: KernelsConfig = field(default_factory=KernelsConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def get_field(self, dotted_path: str) -> Any:
        """Read a top-level or `section.field` value by dotted path."""
        parts = dotted_path.split(".")
        if len(parts) == 1 and hasattr(self, parts[0]):
            return getattr(self, parts[0])
        if len(parts) == 2 and parts[0] in _SECTION_CLASSES:
            section = getattr(self, parts[0])
            if hasattr(section, parts[1]):
                return getattr(section, parts[1])
        raise ConfigValidationError(f"Unknown config field path: '{dotted_path}'")

    def set_field(self, dotted_path: str, value: Any) -> Any:
        """Update a field by dotted path and return its previous value."""
        old_val = self.get_field(dotted_path)
        parts = dotted_path.split(".")
        if len(parts) == 1:
            setattr(self, parts[0], deepcopy(value))
        else:
            setattr(getattr(self, parts[0]), parts[1], deepcopy(value))
        return old_val

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "OptimizeConfig":
        """Construct and validate an OptimizeConfig from a nested dictionary."""
        allowed_top = {"schema_version", "profile", *_SECTION_CLASSES}
        unknown_top = set(raw.keys()) - allowed_top
        if unknown_top:
            raise ConfigValidationError(f"Unknown top-level config keys: {sorted(unknown_top)}")

        kwargs: dict[str, Any] = {}
        if "schema_version" in raw:
            kwargs["schema_version"] = raw["schema_version"]
        if "profile" in raw:
            kwargs["profile"] = raw["profile"]

        for sec_name, sec_cls in _SECTION_CLASSES.items():
            if sec_name not in raw:
                continue
            sec_raw = raw[sec_name]
            if not isinstance(sec_raw, Mapping):
                raise ConfigValidationError(f"Section '{sec_name}' must be a mapping.")
            allowed_fields = set(sec_cls.__dataclass_fields__.keys())
            unknown_fields = set(sec_raw.keys()) - allowed_fields
            if unknown_fields:
                raise ConfigValidationError(
                    f"Unknown keys in section '{sec_name}': {sorted(unknown_fields)}"
                )
            kwargs[sec_name] = sec_cls(**dict(sec_raw))

        cfg = cls(**kwargs)
        validate_config(cfg)
        return cfg


def _require_bool(val: Any, name: str) -> None:
    if not isinstance(val, bool):
        raise ConfigValidationError(f"Field '{name}' must be a bool, got {type(val).__name__}.")


def _require_int(val: Any, name: str, minimum: int = 0) -> None:
    if isinstance(val, bool) or not isinstance(val, int) or val < minimum:
        raise ConfigValidationError(f"Field '{name}' must be an int >= {minimum}, got {val!r}.")


def validate_config(cfg: OptimizeConfig, backend: str | None = None) -> None:
    """Validate field types, allowed values, and cross-field constraints."""
    if cfg.schema_version != SCHEMA_VERSION:
        raise ConfigValidationError(f"Unsupported schema_version={cfg.schema_version}; expected {SCHEMA_VERSION}.")
    if cfg.profile not in VALID_PROFILES:
        raise ConfigValidationError(f"Invalid profile '{cfg.profile}'; allowed: {sorted(VALID_PROFILES)}.")

    # runtime
    _require_bool(cfg.runtime.tf32, "runtime.tf32")
    _require_bool(cfg.runtime.cudnn_benchmark, "runtime.cudnn_benchmark")
    _require_bool(cfg.runtime.jemalloc, "runtime.jemalloc")
    _require_int(cfg.runtime.omp_threads, "runtime.omp_threads", minimum=0)
    if cfg.runtime.alloc_conf is not None and not isinstance(cfg.runtime.alloc_conf, str):
        raise ConfigValidationError("Field 'runtime.alloc_conf' must be str or None.")

    # amp
    _require_bool(cfg.amp.enabled, "amp.enabled")
    _require_bool(cfg.amp.grad_scaler, "amp.grad_scaler")
    if cfg.amp.dtype not in VALID_AMP_DTYPES:
        raise ConfigValidationError(f"Invalid amp.dtype '{cfg.amp.dtype}'; allowed: {sorted(VALID_AMP_DTYPES)}.")
    if cfg.amp.dtype == "bf16" and cfg.amp.grad_scaler:
        raise ConfigValidationError("amp.grad_scaler cannot be True when amp.dtype == 'bf16'.")
    if not cfg.amp.enabled and cfg.amp.grad_scaler:
        raise ConfigValidationError("amp.grad_scaler cannot be True when amp.enabled is False.")

    # compile
    _require_bool(cfg.compile.enabled, "compile.enabled")
    if cfg.compile.mode not in VALID_COMPILE_MODES:
        raise ConfigValidationError(
            f"Invalid compile.mode '{cfg.compile.mode}'; allowed: {sorted(VALID_COMPILE_MODES)}."
        )
    if cfg.compile.dynamic is not None:
        _require_bool(cfg.compile.dynamic, "compile.dynamic")
    if not isinstance(cfg.compile.cache_dir, str) or not cfg.compile.cache_dir.strip():
        raise ConfigValidationError("Field 'compile.cache_dir' must be a non-empty string.")
    _require_bool(cfg.compile.train_cuda_graphs, "compile.train_cuda_graphs")

    # optimizer & attention
    _require_bool(cfg.optimizer.fused, "optimizer.fused")
    _require_bool(cfg.optimizer.foreach, "optimizer.foreach")
    _require_bool(cfg.attention.flash_sdp, "attention.flash_sdp")
    _require_bool(cfg.attention.mem_efficient_sdp, "attention.mem_efficient_sdp")
    _require_bool(cfg.attention.math_sdp, "attention.math_sdp")

    # dataloader
    _require_bool(cfg.dataloader.auto_tune, "dataloader.auto_tune")
    _require_bool(cfg.dataloader.pin_memory, "dataloader.pin_memory")
    _require_bool(cfg.dataloader.persistent_workers, "dataloader.persistent_workers")
    if cfg.dataloader.num_workers is not None:
        _require_int(cfg.dataloader.num_workers, "dataloader.num_workers", minimum=0)
        if cfg.dataloader.num_workers == 0 and cfg.dataloader.persistent_workers:
            raise ConfigValidationError("dataloader.persistent_workers cannot be True when num_workers == 0.")
        if cfg.dataloader.num_workers == 0 and cfg.dataloader.prefetch_factor is not None:
            raise ConfigValidationError("dataloader.prefetch_factor must be None when num_workers == 0.")
    if cfg.dataloader.prefetch_factor is not None:
        _require_int(cfg.dataloader.prefetch_factor, "dataloader.prefetch_factor", minimum=1)

    # inference & kernels
    _require_bool(cfg.inference.inference_mode, "inference.inference_mode")
    _require_bool(cfg.inference.cuda_graphs, "inference.cuda_graphs")
    _require_bool(cfg.kernels.enabled, "kernels.enabled")
    _require_bool(cfg.kernels.fallback_on_error, "kernels.fallback_on_error")
    if not isinstance(cfg.kernels.allow_list, list) or not all(
        isinstance(item, str) for item in cfg.kernels.allow_list
    ):
        raise ConfigValidationError("Field 'kernels.allow_list' must be a list of strings.")

    if backend == "cpu":
        if cfg.runtime.tf32 or cfg.runtime.cudnn_benchmark or cfg.optimizer.fused or cfg.inference.cuda_graphs:
            raise ConfigValidationError("CUDA-specific flags cannot be enabled on CPU backend.")
