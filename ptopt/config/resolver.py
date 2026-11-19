"""Layered ConfigResolver with trace recording and CLI/ENV override support."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from ptopt.config.rules_builtin import LAYER_ORDER, ConfigRule, get_builtin_rules
from ptopt.config.schema import ConfigValidationError, OptimizeConfig, VALID_PROFILES, validate_config
from ptopt.detect.fingerprint import HardwareFingerprint


def _parse_bool_env(raw: str, name: str) -> bool:
    norm = raw.strip().lower()
    if norm in ("1", "true", "yes", "on"):
        return True
    if norm in ("0", "false", "no", "off"):
        return False
    raise ConfigValidationError(f"Invalid boolean value {raw!r} for environment variable {name}.")


_ENV_PARSERS: dict[str, tuple[str, Any]] = {
    "PTOPT_TF32": ("runtime.tf32", _parse_bool_env),
    "PTOPT_CUDNN_BENCHMARK": ("runtime.cudnn_benchmark", _parse_bool_env),
    "PTOPT_OMP_THREADS": ("runtime.omp_threads", lambda v, _n: int(v)),
    "PTOPT_JEMALLOC": ("runtime.jemalloc", _parse_bool_env),
    "PTOPT_ACTIVATION_CKPT": (
        "runtime.activation_checkpointing",
        lambda v, _n: (
            "always"
            if v.strip().lower() in ("1", "true", "always", "on")
            else ("off" if v.strip().lower() in ("0", "false", "off", "no") else v.strip().lower())
        ),
    ),
    "PTOPT_CUDA_PREFETCH": ("dataloader.cuda_prefetch", _parse_bool_env),
    "PTOPT_AMP": ("amp.enabled", _parse_bool_env),
    "PTOPT_AMP_DTYPE": ("amp.dtype", lambda v, _n: v.strip().lower()),
    "PTOPT_GRAD_SCALER": ("amp.grad_scaler", _parse_bool_env),
    "PTOPT_COMPILE": ("compile.enabled", _parse_bool_env),
    "PTOPT_COMPILE_MODE": ("compile.mode", lambda v, _n: v.strip().lower()),
    "PTOPT_TRAIN_CUDA_GRAPHS": ("compile.train_cuda_graphs", _parse_bool_env),
    "PTOPT_FUSED_OPTIMIZER": ("optimizer.fused", _parse_bool_env),
    "PTOPT_FLASH_SDP": ("attention.flash_sdp", _parse_bool_env),
    "PTOPT_DATALOADER_AUTO": ("dataloader.auto_tune", _parse_bool_env),
    "PTOPT_NUM_WORKERS": ("dataloader.num_workers", lambda v, _n: int(v)),
    "PTOPT_INFERENCE_MODE": ("inference.inference_mode", _parse_bool_env),
    "PTOPT_CUDA_GRAPHS": ("inference.cuda_graphs", _parse_bool_env),
    "PTOPT_KERNELS": ("kernels.enabled", _parse_bool_env),
}


@dataclass(frozen=True)
class TraceEntry:
    """Records a single config field change during layered resolution."""

    layer: str
    rule_name: str
    field_path: str
    old_value: Any
    new_value: Any
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResolutionResult:
    """Resolved OptimizeConfig paired with its rule and override audit trail."""

    config: OptimizeConfig
    matched_rules: list[str] = field(default_factory=list)
    trace: list[TraceEntry] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "matched_rules": list(self.matched_rules),
            "trace": [entry.to_dict() for entry in self.trace],
        }


def extract_env_overrides(env: Mapping[str, str]) -> dict[str, Any]:
    """Parse recognized `PTOPT_*` environment variables into dotted field overrides."""
    overrides: dict[str, Any] = {}
    for env_key, (field_path, parser) in _ENV_PARSERS.items():
        if env_key in env and env[env_key] != "":
            try:
                overrides[field_path] = parser(env[env_key], env_key)
            except ConfigValidationError:
                raise
            except ValueError as exc:
                raise ConfigValidationError(f"Invalid value {env[env_key]!r} for {env_key}: {exc}") from exc
    return overrides


class ConfigResolver:
    """Resolves a Schema v1 OptimizeConfig from a HardwareFingerprint and overrides."""

    def __init__(self, rules: Sequence[ConfigRule] | None = None) -> None:
        self._rules = list(rules) if rules is not None else get_builtin_rules()

    def resolve(
        self,
        fingerprint: HardwareFingerprint,
        profile: str | None = None,
        overrides: Mapping[str, Any] | None = None,
        env: Mapping[str, str] | None = None,
    ) -> ResolutionResult:
        env_map = os.environ if env is None else env
        selected_profile = profile or env_map.get("PTOPT_PROFILE") or "balanced"
        if selected_profile not in VALID_PROFILES:
            raise ConfigValidationError(
                f"Invalid profile '{selected_profile}'; allowed: {sorted(VALID_PROFILES)}."
            )

        cfg = OptimizeConfig(profile=selected_profile)
        matched_rules: list[str] = []
        trace: list[TraceEntry] = []

        for layer_name in LAYER_ORDER:
            for rule in self._rules:
                if rule.layer != layer_name:
                    continue
                updates = rule.evaluate(fingerprint, selected_profile)
                if updates is None:
                    continue
                matched_rules.append(rule.name)
                self._apply_updates(cfg, layer_name, rule.name, rule.reason, updates, trace)

        env_overrides = extract_env_overrides(env_map)
        if env_overrides:
            matched_rules.append("env_overrides")
            self._apply_updates(
                cfg,
                "explicit_override",
                "env_overrides",
                "Explicit PTOPT_* environment variable overrides",
                env_overrides,
                trace,
            )

        if overrides:
            matched_rules.append("cli_overrides")
            self._apply_updates(
                cfg,
                "explicit_override",
                "cli_overrides",
                "Explicit CLI / programmatic overrides",
                overrides,
                trace,
            )

        self._normalize_cross_fields(cfg, matched_rules, trace)
        validate_config(cfg, backend=fingerprint.effective_backend)
        return ResolutionResult(config=cfg, matched_rules=matched_rules, trace=trace)

    @staticmethod
    def _apply_updates(
        cfg: OptimizeConfig,
        layer: str,
        rule_name: str,
        reason: str,
        updates: Mapping[str, Any],
        trace: list[TraceEntry],
    ) -> None:
        for field_path, new_val in updates.items():
            old_val = cfg.set_field(field_path, new_val)
            if old_val != new_val:
                trace.append(
                    TraceEntry(
                        layer=layer,
                        rule_name=rule_name,
                        field_path=field_path,
                        old_value=old_val,
                        new_value=new_val,
                        reason=reason,
                    )
                )

    @staticmethod
    def _normalize_cross_fields(
        cfg: OptimizeConfig,
        matched_rules: list[str],
        trace: list[TraceEntry],
    ) -> None:
        """Keep cross-dependent fields consistent after layered merging."""
        norms: list[tuple[str, Any, str]] = []
        if not cfg.amp.enabled and cfg.amp.grad_scaler:
            norms.append(("amp.grad_scaler", False, "Normalize: amp.enabled is False"))
        if cfg.amp.dtype == "bf16" and cfg.amp.grad_scaler:
            norms.append(("amp.grad_scaler", False, "Normalize: bf16 does not use GradScaler"))
        if cfg.amp.enabled and cfg.amp.dtype == "fp16" and "amp.grad_scaler" not in {
            t.field_path for t in trace if t.layer == "explicit_override"
        }:
            if not cfg.amp.grad_scaler:
                norms.append(("amp.grad_scaler", True, "Normalize: fp16 AMP enables GradScaler"))

        if cfg.dataloader.num_workers == 0:
            if cfg.dataloader.persistent_workers:
                norms.append(("dataloader.persistent_workers", False, "Normalize: num_workers == 0"))
            if cfg.dataloader.prefetch_factor is not None:
                norms.append(("dataloader.prefetch_factor", None, "Normalize: num_workers == 0"))

        if norms:
            matched_rules.append("schema_normalization")
            for field_path, new_val, reason in norms:
                old_val = cfg.set_field(field_path, new_val)
                if old_val != new_val:
                    trace.append(
                        TraceEntry(
                            layer="normalization",
                            rule_name="schema_normalization",
                            field_path=field_path,
                            old_value=old_val,
                            new_value=new_val,
                            reason=reason,
                        )
                    )
