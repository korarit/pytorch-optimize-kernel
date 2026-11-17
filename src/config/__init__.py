"""Config Schema v1 and layered hardware-aware resolution package."""

from src.config.resolver import ConfigResolver, ResolutionResult, TraceEntry, extract_env_overrides
from src.config.rules_builtin import LAYER_ORDER, ConfigRule, get_builtin_rules
from src.config.schema import (
    AmpConfig,
    AttentionConfig,
    CompileConfig,
    ConfigValidationError,
    DataLoaderConfig,
    InferenceConfig,
    KernelsConfig,
    OptimizeConfig,
    OptimizerConfig,
    RuntimeConfig,
    SCHEMA_VERSION,
    VALID_AMP_DTYPES,
    VALID_COMPILE_MODES,
    VALID_PROFILES,
    validate_config,
)

__all__ = [
    "AmpConfig",
    "AttentionConfig",
    "CompileConfig",
    "ConfigResolver",
    "ConfigRule",
    "ConfigValidationError",
    "DataLoaderConfig",
    "InferenceConfig",
    "KernelsConfig",
    "LAYER_ORDER",
    "OptimizeConfig",
    "OptimizerConfig",
    "ResolutionResult",
    "RuntimeConfig",
    "SCHEMA_VERSION",
    "TraceEntry",
    "VALID_AMP_DTYPES",
    "VALID_COMPILE_MODES",
    "VALID_PROFILES",
    "extract_env_overrides",
    "get_builtin_rules",
    "validate_config",
]
