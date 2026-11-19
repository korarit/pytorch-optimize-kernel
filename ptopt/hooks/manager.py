"""Runtime hook manager, idempotency guard, lazy torch import hook, and sitecustomize chain-loader."""

from __future__ import annotations

import atexit
import importlib.util
import json
import os
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from ptopt.config import OptimizeConfig, migrate_config_dict
from ptopt.hooks.fallback_reporter import format_fallback_report, write_events_json

PassCallable = Callable[[OptimizeConfig, "HookManager"], str | None]


@dataclass(frozen=True)
class FallbackEvent:
    """Audit event recorded when an optimization pass is applied, skipped, or falls back."""

    pass_name: str
    status: str  # APPLIED | SKIPPED | FALLBACK
    reason: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _TorchImportHook:
    """sys.meta_path finder that triggers registered hooks right after `torch` loads."""

    def __init__(self, manager: "HookManager") -> None:
        self._manager = manager
        self._in_find_spec = False

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if fullname != "torch" or self._in_find_spec:
            return None
        self._in_find_spec = True
        try:
            spec = importlib.util.find_spec("torch")
        finally:
            self._in_find_spec = False

        if spec is None or spec.loader is None:
            return None

        orig_loader = spec.loader
        orig_exec_module = getattr(orig_loader, "exec_module", None)
        if orig_exec_module is None:
            return None

        hook_ref = self
        manager = self._manager

        class _WrappingLoader:
            def create_module(self, mod_spec: Any) -> Any:
                create_fn = getattr(orig_loader, "create_module", None)
                return create_fn(mod_spec) if callable(create_fn) else None

            def exec_module(self, module: Any) -> None:
                orig_exec_module(module)
                if hook_ref in sys.meta_path:
                    sys.meta_path.remove(hook_ref)
                manager.apply_torch_hooks(module)

        spec.loader = _WrappingLoader()  # type: ignore[assignment]
        return spec


class HookManager:
    """Manages idempotent hook initialization, optimization passes, and fallback tracking."""

    def __init__(self) -> None:
        self.bootstrapped: bool = False
        self.torch_patched: bool = False
        self.config: OptimizeConfig | None = None
        self.strict: bool = False
        self.verbose: bool = False
        self.fallback_report: bool = False
        self.telemetry_only: bool = False
        self._passes: list[tuple[str, PassCallable]] = []
        self._events: list[FallbackEvent] = []
        self._import_hook: _TorchImportHook | None = None
        self._atexit_registered: bool = False

    def reset(self) -> None:
        """Reset manager state (primarily for unit tests)."""
        if self._import_hook is not None and self._import_hook in sys.meta_path:
            sys.meta_path.remove(self._import_hook)
        self.bootstrapped = False
        self.torch_patched = False
        self.config = None
        self.strict = False
        self.verbose = False
        self.fallback_report = False
        self.telemetry_only = False
        self._passes.clear()
        self._events.clear()
        self._import_hook = None

    def register_pass(self, name: str, fn: PassCallable) -> None:
        """Register an optimization pass if not already registered under the same name."""
        if any(existing_name == name for existing_name, _ in self._passes):
            return
        self._passes.append((name, fn))

    def record_event(self, pass_name: str, status: str, reason: str, error: str | None = None) -> None:
        self._events.append(FallbackEvent(pass_name=pass_name, status=status, reason=reason, error=error))

    @property
    def events(self) -> list[FallbackEvent]:
        return list(self._events)

    def bootstrap(self, env: Mapping[str, str] | None = None) -> bool:
        """Initialize hooks idempotently from `PTOPT_*` environment variables."""
        if self.bootstrapped:
            return False

        env_map = os.environ if env is None else env
        if env_map.get("PTOPT_ACTIVE") != "1":
            return False

        self.strict = env_map.get("PTOPT_STRICT") == "1"
        self.verbose = env_map.get("PTOPT_VERBOSE") == "1"
        self.fallback_report = env_map.get("PTOPT_FALLBACK_REPORT") == "1"
        self.telemetry_only = env_map.get("PTOPT_TELEMETRY_ONLY") == "1"

        raw_cfg = env_map.get("PTOPT_CONFIG_JSON")
        if raw_cfg:
            try:
                parsed_cfg = json.loads(raw_cfg)
                mig = migrate_config_dict(parsed_cfg, emit_warnings=self.verbose)
                self.config = mig.config
                if mig.migrated:
                    self.record_event(
                        "config.migrate",
                        "MIGRATED",
                        f"Auto-migrated config schema {mig.source_version!r} -> v{mig.target_version}: "
                        + "; ".join(mig.warnings[:3]),
                    )
            except Exception as exc:  # noqa: BLE001
                if self.strict:
                    raise
                self.record_event("config.load", "FALLBACK", "Failed to parse PTOPT_CONFIG_JSON", str(exc))
                self.config = OptimizeConfig()
        else:
            self.config = OptimizeConfig()

        self.bootstrapped = True
        os.environ["_PTOPT_BOOTSTRAPPED"] = "1"

        if (self.fallback_report or self.verbose or env_map.get("PTOPT_EVENTS_OUT")) and not self._atexit_registered:
            atexit.register(self._emit_fallback_summary)
            self._atexit_registered = True

        if "torch" in sys.modules:
            self.apply_torch_hooks(sys.modules["torch"])
        else:
            self._install_torch_import_hook()

        return True

    def _install_torch_import_hook(self) -> None:
        if self._import_hook is None:
            self._import_hook = _TorchImportHook(self)
        if self._import_hook not in sys.meta_path:
            sys.meta_path.insert(0, self._import_hook)

    def apply_torch_hooks(self, torch_mod: Any = None) -> None:
        """Apply all registered passes once after `torch` has been imported."""
        if self.torch_patched:
            return
        self.torch_patched = True
        cfg = self.config or OptimizeConfig()

        if torch_mod is None:
            torch_mod = sys.modules.get("torch")

        from ptopt.hooks.checkpoint_compat import install_checkpoint_compat_hooks  # noqa: PLC0415
        from ptopt.hooks.telemetry_hook import maybe_install_telemetry_hook  # noqa: PLC0415

        telemetry_only = self.telemetry_only or os.environ.get("PTOPT_TELEMETRY_ONLY") == "1"
        if telemetry_only:
            self.record_event(
                "ptopt.passes",
                "SKIPPED",
                "PTOPT_TELEMETRY_ONLY=1 (untouched PyTorch baseline probe)",
            )
        else:
            install_checkpoint_compat_hooks(torch_mod)

            for pass_name, pass_fn in self._passes:
                try:
                    msg = pass_fn(cfg, self)
                    if isinstance(msg, str) and msg.startswith("SKIPPED:"):
                        self.record_event(pass_name, "SKIPPED", msg.removeprefix("SKIPPED:").strip())
                    else:
                        self.record_event(pass_name, "APPLIED", msg or "Applied successfully")
                except Exception as exc:  # noqa: BLE001
                    if self.strict:
                        raise
                    self.record_event(pass_name, "FALLBACK", "Runtime error during pass init", str(exc))
                    if self.verbose:
                        warnings.warn(f"[ptopt] Pass '{pass_name}' fell back: {exc}", RuntimeWarning, stacklevel=2)

        # Install telemetry AFTER passes so nn.Module.__call__ telemetry sits outer to ModuleCallDispatcher
        telemetry_collector = maybe_install_telemetry_hook(torch_mod, self)
        if telemetry_collector is not None:
            self._telemetry_collector = telemetry_collector

    def format_events_summary(self) -> str:
        return format_fallback_report(
            self._events,
            profile=(self.config.profile if self.config is not None else None),
        )

    def _emit_fallback_summary(self) -> None:
        events_out = os.environ.get("PTOPT_EVENTS_OUT")
        if events_out:
            try:
                write_events_json(self._events, events_out)
            except Exception:  # noqa: BLE001
                pass
        if self.fallback_report or self.verbose:
            print(self.format_events_summary(), file=sys.stderr)


_GLOBAL_MANAGER = HookManager()


def get_hook_manager() -> HookManager:
    """Return the process-wide HookManager singleton."""
    return _GLOBAL_MANAGER


def bootstrap_from_env(env: Mapping[str, str] | None = None) -> bool:
    """Convenience entrypoint called by sitecustomize bootstrap."""
    env_map = os.environ if env is None else env
    if env_map.get("PTOPT_TELEMETRY_ONLY") != "1":
        from ptopt.passes import register_builtin_passes  # noqa: PLC0415

        register_builtin_passes(_GLOBAL_MANAGER)
    return _GLOBAL_MANAGER.bootstrap(env=env)


def chain_load_sitecustomize(current_file: str | Path) -> list[Path]:
    """Execute any downstream `sitecustomize.py` on `sys.path` after ours."""
    if os.environ.get("_PTOPT_CHAIN_ACTIVE") == "1":
        return []

    current_dir = Path(current_file).resolve().parent
    loaded: list[Path] = []
    visited_dirs: set[Path] = {current_dir}

    os.environ["_PTOPT_CHAIN_ACTIVE"] = "1"
    try:
        for raw_entry in list(sys.path):
            entry = Path(raw_entry or ".").resolve()
            if entry in visited_dirs or not entry.is_dir():
                continue
            visited_dirs.add(entry)
            candidate = entry / "sitecustomize.py"
            if not candidate.is_file():
                continue
            spec = importlib.util.spec_from_file_location("_ptopt_chained_sitecustomize", candidate)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            loaded.append(candidate)
    finally:
        os.environ.pop("_PTOPT_CHAIN_ACTIVE", None)
    return loaded
