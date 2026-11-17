"""Pass 3.5: Safe automatic `torch.compile`, persistent Inductor cache, dynamic-shape & error fallback."""

from __future__ import annotations

import functools
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.config import OptimizeConfig

if TYPE_CHECKING:
    from src.hooks.manager import HookManager

_AUTO_COMPILE_INSTALLED = False


def _has_cuda_context(mod_self: Any, args: tuple[Any, ...], kwargs: dict[str, Any], torch_mod: Any) -> bool:
    """Return True if module parameters or inputs live on CUDA."""
    tensor_cls = getattr(torch_mod, "Tensor", None)
    if tensor_cls is not None:
        for item in (*args, *kwargs.values()):
            if isinstance(item, tensor_cls) and getattr(item, "is_cuda", False):
                return True
    if hasattr(mod_self, "parameters"):
        try:
            first_param = next(mod_self.parameters(), None)
            if first_param is not None and getattr(first_param, "is_cuda", False):
                return True
        except Exception:  # noqa: BLE001
            pass
    return False


def _is_compilable_module(mod_self: Any, torch_mod: Any) -> bool:
    """Skip parameterless wrappers and built-in `_Loss` modules from redundant compilation."""
    loss_base = getattr(getattr(getattr(torch_mod, "nn", None), "modules", None), "loss", None)
    loss_cls = getattr(loss_base, "_Loss", None) if loss_base is not None else None
    if loss_cls is not None and isinstance(mod_self, loss_cls):
        return False
    if hasattr(mod_self, "parameters"):
        try:
            if next(mod_self.parameters(), None) is not None:
                return True
        except Exception:  # noqa: BLE001
            pass
    return False


def _shape_signature(args: tuple[Any, ...], kwargs: dict[str, Any], torch_mod: Any) -> tuple[Any, ...]:
    """Build lightweight shape+dtype signature of input tensors to detect dynamic-shape workloads."""
    tensor_cls = getattr(torch_mod, "Tensor", None)
    sig: list[Any] = []
    for item in (*args, *kwargs.values()):
        if tensor_cls is not None and isinstance(item, tensor_cls):
            sig.append((tuple(item.shape), str(item.dtype), str(getattr(item, "device", ""))))
    return tuple(sig)


def _configure_cache_and_dynamo(cfg: OptimizeConfig, torch_mod: Any, strict: bool) -> str | None:
    """Configure persistent Inductor cache directory and safe Dynamo error suppression."""
    cache_resolved: str | None = None
    if cfg.compile.cache_dir:
        try:
            cache_path = Path(cfg.compile.cache_dir).expanduser().resolve()
            cache_path.mkdir(parents=True, exist_ok=True)
            triton_path = cache_path / "triton"
            triton_path.mkdir(parents=True, exist_ok=True)
            cache_resolved = str(cache_path)
            os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", cache_resolved)
            os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
            os.environ.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "1")
            os.environ.setdefault("TRITON_CACHE_DIR", str(triton_path))
        except Exception:  # noqa: BLE001
            cache_resolved = None

    dynamo = getattr(torch_mod, "_dynamo", None)
    dynamo_cfg = getattr(dynamo, "config", None) if dynamo is not None else None
    if dynamo_cfg is not None:
        try:
            if not strict and os.environ.get("PTOPT_COMPILE_STRICT") != "1":
                dynamo_cfg.suppress_errors = True
        except Exception:  # noqa: BLE001
            pass
    return cache_resolved


class AutoCompileRuntime:
    """Tracks compiled modules, dynamic-shape transitions, and compile fallback events."""

    def __init__(self, cfg: OptimizeConfig, manager: "HookManager", torch_mod: Any) -> None:
        self.cfg = cfg
        self.manager = manager
        self.torch = torch_mod
        self.mode = cfg.compile.mode
        self.dynamic = cfg.compile.dynamic
        self.compiled_count = 0
        self.fallback_count = 0
        self.setup_time_ms = 0.0
        self.cache_dir = _configure_cache_and_dynamo(cfg, torch_mod, strict=manager.strict)


def _attempt_compile_module(
    mod_self: Any,
    rt: AutoCompileRuntime,
    torch_mod: Any,
    *,
    dynamic_override: bool | None,
    reason_tag: str,
) -> bool:
    """Compile `mod_self._call_impl` in place via `torch.compile` with safety fallback."""
    t0 = time.perf_counter()
    target_fn = getattr(mod_self, "_call_impl", None)
    if not callable(target_fn):
        return False

    compile_kwargs: dict[str, Any] = {"mode": rt.mode}
    if dynamic_override is not None:
        compile_kwargs["dynamic"] = dynamic_override

    try:
        mod_self._compiled_call_impl = torch_mod.compile(target_fn, **compile_kwargs)
        rt.compiled_count += 1
        rt.setup_time_ms += (time.perf_counter() - t0) * 1000.0
        mod_self._ptopt_compile_active = True
        mod_self._ptopt_compile_dynamic = dynamic_override
        rt.manager.record_event(
            "compile.auto",
            "APPLIED",
            f"{reason_tag} {type(mod_self).__name__} (mode={rt.mode}, dynamic={dynamic_override})",
        )
        return True
    except Exception as exc:  # noqa: BLE001
        mod_self._compiled_call_impl = None
        mod_self._ptopt_compile_active = False
        mod_self._ptopt_compile_failed = True
        rt.fallback_count += 1
        rt.setup_time_ms += (time.perf_counter() - t0) * 1000.0
        rt.manager.record_event(
            "compile.auto",
            "FALLBACK",
            f"torch.compile setup failed on {type(mod_self).__name__}; reverting to eager",
            str(exc),
        )
        if rt.manager.strict:
            raise
        return False


def install_auto_compile_hook(
    cfg: OptimizeConfig,
    manager: "HookManager",
    torch_mod: Any,
) -> AutoCompileRuntime | None:
    """Install idempotent lazy top-level `nn.Module.__call__` compilation hook."""
    global _AUTO_COMPILE_INSTALLED  # noqa: PLW0603
    if _AUTO_COMPILE_INSTALLED or torch_mod is None or not hasattr(torch_mod, "nn"):
        return None
    if not cfg.compile.enabled or not callable(getattr(torch_mod, "compile", None)):
        return None

    rt = AutoCompileRuntime(cfg, manager, torch_mod)
    nn_mod = torch_mod.nn.Module
    call_attr = "__call__"
    orig_mod_call = getattr(nn_mod, call_attr)
    depth = [0]
    allow_cpu = os.environ.get("PTOPT_COMPILE_CPU") == "1"
    auto_dynamic_on_change = os.environ.get("PTOPT_COMPILE_DYNAMIC_ON_CHANGE", "1") == "1"

    @functools.wraps(orig_mod_call)
    def _compile_mod_call(mod_self: Any, *args: Any, **kwargs: Any) -> Any:
        depth[0] += 1
        is_top = depth[0] == 1
        try:
            if not is_top:
                return orig_mod_call(mod_self, *args, **kwargs)

            if not getattr(mod_self, "_ptopt_compile_checked", False):
                mod_self._ptopt_compile_checked = True
                # Detect pre-compiled module (OptimizedModule or direct mod.compile())
                if hasattr(mod_self, "_orig_mod") or getattr(mod_self, "_compiled_call_impl", None) is not None:
                    mod_self._ptopt_compile_preexisting = True
                    manager.record_event(
                        "compile.auto",
                        "SKIPPED",
                        f"Pre-existing torch.compile detected on {type(mod_self).__name__}; yielding",
                    )
                elif not _is_compilable_module(mod_self, torch_mod):
                    mod_self._ptopt_compile_ineligible = True
                elif not allow_cpu and not _has_cuda_context(mod_self, args, kwargs, torch_mod):
                    mod_self._ptopt_compile_ineligible = True
                    manager.record_event(
                        "compile.auto",
                        "SKIPPED",
                        f"CPU-only module {type(mod_self).__name__} skipped (set PTOPT_COMPILE_CPU=1 to force)",
                    )
                else:
                    mod_self._ptopt_seen_shapes = {_shape_signature(args, kwargs, torch_mod)}
                    _attempt_compile_module(
                        mod_self,
                        rt,
                        torch_mod,
                        dynamic_override=rt.dynamic,
                        reason_tag="Compiled",
                    )
            elif getattr(mod_self, "_ptopt_compile_active", False) and not getattr(mod_self, "_ptopt_compile_failed", False):
                seen = getattr(mod_self, "_ptopt_seen_shapes", None)
                if seen is not None and len(seen) < 8:
                    sig = _shape_signature(args, kwargs, torch_mod)
                    if sig and sig not in seen:
                        seen.add(sig)
                        cur_dyn = getattr(mod_self, "_ptopt_compile_dynamic", rt.dynamic)
                        if len(seen) == 2 and cur_dyn is not True and auto_dynamic_on_change:
                            manager.record_event(
                                "compile.auto",
                                "FALLBACK",
                                f"Dynamic input shapes detected on {type(mod_self).__name__}; switching to dynamic=True",
                            )
                            _attempt_compile_module(
                                mod_self,
                                rt,
                                torch_mod,
                                dynamic_override=True,
                                reason_tag="Recompiled(dynamic)",
                            )

            try:
                return orig_mod_call(mod_self, *args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                if getattr(mod_self, "_ptopt_compile_active", False) and not getattr(mod_self, "_ptopt_compile_failed", False):
                    mod_self._compiled_call_impl = None
                    mod_self._ptopt_compile_active = False
                    mod_self._ptopt_compile_failed = True
                    rt.fallback_count += 1
                    manager.record_event(
                        "compile.auto",
                        "FALLBACK",
                        f"Compiled execution failed on {type(mod_self).__name__}; falling back to eager",
                        str(exc),
                    )
                    if manager.strict:
                        raise
                    return orig_mod_call(mod_self, *args, **kwargs)
                raise
        finally:
            depth[0] -= 1

    nn_mod._ptopt_compile_call_attr = call_attr
    nn_mod._ptopt_compile_orig_call = orig_mod_call
    setattr(nn_mod, call_attr, _compile_mod_call)
    _AUTO_COMPILE_INSTALLED = True
    torch_mod._ptopt_compile_runtime = rt
    return rt


def uninstall_auto_compile_hook(torch_mod: Any) -> None:
    """Restore monkey-patched `nn.Module.__call__` compile hook (for unit tests)."""
    global _AUTO_COMPILE_INSTALLED  # noqa: PLW0603
    if not _AUTO_COMPILE_INSTALLED or torch_mod is None:
        return
    nn_mod = getattr(torch_mod, "nn", None) and torch_mod.nn.Module
    if nn_mod is not None and hasattr(nn_mod, "_ptopt_compile_orig_call"):
        setattr(nn_mod, getattr(nn_mod, "_ptopt_compile_call_attr"), getattr(nn_mod, "_ptopt_compile_orig_call"))
        delattr(nn_mod, "_ptopt_compile_call_attr")
        delattr(nn_mod, "_ptopt_compile_orig_call")
    if hasattr(torch_mod, "_ptopt_compile_runtime"):
        delattr(torch_mod, "_ptopt_compile_runtime")
    _AUTO_COMPILE_INSTALLED = False


def apply_auto_compile_pass(
    cfg: OptimizeConfig,
    manager: "HookManager",
    torch_mod: Any = None,
) -> str:
    """Enable safe lazy `torch.compile` pass when `cfg.compile.enabled` is True."""
    if not cfg.compile.enabled:
        return "SKIPPED: compile.enabled is False"

    tmod = torch_mod if torch_mod is not None else sys.modules.get("torch")
    if tmod is None or not callable(getattr(tmod, "compile", None)):
        return "SKIPPED: torch.compile unavailable"

    rt = install_auto_compile_hook(cfg, manager, tmod)
    if rt is None:
        return "SKIPPED: Auto-Compile already active or ineligible"
    return f"Auto-Compile hook installed (mode={rt.mode}, dynamic={rt.dynamic}, cache={rt.cache_dir})"
