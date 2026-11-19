"""Pass 3.5: Safe automatic `torch.compile`, persistent Inductor cache, dynamic-shape & error fallback."""

from __future__ import annotations

import functools
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ptopt.config import OptimizeConfig, detect_runtime_cache_key, validate_or_scope_cache_dir
from ptopt.hooks.module_dispatch import (
    build_shape_signature,
    has_cuda_context,
    iter_nested_tensors,
    register_module_call_handler,
    unregister_module_call_handler,
)

if TYPE_CHECKING:
    from ptopt.hooks.manager import HookManager

_AUTO_COMPILE_INSTALLED = False


def _has_cuda_context(mod_self: Any, args: tuple[Any, ...], kwargs: dict[str, Any], torch_mod: Any) -> bool:
    """Return True if module parameters or inputs live on CUDA."""
    return has_cuda_context(mod_self, args, kwargs, torch_mod)


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


_DEFAULT_STATIC_TINY_MIN_PARAMS = 50_000
_DEFAULT_STATIC_TINY_MIN_INPUT_ELEMS = 32_768
_DEFAULT_DYNAMIC_MIN_PARAMS = 250_000
_DEFAULT_DYNAMIC_MIN_INPUT_ELEMS = 65_536
_DEFAULT_TRAIN_CG_MAX_PARAMS = 15_000_000


def _detect_cuda_sm_for_compile(torch_mod: Any) -> int:
    override = os.environ.get("PTOPT_CUDA_SM_OVERRIDE")
    if override:
        try:
            return int(override)
        except ValueError:
            pass
    try:
        cuda_api = getattr(torch_mod, "cuda", None)
        if not cuda_api or not cuda_api.is_available():
            return 0
        idx = int(cuda_api.current_device()) if hasattr(cuda_api, "current_device") else 0
        props = cuda_api.get_device_properties(idx)
        return int(props.major) * 10 + int(props.minor)
    except Exception:  # noqa: BLE001
        return 0


def _invoke_cudagraph_mark_step_begin(torch_mod: Any) -> None:
    compiler = getattr(torch_mod, "compiler", None)
    mark_fn = getattr(compiler, "cudagraph_mark_step_begin", None)
    if callable(mark_fn):
        mark_fn()


def _has_active_dropout(mod_self: Any, torch_mod: Any) -> bool:
    if os.environ.get("PTOPT_TRAIN_CG_ALLOW_DROPOUT") == "1" or not getattr(mod_self, "training", True):
        return False
    nn_ns = getattr(torch_mod, "nn", None)
    drop_cls = getattr(nn_ns, "Dropout", None)
    if drop_cls is None or not hasattr(mod_self, "modules"):
        return False
    try:
        return any(isinstance(sub, drop_cls) and float(getattr(sub, "p", 0.0)) > 0.0 for sub in mod_self.modules())
    except Exception:  # noqa: BLE001
        return False


def _is_eligible_for_train_cuda_graphs(
    mod_self: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    torch_mod: Any,
    rt: "AutoCompileRuntime",
) -> tuple[bool, str]:
    """Return `(eligible, reason)` for safe static-shape training CUDA Graphs."""
    if not rt.train_cuda_graphs:
        return False, "compile.train_cuda_graphs is False"
    if not getattr(mod_self, "training", True):
        return False, "module is in eval() mode"
    if rt.dynamic is True:
        return False, "compile.dynamic=True requested"
    if getattr(getattr(rt.cfg, "amp", None), "grad_scaler", False):
        return False, "GradScaler enabled (CPU .item() step skip hazard)"

    sm_int = _detect_cuda_sm_for_compile(torch_mod)
    if sm_int < 80 and os.environ.get("PTOPT_TRAIN_CG_FORCE") != "1":
        return False, f"sm_{sm_int} < 80 (requires Ampere/Ada/Hopper sm>=80)"
    if not _has_cuda_context(mod_self, args, kwargs, torch_mod) and os.environ.get("PTOPT_TRAIN_CG_FORCE") != "1":
        return False, "non-CUDA module/input context"
    if _has_active_dropout(mod_self, torch_mod):
        return False, "active Dropout stochastic mask risk (set PTOPT_TRAIN_CG_ALLOW_DROPOUT=1 to override)"

    try:
        max_params = int(os.environ.get("PTOPT_TRAIN_CG_MAX_PARAMS", str(_DEFAULT_TRAIN_CG_MAX_PARAMS)))
    except ValueError:
        max_params = _DEFAULT_TRAIN_CG_MAX_PARAMS
    total_params, _ = _count_module_params_and_elems(mod_self, args, kwargs, torch_mod)
    if total_params > max_params:
        return False, f"params={total_params} > {max_params} (large compute-bound module)"
    return True, f"static launch-bound module (params={total_params}, sm_{sm_int})"


def _shape_signature(args: tuple[Any, ...], kwargs: dict[str, Any], torch_mod: Any) -> tuple[Any, ...]:
    """Build hashable shape+dtype+device signature of (possibly nested) input tensors."""
    return build_shape_signature(args, kwargs, torch_mod, include_device=True)


def _count_module_params_and_elems(
    mod_self: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    torch_mod: Any,
) -> tuple[int, int]:
    total_params = 0
    if hasattr(mod_self, "parameters"):
        try:
            total_params = sum(int(p.numel()) for p in mod_self.parameters())
        except Exception:  # noqa: BLE001
            total_params = 0

    tensor_cls = getattr(torch_mod, "Tensor", None)
    input_elems = 0
    if tensor_cls is not None:
        for item in (*args, *kwargs.values()):
            for tensor in iter_nested_tensors(item, tensor_cls):
                try:
                    input_elems += int(tensor.numel())
                except Exception:  # noqa: BLE001
                    pass
    return total_params, input_elems


def _is_worth_static_compile(
    mod_self: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    torch_mod: Any,
    allow_cpu: bool,
) -> tuple[bool, str]:
    """Skip compiling tiny CUDA modules whose JIT/guard overhead outweighs compute."""
    if os.environ.get("PTOPT_COMPILE_TINY") == "1":
        return True, "tiny compile explicitly enabled"

    default_params = "0" if allow_cpu else str(_DEFAULT_STATIC_TINY_MIN_PARAMS)
    try:
        min_params = int(os.environ.get("PTOPT_COMPILE_STATIC_MIN_PARAMS", default_params))
    except ValueError:
        min_params = 0 if allow_cpu else _DEFAULT_STATIC_TINY_MIN_PARAMS
    if min_params <= 0:
        return True, "static tiny gate disabled"

    try:
        min_elems = int(
            os.environ.get("PTOPT_COMPILE_STATIC_MIN_ELEMS", str(_DEFAULT_STATIC_TINY_MIN_INPUT_ELEMS))
        )
    except ValueError:
        min_elems = _DEFAULT_STATIC_TINY_MIN_INPUT_ELEMS

    total_params, input_elems = _count_module_params_and_elems(mod_self, args, kwargs, torch_mod)
    if total_params < min_params and input_elems < min_elems:
        return (
            False,
            f"params={total_params} < {min_params} and input_elems={input_elems} < {min_elems}",
        )
    return True, f"params={total_params}, input_elems={input_elems}"


def _is_worth_dynamic_compile(
    mod_self: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    torch_mod: Any,
    rt: "AutoCompileRuntime",
) -> tuple[bool, str]:
    """Return True if recompiling with `dynamic=True` is likely faster than eager fallback."""
    if rt.dynamic is True or os.environ.get("PTOPT_COMPILE_DYNAMIC_SMALL") == "1":
        return True, "dynamic compile explicitly requested"

    try:
        min_params = int(os.environ.get("PTOPT_COMPILE_DYNAMIC_MIN_PARAMS", str(_DEFAULT_DYNAMIC_MIN_PARAMS)))
    except ValueError:
        min_params = _DEFAULT_DYNAMIC_MIN_PARAMS
    try:
        min_elems = int(os.environ.get("PTOPT_COMPILE_DYNAMIC_MIN_ELEMS", str(_DEFAULT_DYNAMIC_MIN_INPUT_ELEMS)))
    except ValueError:
        min_elems = _DEFAULT_DYNAMIC_MIN_INPUT_ELEMS

    total_params, input_elems = _count_module_params_and_elems(mod_self, args, kwargs, torch_mod)

    if total_params < min_params and input_elems < min_elems:
        return (
            False,
            f"params={total_params} < {min_params} and input_elems={input_elems} < {min_elems}",
        )
    return True, f"params={total_params}, input_elems={input_elems}"


def _configure_cache_and_dynamo(cfg: OptimizeConfig, torch_mod: Any, strict: bool) -> str | None:
    """Configure versioned persistent Inductor cache directory and safe Dynamo error suppression."""
    cache_resolved: str | None = None
    if cfg.compile.cache_dir:
        try:
            key_info = detect_runtime_cache_key(torch_mod)
            validated = validate_or_scope_cache_dir(
                cfg.compile.cache_dir,
                key_info,
                write_meta=True,
                emit_warnings=True,
            )
            cache_resolved = str(validated.cache_dir)
            if validated.rotated:
                os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_resolved
                os.environ["TRITON_CACHE_DIR"] = str(validated.triton_dir)
            else:
                os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", cache_resolved)
                os.environ.setdefault("TRITON_CACHE_DIR", str(validated.triton_dir))
            os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
            os.environ.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "1")
            os.environ["PTOPT_CACHE_KEY"] = key_info.cache_key
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
        self.train_cuda_graphs = bool(getattr(cfg.compile, "train_cuda_graphs", False))
        self.compiled_count = 0
        self.train_cg_count = 0
        self.fallback_count = 0
        self.setup_time_ms = 0.0
        self.cache_dir = _configure_cache_and_dynamo(cfg, torch_mod, strict=manager.strict)


def _resolve_non_cg_compile_mode(requested_mode: str) -> str:
    """Ensure Inductor does not silently enable CUDA Graphs when train_cuda_graphs is disabled/ineligible."""
    if requested_mode == "max-autotune":
        return "max-autotune-no-cudagraphs"
    if requested_mode == "reduce-overhead":
        return "default"
    return requested_mode


def _attempt_compile_module(
    mod_self: Any,
    rt: AutoCompileRuntime,
    torch_mod: Any,
    *,
    dynamic_override: bool | None,
    reason_tag: str,
    mode_override: str | None = None,
) -> bool:
    """Compile `mod_self._call_impl` in place via `torch.compile` with safety fallback."""
    t0 = time.perf_counter()
    target_fn = getattr(mod_self, "_call_impl", None)
    if not callable(target_fn):
        return False

    chosen_mode = mode_override if mode_override is not None else rt.mode
    compile_kwargs: dict[str, Any] = {"mode": chosen_mode}
    if dynamic_override is not None:
        compile_kwargs["dynamic"] = dynamic_override

    is_train = bool(getattr(mod_self, "training", True))
    try:
        compiled_impl = torch_mod.compile(target_fn, **compile_kwargs)
        mod_self._compiled_call_impl = compiled_impl
        if is_train:
            mod_self._ptopt_train_compiled_impl = compiled_impl
            mod_self._ptopt_train_compile_active = True
            mod_self._ptopt_train_compile_failed = False
        else:
            mod_self._ptopt_eval_compiled_impl = compiled_impl
            mod_self._ptopt_eval_compile_active = True
            mod_self._ptopt_eval_compile_failed = False
        rt.compiled_count += 1
        rt.setup_time_ms += (time.perf_counter() - t0) * 1000.0
        mod_self._ptopt_compile_active = True
        mod_self._ptopt_compile_dynamic = dynamic_override
        mod_self._ptopt_compile_mode = chosen_mode
        rt.manager.record_event(
            "compile.auto",
            "APPLIED",
            f"{reason_tag} {type(mod_self).__name__} (mode={chosen_mode}, dynamic={dynamic_override})",
        )
        return True
    except Exception as exc:  # noqa: BLE001
        mod_self._compiled_call_impl = None
        if is_train:
            mod_self._ptopt_train_compiled_impl = None
            mod_self._ptopt_train_compile_active = False
            mod_self._ptopt_train_compile_failed = True
        else:
            mod_self._ptopt_eval_compiled_impl = None
            mod_self._ptopt_eval_compile_active = False
            mod_self._ptopt_eval_compile_failed = True
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
    allow_cpu = os.environ.get("PTOPT_COMPILE_CPU") == "1"
    auto_dynamic_on_change = os.environ.get("PTOPT_COMPILE_DYNAMIC_ON_CHANGE", "1") == "1"

    def _clear_active_compile_for_mode(mod_self: Any, is_train: bool) -> None:
        mod_self._compiled_call_impl = None
        mod_self._ptopt_compile_active = False
        if is_train:
            mod_self._ptopt_train_compiled_impl = None
            mod_self._ptopt_train_compile_active = False
        else:
            mod_self._ptopt_eval_compiled_impl = None
            mod_self._ptopt_eval_compile_active = False

    def _compile_mod_call(mod_self: Any, args: tuple[Any, ...], kwargs: dict[str, Any], next_fn: Any) -> Any:
        is_train = bool(getattr(mod_self, "training", True))

        # Keep train and eval compiled handles isolated when a single model alternates .train() / .eval()
        if not is_train and getattr(mod_self, "_ptopt_infer_cg_active", False):
            eval_impl = getattr(mod_self, "_ptopt_eval_compiled_impl", None)
            if eval_impl is not None:
                mod_self._compiled_call_impl = eval_impl
            return next_fn(mod_self, *args, **kwargs)

        checked_attr = "_ptopt_train_compile_checked" if is_train else "_ptopt_eval_compile_checked"
        active_attr = "_ptopt_train_compile_active" if is_train else "_ptopt_eval_compile_active"
        failed_attr = "_ptopt_train_compile_failed" if is_train else "_ptopt_eval_compile_failed"
        impl_attr = "_ptopt_train_compiled_impl" if is_train else "_ptopt_eval_compiled_impl"
        other_impl = getattr(mod_self, "_ptopt_eval_compiled_impl" if is_train else "_ptopt_train_compiled_impl", None)

        if not getattr(mod_self, checked_attr, False):
            setattr(mod_self, checked_attr, True)
            mod_self._ptopt_compile_checked = True

            cur_impl = getattr(mod_self, "_compiled_call_impl", None)
            is_user_precompiled = hasattr(mod_self, "_orig_mod") or (
                cur_impl is not None and cur_impl is not other_impl
            )
            can_reuse_other = (
                other_impl is not None
                and not is_user_precompiled
                and (
                    (not is_train and not getattr(mod_self, "_ptopt_train_cg_active", False))
                    or (is_train and not rt.train_cuda_graphs and not getattr(mod_self, "_ptopt_infer_cg_active", False))
                )
            )
            if is_user_precompiled:
                mod_self._ptopt_compile_preexisting = True
                manager.record_event(
                    "compile.auto",
                    "SKIPPED",
                    f"Pre-existing torch.compile detected on {type(mod_self).__name__}; yielding",
                )
            elif can_reuse_other:
                setattr(mod_self, impl_attr, other_impl)
                setattr(mod_self, active_attr, True)
                mod_self._compiled_call_impl = other_impl
                mod_self._ptopt_compile_active = True
            elif not _is_compilable_module(mod_self, torch_mod):
                mod_self._ptopt_compile_ineligible = True
            elif getattr(mod_self, "_ptopt_act_ckpt_active", False):
                mod_self._ptopt_compile_ineligible = True
                manager.record_event(
                    "compile.auto",
                    "SKIPPED",
                    f"Activation checkpointing active on {type(mod_self).__name__}; bypassing compile for VRAM safety",
                )
            elif not allow_cpu and not _has_cuda_context(mod_self, args, kwargs, torch_mod):
                mod_self._ptopt_compile_ineligible = True
                manager.record_event(
                    "compile.auto",
                    "SKIPPED",
                    f"CPU-only module {type(mod_self).__name__} skipped (set PTOPT_COMPILE_CPU=1 to force)",
                )
            else:
                worth_static, static_reason = _is_worth_static_compile(mod_self, args, kwargs, torch_mod, allow_cpu)
                if not worth_static:
                    mod_self._ptopt_compile_ineligible = True
                    _clear_active_compile_for_mode(mod_self, is_train)
                    manager.record_event(
                        "compile.auto",
                        "SKIPPED",
                        f"Tiny module {type(mod_self).__name__} skipped compile for profitability ({static_reason})",
                    )
                else:
                    mod_self._ptopt_seen_shapes = {_shape_signature(args, kwargs, torch_mod)}
                    mod_self._ptopt_cg_stable_steps = 1
                    mod_self._ptopt_train_cg_active = False

                    if is_train and rt.train_cuda_graphs and rt.dynamic is not True:
                        ok_cg, reason_cg = _is_eligible_for_train_cuda_graphs(mod_self, args, kwargs, torch_mod, rt)
                        if ok_cg:
                            # Single-pass warmup: run Step 0 eager to verify 2-step static shape before compiling once
                            mod_self._ptopt_train_cg_pending = True
                            mod_self._ptopt_train_cg_reason = reason_cg
                            mod_self._compiled_call_impl = None
                            mod_self._ptopt_compile_active = False
                        else:
                            mod_self._ptopt_train_cg_checked = True
                            manager.record_event(
                                "compile.train_cuda_graphs",
                                "SKIPPED",
                                f"Training CUDA Graphs skipped on {type(mod_self).__name__}: {reason_cg}",
                            )
                            _attempt_compile_module(
                                mod_self,
                                rt,
                                torch_mod,
                                dynamic_override=rt.dynamic,
                                mode_override=_resolve_non_cg_compile_mode(rt.mode),
                                reason_tag="Compiled",
                            )
                    else:
                        init_mode = _resolve_non_cg_compile_mode(rt.mode) if (not is_train or not rt.train_cuda_graphs) else rt.mode
                        _attempt_compile_module(
                            mod_self,
                            rt,
                            torch_mod,
                            dynamic_override=rt.dynamic,
                            mode_override=init_mode,
                            reason_tag="Compiled",
                        )
        elif is_train and getattr(mod_self, "_ptopt_train_cg_pending", False):
            mod_self._ptopt_train_cg_pending = False
            mod_self._ptopt_train_cg_checked = True
            seen = getattr(mod_self, "_ptopt_seen_shapes", None)
            sig = _shape_signature(args, kwargs, torch_mod)
            if seen is not None and sig and sig in seen:
                reason_cg = getattr(mod_self, "_ptopt_train_cg_reason", "static launch-bound module")
                ok_recomp = _attempt_compile_module(
                    mod_self,
                    rt,
                    torch_mod,
                    dynamic_override=False,
                    mode_override="reduce-overhead",
                    reason_tag="Compiled(train_cuda_graphs)",
                )
                if ok_recomp:
                    mod_self._ptopt_train_cg_active = True
                    rt.train_cg_count += 1
                    manager.record_event(
                        "compile.train_cuda_graphs",
                        "APPLIED",
                        f"Enabled training CUDA Graphs on {type(mod_self).__name__} ({reason_cg})",
                    )
            else:
                if seen is not None and sig:
                    seen.add(sig)
                mod_self._ptopt_train_cg_failed = True
                rt.fallback_count += 1
                manager.record_event(
                    "compile.train_cuda_graphs",
                    "FALLBACK",
                    f"Dynamic shape detected on {type(mod_self).__name__}; skipping CUDA Graphs",
                )
                worth_dyn, dyn_reason = _is_worth_dynamic_compile(mod_self, args, kwargs, torch_mod, rt)
                if not worth_dyn:
                    mod_self._ptopt_compile_ineligible = True
                    _clear_active_compile_for_mode(mod_self, is_train)
                    manager.record_event(
                        "compile.auto",
                        "FALLBACK",
                        f"Small dynamic-shape module {type(mod_self).__name__} stayed eager for profitability ({dyn_reason})",
                    )
                else:
                    _attempt_compile_module(
                        mod_self,
                        rt,
                        torch_mod,
                        dynamic_override=True,
                        mode_override="default",
                        reason_tag="Compiled(dynamic)",
                    )
        elif getattr(mod_self, active_attr, False) and not getattr(mod_self, failed_attr, False):
            mod_self._compiled_call_impl = getattr(mod_self, impl_attr, None)
            mod_self._ptopt_compile_active = True
            seen = getattr(mod_self, "_ptopt_seen_shapes", None)
            if seen is not None and len(seen) < 8:
                sig = _shape_signature(args, kwargs, torch_mod)
                if sig and sig not in seen:
                    seen.add(sig)
                    if is_train and getattr(mod_self, "_ptopt_train_cg_active", False):
                        mod_self._ptopt_train_cg_active = False
                        mod_self._ptopt_train_cg_failed = True
                        rt.fallback_count += 1
                        manager.record_event(
                            "compile.train_cuda_graphs",
                            "FALLBACK",
                            f"Dynamic shape detected on {type(mod_self).__name__}; demoted CUDA Graphs to mode=default",
                        )
                    cur_dyn = getattr(mod_self, "_ptopt_compile_dynamic", rt.dynamic)
                    if len(seen) == 2 and cur_dyn is not True and auto_dynamic_on_change:
                        from ptopt.passes.channels_last import has_conv2d, has_conv3d  # noqa: PLC0415

                        cudnn_backend = getattr(getattr(torch_mod, "backends", None), "cudnn", None)
                        if (
                            cudnn_backend is not None
                            and getattr(cudnn_backend, "benchmark", False)
                            and (has_conv2d(mod_self, torch_mod) or has_conv3d(mod_self, torch_mod))
                        ):
                            cudnn_backend.benchmark = False
                            manager.record_event(
                                "runtime.cudnn_benchmark",
                                "FALLBACK",
                                f"Disabled cudnn_benchmark due to dynamic Conv shapes on {type(mod_self).__name__}",
                            )
                        worth_dyn, dyn_reason = _is_worth_dynamic_compile(mod_self, args, kwargs, torch_mod, rt)
                        if not worth_dyn:
                            _clear_active_compile_for_mode(mod_self, is_train)
                            mod_self._ptopt_compile_ineligible = True
                            rt.fallback_count += 1
                            manager.record_event(
                                "compile.auto",
                                "FALLBACK",
                                f"Small dynamic-shape module {type(mod_self).__name__} reverted to eager for profitability ({dyn_reason})",
                            )
                        else:
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
                                mode_override="default",
                                reason_tag="Recompiled(dynamic)",
                            )
        elif not getattr(mod_self, "_ptopt_compile_preexisting", False):
            mod_self._compiled_call_impl = getattr(mod_self, impl_attr, None)

        if is_train and getattr(mod_self, "_ptopt_train_cg_active", False):
            _invoke_cudagraph_mark_step_begin(torch_mod)

        try:
            return next_fn(mod_self, *args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            # Let activation_checkpointing (outer handler order 20) rescue OOM before falling back compile to eager
            cuda_ns = getattr(torch_mod, "cuda", None)
            oom_types = tuple(
                t for t in (getattr(cuda_ns, "OutOfMemoryError", None) if cuda_ns else None, getattr(torch_mod, "OutOfMemoryError", None))
                if isinstance(t, type)
            )
            is_oom = (bool(oom_types) and isinstance(exc, oom_types)) or (
                isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()
            )
            if (
                is_oom
                and is_train
                and getattr(cfg.runtime, "activation_checkpointing", "off") == "auto"
                and not manager.strict
                and getattr(torch_mod, "_ptopt_act_ckpt_runtime", None) is not None
                and os.environ.get("PTOPT_ACT_CKPT_OOM_RETRY", "1").strip() != "0"
                and not getattr(mod_self, "_ptopt_act_ckpt_oom_retried", False)
                and (not getattr(mod_self, "_ptopt_act_ckpt_active", False) or getattr(mod_self, "_ptopt_act_ckpt_partial", False))
            ):
                raise
            if is_train and getattr(mod_self, "_ptopt_train_cg_active", False) and not getattr(mod_self, "_ptopt_train_cg_failed", False):
                mod_self._ptopt_train_cg_active = False
                mod_self._ptopt_train_cg_failed = True
                rt.fallback_count += 1
                manager.record_event(
                    "compile.train_cuda_graphs",
                    "FALLBACK",
                    f"Training CUDA Graphs runtime error on {type(mod_self).__name__}; reverting to mode=default",
                    str(exc),
                )
                recomp_ok = _attempt_compile_module(
                    mod_self,
                    rt,
                    torch_mod,
                    dynamic_override=rt.dynamic,
                    mode_override="default",
                    reason_tag="Fallback(default)",
                )
                if recomp_ok:
                    try:
                        return next_fn(mod_self, *args, **kwargs)
                    except Exception:  # noqa: BLE001
                        pass
            if getattr(mod_self, active_attr, False) and not getattr(mod_self, failed_attr, False):
                _clear_active_compile_for_mode(mod_self, is_train)
                setattr(mod_self, failed_attr, True)
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
                return next_fn(mod_self, *args, **kwargs)
            raise

    register_module_call_handler(torch_mod, "compile.auto", 60, _compile_mod_call)
    _AUTO_COMPILE_INSTALLED = True
    torch_mod._ptopt_compile_runtime = rt
    return rt


def uninstall_auto_compile_hook(torch_mod: Any) -> None:
    """Restore monkey-patched `nn.Module.__call__` compile hook (for unit tests)."""
    global _AUTO_COMPILE_INSTALLED  # noqa: PLW0603
    if not _AUTO_COMPILE_INSTALLED or torch_mod is None:
        return
    unregister_module_call_handler(torch_mod, "compile.auto")
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
