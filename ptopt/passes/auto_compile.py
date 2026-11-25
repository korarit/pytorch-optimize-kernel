"""Pass 3.5: Safe automatic `torch.compile`, persistent Inductor cache, dynamic-shape & error fallback."""

from __future__ import annotations

import functools
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ptopt.config import OptimizeConfig, detect_runtime_cache_key, validate_or_scope_cache_dir
from ptopt.passes.memory_allocator import probe_cuda_vram_headroom
from ptopt.hooks.module_dispatch import (
    build_shape_signature,
    estimate_module_depth,
    has_cuda_context,
    iter_nested_tensors,
    register_module_call_handler,
    resolve_dataparallel_target,
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
        depth = estimate_module_depth(mod_self)
        if depth >= 8:
            return True, f"launch-bound depth={depth}, params={total_params}"
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
            if (
                os.environ.get("PTOPT_DYNAMO_CAPTURE_DYNAMIC_OPS", "1") != "0"
                and hasattr(dynamo_cfg, "capture_dynamic_output_shape_ops")
            ):
                dynamo_cfg.capture_dynamic_output_shape_ops = True
            try:
                cache_limit = int(os.environ.get("PTOPT_DYNAMO_CACHE_LIMIT", "32"))
            except ValueError:
                cache_limit = 32
            if cache_limit > 0 and hasattr(dynamo_cfg, "cache_size_limit"):
                cur_limit = int(getattr(dynamo_cfg, "cache_size_limit", 8) or 8)
                if cur_limit < cache_limit:
                    dynamo_cfg.cache_size_limit = cache_limit
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
        self.bw_tune_checked = False
        self._restorable_inductor_flags: list[tuple[Any, str, Any]] = []
        self.cache_dir = _configure_cache_and_dynamo(cfg, torch_mod, strict=manager.strict)


def configure_inductor_bandwidth_tuning(
    cfg: OptimizeConfig,
    torch_mod: Any,
    manager: "HookManager",
    *,
    chosen_mode: str,
) -> dict[str, Any]:
    """Enable Inductor coordinate-descent & epilogue bandwidth tuning when VRAM headroom allows."""
    env_flag = os.environ.get("PTOPT_INDUCTOR_BW_TUNE")
    if env_flag == "0":
        info = {"enabled": False, "reason": "Disabled by PTOPT_INDUCTOR_BW_TUNE=0"}
        try:
            torch_mod._ptopt_inductor_bw_tune = info
        except Exception:  # noqa: BLE001
            pass
        manager.record_event("compile.inductor_bw_tune", "SKIPPED", info["reason"])
        return info

    is_max_auto = chosen_mode.startswith("max-autotune")
    boost_enabled = bool(getattr(cfg.runtime, "bandwidth_boost", False))
    if env_flag != "1" and not (boost_enabled and is_max_auto):
        reason_skip = (
            "runtime.bandwidth_boost is False (opt-in via --bandwidth-boost or PTOPT_INDUCTOR_BW_TUNE=1)"
            if not boost_enabled
            else f"mode={chosen_mode} (requires max-autotune or PTOPT_INDUCTOR_BW_TUNE=1)"
        )
        info = {"enabled": False, "reason": reason_skip}
        try:
            torch_mod._ptopt_inductor_bw_tune = info
        except Exception:  # noqa: BLE001
            pass
        manager.record_event("compile.inductor_bw_tune", "SKIPPED", info["reason"])
        return info

    headroom = probe_cuda_vram_headroom(torch_mod)
    if env_flag != "1":
        if headroom is None:
            info = {"enabled": False, "reason": "CUDA VRAM headroom unavailable"}
            try:
                torch_mod._ptopt_inductor_bw_tune = info
            except Exception:  # noqa: BLE001
                pass
            manager.record_event("compile.inductor_bw_tune", "SKIPPED", info["reason"])
            return info
        if headroom["total_mb"] < 16384.0 or headroom["used_ratio"] >= 0.75:
            info = {
                "enabled": False,
                "reason": (
                    f"VRAM headroom insufficient (total={headroom['total_mb']}MB, "
                    f"used={headroom['used_ratio']:.1%})"
                ),
                "vram_free_mb": headroom["free_mb"],
                "used_ratio": headroom["used_ratio"],
            }
            try:
                torch_mod._ptopt_inductor_bw_tune = info
            except Exception:  # noqa: BLE001
                pass
            manager.record_event("compile.inductor_bw_tune", "SKIPPED", info["reason"])
            return info

    applied_flags: list[str] = []
    try:
        import importlib

        ind_cfg = getattr(importlib.import_module("torch._inductor"), "config", None)
        if ind_cfg is not None:
            if hasattr(ind_cfg, "coordinate_descent_tuning"):
                ind_cfg.coordinate_descent_tuning = True
                applied_flags.append("coordinate_descent_tuning=True")
            if hasattr(ind_cfg, "epilogue_fusion"):
                ind_cfg.epilogue_fusion = True
                applied_flags.append("epilogue_fusion=True")
    except Exception as exc:  # noqa: BLE001
        info = {"enabled": False, "reason": f"Inductor config unavailable: {exc}"}
        try:
            torch_mod._ptopt_inductor_bw_tune = info
        except Exception:  # noqa: BLE001
            pass
        manager.record_event("compile.inductor_bw_tune", "SKIPPED", info["reason"])
        return info

    info = {
        "enabled": bool(applied_flags),
        "flags": applied_flags,
        "vram_free_mb": headroom["free_mb"] if headroom else None,
        "used_ratio": headroom["used_ratio"] if headroom else None,
    }
    try:
        torch_mod._ptopt_inductor_bw_tune = info
    except Exception:  # noqa: BLE001
        pass
    if applied_flags:
        hr_note = f", free_vram={headroom['free_mb']:.1f}MB ({headroom['used_ratio']:.1%} used)" if headroom else ""
        manager.record_event(
            "compile.inductor_bw_tune",
            "APPLIED",
            f"Enabled {', '.join(applied_flags)}{hr_note}",
        )
    else:
        manager.record_event("compile.inductor_bw_tune", "SKIPPED", "No tunable Inductor flags exposed")
    return info


def _read_dynamo_recompile_and_gb_counts() -> tuple[int, int]:
    """Return total TorchDynamo `(recompiles, graph_breaks + unimplemented)` counters."""
    du = sys.modules.get("torch._dynamo.utils")
    counters = getattr(du, "counters", None) if du is not None else None
    if not counters:
        return 0, 0
    try:
        recomp = sum(int(v) for v in (counters.get("recompiles") or {}).values())
        gb = sum(int(v) for v in (counters.get("graph_break") or {}).values()) + sum(
            int(v) for v in (counters.get("unimplemented") or {}).values()
        )
        return recomp, gb
    except Exception:  # noqa: BLE001
        return 0, 0


def _has_integer_index_input(args: tuple[Any, ...], kwargs: dict[str, Any], torch_mod: Any) -> bool:
    """Return True if inputs include integer/index tensors (e.g. token IDs whose sequence length often varies)."""
    tensor_cls = getattr(torch_mod, "Tensor", None)
    if tensor_cls is None:
        return False
    for item in (*args, *kwargs.values()):
        for tensor in iter_nested_tensors(item, tensor_cls):
            try:
                if not tensor.is_floating_point() and not tensor.is_complex():
                    return True
            except Exception:  # noqa: BLE001
                pass
    return False


def _discover_regional_blocks(
    root_mod: Any,
    torch_mod: Any,
    *,
    min_blocks: int = 4,
) -> list[Any]:
    """Return blocks from all non-overlapping homogeneous-shape `ModuleList`/`Sequential` containers."""
    nn_ns = getattr(torch_mod, "nn", None)
    mod_list_cls = getattr(nn_ns, "ModuleList", None)
    seq_cls = getattr(nn_ns, "Sequential", None)
    container_types = tuple(c for c in (mod_list_cls, seq_cls) if c is not None)
    if not container_types or not hasattr(root_mod, "modules"):
        return []

    from ptopt.passes.activation_checkpoint import _is_leaf_primitive_module  # noqa: PLC0415

    selected_blocks: list[Any] = []
    owned_submodules: set[int] = set()
    for container in root_mod.modules():
        if container is root_mod or not isinstance(container, container_types) or id(container) in owned_submodules:
            continue
        children = list(container.children()) if hasattr(container, "children") else []
        if len(children) < min_blocks:
            continue
        if len({type(c) for c in children}) != 1:
            continue
        c0 = children[0]
        if _is_leaf_primitive_module(c0, torch_mod):
            continue
        has_subchildren = hasattr(c0, "children") and any(True for _ in c0.children())
        if not has_subchildren:
            continue
        if any(
            id(c) in owned_submodules
            or getattr(c, "_ptopt_ckpt_wrapped", False)
            or getattr(c, "_ptopt_act_ckpt_active", False)
            for c in children
        ):
            continue
        # Verify identical parameter shapes across all blocks (excludes Pyramid/UNet lists with varying widths)
        try:
            shape_sigs = {tuple(tuple(p.shape) for p in c.parameters()) for c in children}
        except Exception:  # noqa: BLE001
            continue
        if len(shape_sigs) != 1 or not next(iter(shape_sigs)):
            continue
        selected_blocks.extend(children)
        for c in children:
            for sub in c.modules():
                owned_submodules.add(id(sub))
    return selected_blocks


def _should_use_auto_regional_compile(
    root_mod: Any,
    torch_mod: Any,
    rt: "AutoCompileRuntime",
    *,
    outer_mod: Any = None,
) -> list[Any]:
    """Return homogeneous `ModuleList` blocks when regional compilation avoids monolithic trace bottlenecks."""
    env = os.environ.get("PTOPT_COMPILE_REGIONAL", "").strip()
    if env == "0":
        return []
    if getattr(root_mod, "_ptopt_act_ckpt_active", False) or (
        outer_mod is not None and getattr(outer_mod, "_ptopt_act_ckpt_active", False)
    ):
        return []
    if env == "1":
        return _discover_regional_blocks(root_mod, torch_mod, min_blocks=4)

    blocks = _discover_regional_blocks(root_mod, torch_mod, min_blocks=6)
    if not blocks:
        return []
    total_params, _ = _count_module_params_and_elems(root_mod, (), {}, torch_mod)
    if total_params < 20_000_000:
        return []
    return blocks


def _maybe_enable_aggressive_shape_padding(rt: "AutoCompileRuntime", chosen_mode: str, torch_mod: Any) -> None:
    """Enable cheap TensorCore shape padding on CUDA when aggressive profile runs `default` compile mode."""
    if getattr(rt.cfg, "profile", "") != "aggressive" or chosen_mode != "default":
        return
    env_flag = os.environ.get("PTOPT_INDUCTOR_SHAPE_PADDING", "").strip()
    if env_flag == "0":
        return
    cuda_ns = getattr(torch_mod, "cuda", None)
    cuda_ok = bool(cuda_ns and getattr(cuda_ns, "is_available", lambda: False)())
    if not cuda_ok and env_flag != "1":
        return
    try:
        import importlib

        ind_cfg = getattr(importlib.import_module("torch._inductor"), "config", None)
        if ind_cfg is not None and hasattr(ind_cfg, "shape_padding"):
            prev_val = bool(ind_cfg.shape_padding)
            if not prev_val:
                if not any(obj is ind_cfg and attr == "shape_padding" for obj, attr, _ in rt._restorable_inductor_flags):
                    rt._restorable_inductor_flags.append((ind_cfg, "shape_padding", prev_val))
                ind_cfg.shape_padding = True
    except Exception:  # noqa: BLE001
        pass


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
    if not rt.bw_tune_checked:
        rt.bw_tune_checked = True
        configure_inductor_bandwidth_tuning(rt.cfg, torch_mod, rt.manager, chosen_mode=chosen_mode)
    _maybe_enable_aggressive_shape_padding(rt, chosen_mode, torch_mod)
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
        mod_self._ptopt_compiled_steps = 0
        mod_self._ptopt_mod_recompiles = 0
        mod_self._ptopt_mod_graph_breaks = 0
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


def _attempt_compile_regional_blocks(
    mod_self: Any,
    blocks: list[Any],
    rt: AutoCompileRuntime,
    torch_mod: Any,
    *,
    dynamic_override: bool | None,
    reason_tag: str,
    mode_override: str | None = None,
) -> bool:
    """Compile each repeated block in `blocks` while leaving outer root module eager."""
    t0 = time.perf_counter()
    chosen_mode = mode_override if mode_override is not None else _resolve_non_cg_compile_mode(rt.mode)
    if not rt.bw_tune_checked:
        rt.bw_tune_checked = True
        configure_inductor_bandwidth_tuning(rt.cfg, torch_mod, rt.manager, chosen_mode=chosen_mode)
    _maybe_enable_aggressive_shape_padding(rt, chosen_mode, torch_mod)
    compile_kwargs: dict[str, Any] = {"mode": chosen_mode}
    if dynamic_override is not None:
        compile_kwargs["dynamic"] = dynamic_override

    is_train = bool(getattr(mod_self, "training", True))
    impl_attr = "_ptopt_train_compiled_impl" if is_train else "_ptopt_eval_compiled_impl"
    active_attr = "_ptopt_train_compile_active" if is_train else "_ptopt_eval_compile_active"
    failed_attr = "_ptopt_train_compile_failed" if is_train else "_ptopt_eval_compile_failed"
    reg_active_attr = "_ptopt_train_regional_active" if is_train else "_ptopt_eval_regional_active"

    compiled_blks: list[Any] = []
    try:
        mod_self._compiled_call_impl = None
        for blk in blocks:
            if (
                getattr(blk, "_ptopt_ckpt_wrapped", False)
                or getattr(blk, "_ptopt_act_ckpt_active", False)
                or hasattr(blk, "_orig_mod")
            ):
                continue
            blk_fn = getattr(blk, "_call_impl", None)
            if not callable(blk_fn):
                continue
            blk_impl = torch_mod.compile(blk_fn, **compile_kwargs)
            blk._compiled_call_impl = blk_impl
            setattr(blk, impl_attr, blk_impl)
            setattr(blk, active_attr, True)
            setattr(blk, failed_attr, False)
            blk._ptopt_compile_active = True
            compiled_blks.append(blk)

        if not compiled_blks:
            return False

        setattr(mod_self, impl_attr, None)
        setattr(mod_self, active_attr, True)
        setattr(mod_self, failed_attr, False)
        setattr(mod_self, reg_active_attr, True)
        mod_self._ptopt_regional_blocks = compiled_blks
        mod_self._ptopt_compile_active = True
        mod_self._ptopt_compile_dynamic = dynamic_override
        mod_self._ptopt_compile_mode = chosen_mode
        rt.compiled_count += 1
        rt.setup_time_ms += (time.perf_counter() - t0) * 1000.0
        n_groups = len({type(b) for b in compiled_blks})
        blk_desc = f"{len(compiled_blks)}x {type(compiled_blks[0]).__name__}" if n_groups == 1 else f"{len(compiled_blks)} blocks ({n_groups} stacks)"
        rt.manager.record_event(
            "compile.auto",
            "APPLIED",
            f"{reason_tag} {type(mod_self).__name__} via {blk_desc} "
            f"(mode={chosen_mode}, dynamic={dynamic_override})",
        )
        return True
    except Exception as exc:  # noqa: BLE001
        for blk in compiled_blks:
            blk._compiled_call_impl = None
            setattr(blk, impl_attr, None)
            setattr(blk, active_attr, False)
            setattr(blk, failed_attr, True)
            blk._ptopt_compile_active = False
        setattr(mod_self, reg_active_attr, False)
        if not getattr(mod_self, "_ptopt_eval_regional_active" if is_train else "_ptopt_train_regional_active", False):
            mod_self._ptopt_regional_blocks = []
        setattr(mod_self, impl_attr, None)
        setattr(mod_self, active_attr, False)
        setattr(mod_self, failed_attr, True)
        mod_self._ptopt_compile_active = False
        mod_self._ptopt_compile_failed = True
        rt.fallback_count += 1
        rt.setup_time_ms += (time.perf_counter() - t0) * 1000.0
        rt.manager.record_event(
            "compile.auto",
            "FALLBACK",
            f"Regional block compile failed on {type(mod_self).__name__}; reverting to eager",
            str(exc),
        )
        if rt.manager.strict:
            raise
        return False


def _has_conv1d(mod_self: Any, torch_mod: Any) -> bool:
    c1d_cls = getattr(getattr(torch_mod, "nn", None), "Conv1d", None)
    if c1d_cls is None or not hasattr(mod_self, "modules"):
        return False
    return any(isinstance(sub, c1d_cls) for sub in mod_self.modules())


def _disable_cudnn_benchmark_for_dynamic_conv(
    mod_self: Any,
    torch_mod: Any,
    manager: "HookManager",
    reason: str,
) -> None:
    """Disable `cudnn.benchmark` when dynamic shapes hit Conv1d/Conv2d/Conv3d modules."""
    from ptopt.passes.channels_last import has_conv2d, has_conv3d  # noqa: PLC0415

    cudnn_backend = getattr(getattr(torch_mod, "backends", None), "cudnn", None)
    if (
        cudnn_backend is not None
        and getattr(cudnn_backend, "benchmark", False)
        and (_has_conv1d(mod_self, torch_mod) or has_conv2d(mod_self, torch_mod) or has_conv3d(mod_self, torch_mod))
    ):
        cudnn_backend.benchmark = False
        manager.record_event("runtime.cudnn_benchmark", "FALLBACK", reason)


def _has_explicit_dynamic_override(cfg: OptimizeConfig) -> bool:
    """Return True if user CLI/env explicitly set `compile.dynamic`."""
    raw_explicit = os.environ.get("PTOPT_EXPLICIT_OVERRIDES", "")
    if raw_explicit:
        keys = {k.strip() for k in raw_explicit.split(",") if k.strip()}
        if "compile.dynamic" in keys:
            return True
    for entry in getattr(cfg, "rule_trace", None) or ():
        if isinstance(entry, dict) and entry.get("name") in ("cli_overrides", "env_overrides", "explicit_override"):
            changes = entry.get("changes")
            if isinstance(changes, dict) and "compile.dynamic" in changes:
                return True
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
    preempt_dynamic = os.environ.get("PTOPT_COMPILE_PREEMPT_DYNAMIC", "1") == "1"
    explicit_dynamic_override = _has_explicit_dynamic_override(cfg)

    def _clear_active_compile_for_mode(mod_self: Any, is_train: bool) -> None:
        mod_self._compiled_call_impl = None
        mod_self._ptopt_compile_active = False
        reg_active_attr = "_ptopt_train_regional_active" if is_train else "_ptopt_eval_regional_active"
        if is_train:
            mod_self._ptopt_train_compiled_impl = None
            mod_self._ptopt_train_compile_active = False
        else:
            mod_self._ptopt_eval_compiled_impl = None
            mod_self._ptopt_eval_compile_active = False
        for blk in getattr(mod_self, "_ptopt_regional_blocks", ()):
            blk._compiled_call_impl = None
            blk._ptopt_compile_active = False
            if is_train:
                blk._ptopt_train_compiled_impl = None
                blk._ptopt_train_compile_active = False
            else:
                blk._ptopt_eval_compiled_impl = None
                blk._ptopt_eval_compile_active = False
        setattr(mod_self, reg_active_attr, False)

    def _compile_mod_call(mod_self: Any, args: tuple[Any, ...], kwargs: dict[str, Any], next_fn: Any) -> Any:
        target_mod, is_dp_wrap, dp_reason = resolve_dataparallel_target(mod_self, torch_mod)
        if is_dp_wrap and target_mod is None:
            if not getattr(mod_self, "_ptopt_dp_skip_logged", False):
                mod_self._ptopt_dp_skip_logged = True
                manager.record_event("compile.auto", "SKIPPED", dp_reason or "DataParallel skipped")
            return next_fn(mod_self, *args, **kwargs)
        if is_dp_wrap and not getattr(mod_self, "_ptopt_dp_unwrap_logged", False):
            mod_self._ptopt_dp_unwrap_logged = True
            manager.record_event("compile.auto", "APPLIED", dp_reason or "Unwrapped single-GPU DataParallel")

        is_train = bool(getattr(target_mod, "training", getattr(mod_self, "training", True)))

        # Keep train and eval compiled handles isolated when a single model alternates .train() / .eval()
        if not is_train and getattr(target_mod, "_ptopt_infer_cg_active", False):
            eval_impl = getattr(target_mod, "_ptopt_eval_compiled_impl", None)
            if eval_impl is not None:
                target_mod._compiled_call_impl = eval_impl
            return next_fn(mod_self, *args, **kwargs)

        reg_active_attr = "_ptopt_train_regional_active" if is_train else "_ptopt_eval_regional_active"
        other_reg_attr = "_ptopt_eval_regional_active" if is_train else "_ptopt_train_regional_active"

        if not is_train:
            eval_calls = int(getattr(target_mod, "_ptopt_auto_eval_calls", 0)) + 1
            target_mod._ptopt_auto_eval_calls = eval_calls
            try:
                min_eval_calls = int(os.environ.get("PTOPT_COMPILE_EVAL_MIN_CALLS", "2"))
            except ValueError:
                min_eval_calls = 2
            if eval_calls < min_eval_calls and not getattr(target_mod, "_ptopt_eval_compile_checked", False):
                if not hasattr(target_mod, "_ptopt_auto_eval_warmup_shape"):
                    target_mod._ptopt_auto_eval_warmup_shape = _shape_signature(args, kwargs, torch_mod)
                if not getattr(target_mod, "_ptopt_compile_preexisting", False):
                    target_mod._compiled_call_impl = None
                if getattr(target_mod, "_ptopt_train_regional_active", False) or getattr(target_mod, "_ptopt_eval_regional_active", False):
                    for blk in getattr(target_mod, "_ptopt_regional_blocks", ()):
                        blk._compiled_call_impl = None
                return next_fn(mod_self, *args, **kwargs)

        checked_attr = "_ptopt_train_compile_checked" if is_train else "_ptopt_eval_compile_checked"
        active_attr = "_ptopt_train_compile_active" if is_train else "_ptopt_eval_compile_active"
        failed_attr = "_ptopt_train_compile_failed" if is_train else "_ptopt_eval_compile_failed"
        impl_attr = "_ptopt_train_compiled_impl" if is_train else "_ptopt_eval_compiled_impl"
        other_attr = "_ptopt_eval_compiled_impl" if is_train else "_ptopt_train_compiled_impl"
        other_impl = getattr(target_mod, other_attr, None)
        other_reg_active = bool(getattr(target_mod, other_reg_attr, False)) and bool(
            getattr(target_mod, "_ptopt_regional_blocks", None)
        )

        if not getattr(target_mod, checked_attr, False):
            setattr(target_mod, checked_attr, True)
            target_mod._ptopt_compile_checked = True

            cur_impl = getattr(target_mod, "_compiled_call_impl", None)
            is_user_precompiled = hasattr(target_mod, "_orig_mod") or (
                cur_impl is not None and cur_impl is not other_impl
            )
            can_reuse_other = (
                (other_impl is not None or other_reg_active)
                and not is_user_precompiled
                and not getattr(target_mod, "_ptopt_compile_demoted", False)
                and (
                    (not is_train and not getattr(target_mod, "_ptopt_train_cg_active", False))
                    or (is_train and not rt.train_cuda_graphs and not getattr(target_mod, "_ptopt_infer_cg_active", False))
                )
            )
            cur_sig = _shape_signature(args, kwargs, torch_mod)
            eval_warmup_sig = None if is_train else getattr(target_mod, "_ptopt_auto_eval_warmup_shape", None)
            eval_warmup_dynamic = bool(
                not is_train
                and eval_warmup_sig
                and cur_sig
                and cur_sig != eval_warmup_sig
                and auto_dynamic_on_change
                and rt.dynamic is not True
                and not explicit_dynamic_override
            )
            if eval_warmup_dynamic:
                # Shape changed between eval warmup calls -> skip reusing static other_impl and compile dynamic=True once
                can_reuse_other = False
            if is_user_precompiled:
                target_mod._ptopt_compile_preexisting = True
                manager.record_event(
                    "compile.auto",
                    "SKIPPED",
                    f"Pre-existing torch.compile detected on {type(target_mod).__name__}; yielding",
                )
            elif can_reuse_other:
                if other_reg_active:
                    setattr(target_mod, impl_attr, None)
                    setattr(target_mod, active_attr, True)
                    setattr(target_mod, reg_active_attr, True)
                    target_mod._compiled_call_impl = None
                    target_mod._ptopt_compile_active = True
                    for blk in getattr(target_mod, "_ptopt_regional_blocks", ()):
                        blk_other = getattr(blk, other_attr, None)
                        if blk_other is not None:
                            setattr(blk, impl_attr, blk_other)
                            setattr(blk, active_attr, True)
                            blk._compiled_call_impl = blk_other
                            blk._ptopt_compile_active = True
                else:
                    setattr(target_mod, impl_attr, other_impl)
                    setattr(target_mod, active_attr, True)
                    target_mod._compiled_call_impl = other_impl
                    target_mod._ptopt_compile_active = True
            elif not _is_compilable_module(target_mod, torch_mod):
                target_mod._ptopt_compile_ineligible = True
            elif getattr(target_mod, "_ptopt_act_ckpt_active", False) or getattr(mod_self, "_ptopt_act_ckpt_active", False):
                target_mod._ptopt_compile_ineligible = True
                manager.record_event(
                    "compile.auto",
                    "SKIPPED",
                    f"Activation checkpointing active on {type(target_mod).__name__}; bypassing compile for VRAM safety",
                )
            elif not allow_cpu and not _has_cuda_context(target_mod, args, kwargs, torch_mod):
                target_mod._ptopt_compile_ineligible = True
                manager.record_event(
                    "compile.auto",
                    "SKIPPED",
                    f"CPU-only module {type(target_mod).__name__} skipped (set PTOPT_COMPILE_CPU=1 to force)",
                )
            else:
                worth_static, static_reason = _is_worth_static_compile(target_mod, args, kwargs, torch_mod, allow_cpu)
                if not worth_static:
                    target_mod._ptopt_compile_ineligible = True
                    _clear_active_compile_for_mode(target_mod, is_train)
                    manager.record_event(
                        "compile.auto",
                        "SKIPPED",
                        f"Tiny module {type(target_mod).__name__} skipped compile for profitability ({static_reason})",
                    )
                else:
                    seen_init = {cur_sig} if cur_sig else set()
                    if eval_warmup_sig:
                        seen_init.add(eval_warmup_sig)
                    target_mod._ptopt_seen_shapes = seen_init
                    target_mod._ptopt_cg_stable_steps = 1
                    target_mod._ptopt_train_cg_active = False

                    dl_has_rem = (
                        is_train
                        and rt.dynamic is not True
                        and not explicit_dynamic_override
                        and auto_dynamic_on_change
                        and preempt_dynamic
                        and bool(getattr(torch_mod, "_ptopt_dataloader_has_remainder", False))
                    )
                    reg_blocks = _should_use_auto_regional_compile(target_mod, torch_mod, rt, outer_mod=mod_self)
                    if reg_blocks:
                        reg_needs_dyn = bool(
                            not explicit_dynamic_override
                            and (
                                rt.dynamic is True
                                or dl_has_rem
                                or eval_warmup_dynamic
                                or _has_integer_index_input(args, kwargs, torch_mod)
                            )
                        )
                        reg_dyn = True if reg_needs_dyn else rt.dynamic
                        reg_mode = "default" if reg_dyn is True else _resolve_non_cg_compile_mode(rt.mode)
                        _attempt_compile_regional_blocks(
                            target_mod,
                            reg_blocks,
                            rt,
                            torch_mod,
                            dynamic_override=reg_dyn,
                            mode_override=reg_mode,
                            reason_tag="Compiled(regional)",
                        )
                    else:
                        if dl_has_rem or eval_warmup_dynamic:
                            _disable_cudnn_benchmark_for_dynamic_conv(
                                target_mod,
                                torch_mod,
                                manager,
                                f"Disabled cudnn_benchmark due to dynamic shapes on {type(target_mod).__name__}",
                            )
                            worth_dyn, dyn_reason = _is_worth_dynamic_compile(target_mod, args, kwargs, torch_mod, rt)
                            if is_train and rt.train_cuda_graphs:
                                target_mod._ptopt_train_cg_checked = True
                                manager.record_event(
                                    "compile.train_cuda_graphs",
                                    "SKIPPED",
                                    f"DataLoader remainder batch detected on {type(target_mod).__name__}; using dynamic=True",
                                )
                            if not worth_dyn:
                                target_mod._ptopt_compile_ineligible = True
                                _clear_active_compile_for_mode(target_mod, is_train)
                                manager.record_event(
                                    "compile.auto",
                                    "SKIPPED",
                                    f"Small dynamic-shape module {type(target_mod).__name__} stayed eager for profitability ({dyn_reason})",
                                )
                            else:
                                rem_info = getattr(torch_mod, "_ptopt_dataloader_remainder_info", None)
                                rem_note = (
                                    f"dataset={rem_info[0]}, batch_size={rem_info[1]}"
                                    if isinstance(rem_info, tuple) and len(rem_info) == 2
                                    else "uneven remainder batch"
                                )
                                dyn_tag = "Compiled(dynamic,eval_warmup)" if eval_warmup_dynamic else f"Compiled(dynamic,dl_remainder:{rem_note})"
                                _attempt_compile_module(
                                    target_mod,
                                    rt,
                                    torch_mod,
                                    dynamic_override=True,
                                    mode_override="default",
                                    reason_tag=dyn_tag,
                                )
                        elif is_train and rt.train_cuda_graphs and rt.dynamic is not True:
                            ok_cg, reason_cg = _is_eligible_for_train_cuda_graphs(target_mod, args, kwargs, torch_mod, rt)
                            if ok_cg:
                                # Single-pass warmup: run Step 0 eager to verify 2-step static shape before compiling once
                                target_mod._ptopt_train_cg_pending = True
                                target_mod._ptopt_train_cg_reason = reason_cg
                                target_mod._compiled_call_impl = None
                                target_mod._ptopt_compile_active = False
                            else:
                                target_mod._ptopt_train_cg_checked = True
                                manager.record_event(
                                    "compile.train_cuda_graphs",
                                    "SKIPPED",
                                    f"Training CUDA Graphs skipped on {type(target_mod).__name__}: {reason_cg}",
                                )
                                _attempt_compile_module(
                                    target_mod,
                                    rt,
                                    torch_mod,
                                    dynamic_override=rt.dynamic,
                                    mode_override=_resolve_non_cg_compile_mode(rt.mode),
                                    reason_tag="Compiled",
                                )
                        else:
                            init_mode = _resolve_non_cg_compile_mode(rt.mode) if (not is_train or not rt.train_cuda_graphs) else rt.mode
                            _attempt_compile_module(
                                target_mod,
                                rt,
                                torch_mod,
                                dynamic_override=rt.dynamic,
                                mode_override=init_mode,
                                reason_tag="Compiled",
                            )
        elif is_train and getattr(target_mod, "_ptopt_train_cg_pending", False):
            target_mod._ptopt_train_cg_pending = False
            target_mod._ptopt_train_cg_checked = True
            seen = getattr(target_mod, "_ptopt_seen_shapes", None)
            sig = _shape_signature(args, kwargs, torch_mod)
            if seen is not None and sig and sig in seen:
                reason_cg = getattr(target_mod, "_ptopt_train_cg_reason", "static launch-bound module")
                ok_recomp = _attempt_compile_module(
                    target_mod,
                    rt,
                    torch_mod,
                    dynamic_override=False,
                    mode_override="reduce-overhead",
                    reason_tag="Compiled(train_cuda_graphs)",
                )
                if ok_recomp:
                    target_mod._ptopt_train_cg_active = True
                    rt.train_cg_count += 1
                    manager.record_event(
                        "compile.train_cuda_graphs",
                        "APPLIED",
                        f"Enabled training CUDA Graphs on {type(target_mod).__name__} ({reason_cg})",
                    )
            else:
                if seen is not None and sig:
                    seen.add(sig)
                target_mod._ptopt_train_cg_failed = True
                rt.fallback_count += 1
                manager.record_event(
                    "compile.train_cuda_graphs",
                    "FALLBACK",
                    f"Dynamic shape detected on {type(target_mod).__name__}; skipping CUDA Graphs",
                )
                worth_dyn, dyn_reason = _is_worth_dynamic_compile(target_mod, args, kwargs, torch_mod, rt)
                if not worth_dyn:
                    target_mod._ptopt_compile_ineligible = True
                    _clear_active_compile_for_mode(target_mod, is_train)
                    manager.record_event(
                        "compile.auto",
                        "FALLBACK",
                        f"Small dynamic-shape module {type(target_mod).__name__} stayed eager for profitability ({dyn_reason})",
                    )
                else:
                    _attempt_compile_module(
                        target_mod,
                        rt,
                        torch_mod,
                        dynamic_override=True,
                        mode_override="default",
                        reason_tag="Compiled(dynamic)",
                    )
        elif getattr(target_mod, active_attr, False) and not getattr(target_mod, failed_attr, False):
            if getattr(target_mod, reg_active_attr, False):
                target_mod._compiled_call_impl = None
                target_mod._ptopt_compile_active = True
                for blk in getattr(target_mod, "_ptopt_regional_blocks", ()):
                    blk._compiled_call_impl = getattr(blk, impl_attr, None)
                    blk._ptopt_compile_active = blk._compiled_call_impl is not None
                seen_reg = getattr(target_mod, "_ptopt_seen_shapes", None)
                if seen_reg is not None and len(seen_reg) < 8:
                    sig_reg = _shape_signature(args, kwargs, torch_mod)
                    if sig_reg and sig_reg not in seen_reg:
                        seen_reg.add(sig_reg)
                        cur_reg_dyn = getattr(target_mod, "_ptopt_compile_dynamic", rt.dynamic)
                        if len(seen_reg) == 2 and cur_reg_dyn is not True and auto_dynamic_on_change and not explicit_dynamic_override:
                            reg_blocks = list(getattr(target_mod, "_ptopt_regional_blocks", ()))
                            if reg_blocks:
                                _attempt_compile_regional_blocks(
                                    target_mod,
                                    reg_blocks,
                                    rt,
                                    torch_mod,
                                    dynamic_override=True,
                                    mode_override="default",
                                    reason_tag="Recompiled(regional,dynamic)",
                                )
            else:
                target_mod._compiled_call_impl = getattr(target_mod, impl_attr, None)
                target_mod._ptopt_compile_active = True
                seen = getattr(target_mod, "_ptopt_seen_shapes", None)
                cur_dyn = getattr(target_mod, "_ptopt_compile_dynamic", rt.dynamic)
                if (
                    seen is not None
                    and len(seen) < 8
                    and (cur_dyn is not True or getattr(target_mod, "_ptopt_train_cg_active", False))
                ):
                    sig = _shape_signature(args, kwargs, torch_mod)
                    if sig and sig not in seen:
                        seen.add(sig)
                        if is_train and getattr(target_mod, "_ptopt_train_cg_active", False):
                            target_mod._ptopt_train_cg_active = False
                            target_mod._ptopt_train_cg_failed = True
                            rt.fallback_count += 1
                            manager.record_event(
                                "compile.train_cuda_graphs",
                                "FALLBACK",
                                f"Dynamic shape detected on {type(target_mod).__name__}; demoted CUDA Graphs to mode=default",
                            )
                        cur_dyn = getattr(target_mod, "_ptopt_compile_dynamic", rt.dynamic)
                        if len(seen) == 2 and cur_dyn is not True and auto_dynamic_on_change:
                            _disable_cudnn_benchmark_for_dynamic_conv(
                                target_mod,
                                torch_mod,
                                manager,
                                f"Disabled cudnn_benchmark due to dynamic Conv shapes on {type(target_mod).__name__}",
                            )
                            worth_dyn, dyn_reason = _is_worth_dynamic_compile(target_mod, args, kwargs, torch_mod, rt)
                            if not worth_dyn:
                                _clear_active_compile_for_mode(target_mod, is_train)
                                target_mod._ptopt_compile_ineligible = True
                                rt.fallback_count += 1
                                manager.record_event(
                                    "compile.auto",
                                    "FALLBACK",
                                    f"Small dynamic-shape module {type(target_mod).__name__} reverted to eager for profitability ({dyn_reason})",
                                )
                            else:
                                manager.record_event(
                                    "compile.auto",
                                    "FALLBACK",
                                    f"Dynamic input shapes detected on {type(target_mod).__name__}; switching to dynamic=True",
                                )
                                _attempt_compile_module(
                                    target_mod,
                                    rt,
                                    torch_mod,
                                    dynamic_override=True,
                                    mode_override="default",
                                    reason_tag="Recompiled(dynamic)",
                                )
        elif not getattr(target_mod, "_ptopt_compile_preexisting", False):
            target_mod._compiled_call_impl = getattr(target_mod, impl_attr, None)

        if is_train and getattr(target_mod, "_ptopt_train_cg_active", False):
            _invoke_cudagraph_mark_step_begin(torch_mod)

        check_thrash = (
            getattr(target_mod, active_attr, False)
            and not getattr(target_mod, reg_active_attr, False)
            and not getattr(target_mod, "_ptopt_compile_demoted", False)
            and not getattr(target_mod, "_ptopt_train_cg_active", False)
            and int(getattr(target_mod, "_ptopt_compiled_steps", 0)) < 8
        )
        r_pre, gb_pre = _read_dynamo_recompile_and_gb_counts() if check_thrash else (0, 0)

        try:
            res = next_fn(mod_self, *args, **kwargs)
            if check_thrash:
                c_steps = int(getattr(target_mod, "_ptopt_compiled_steps", 0)) + 1
                target_mod._ptopt_compiled_steps = c_steps
                r_post, gb_post = _read_dynamo_recompile_and_gb_counts()
                d_r_call = max(0, r_post - r_pre)
                d_gb_call = max(0, gb_post - gb_pre)
                mod_r = int(getattr(target_mod, "_ptopt_mod_recompiles", 0)) + d_r_call
                mod_gb = int(getattr(target_mod, "_ptopt_mod_graph_breaks", 0)) + d_gb_call
                target_mod._ptopt_mod_recompiles = mod_r
                target_mod._ptopt_mod_graph_breaks = mod_gb
                try:
                    max_recomp = int(os.environ.get("PTOPT_COMPILE_MAX_RECOMPILES", "6"))
                except ValueError:
                    max_recomp = 6
                try:
                    max_gb = int(os.environ.get("PTOPT_COMPILE_MAX_GRAPH_BREAKS", "12"))
                except ValueError:
                    max_gb = 12
                recomp_thrash = max_recomp > 0 and mod_r >= max_recomp
                severe_gb_thrash = max_gb > 0 and mod_gb >= max_gb and (mod_r > 0 or c_steps == 1)
                if recomp_thrash or severe_gb_thrash:
                    target_mod._ptopt_compile_demoted = True
                    _clear_active_compile_for_mode(target_mod, is_train)
                    rt.fallback_count += 1
                    demote_blocks = (
                        _discover_regional_blocks(target_mod, torch_mod, min_blocks=4)
                        if os.environ.get("PTOPT_COMPILE_REGIONAL", "").strip() != "0"
                        else []
                    )
                    if demote_blocks:
                        _attempt_compile_regional_blocks(
                            target_mod,
                            demote_blocks,
                            rt,
                            torch_mod,
                            dynamic_override=True,
                            mode_override="default",
                            reason_tag=f"Demoted thrashing root compile (recompiles={mod_r}, graph_breaks={mod_gb}) -> regional",
                        )
                    else:
                        manager.record_event(
                            "compile.auto",
                            "FALLBACK",
                            f"Compiled execution thrashing on {type(target_mod).__name__} "
                            f"(recompiles={mod_r}, graph_breaks={mod_gb}); demoted to eager",
                        )
                        if getattr(cfg.kernels, "enabled", False) and (allow_cpu or _has_cuda_context(target_mod, args, kwargs, torch_mod)):
                            try:
                                from ptopt.kernels import enable_custom_kernels_on_module  # noqa: PLC0415

                                enable_custom_kernels_on_module(target_mod, torch_mod, manager)
                            except Exception:  # noqa: BLE001
                                pass
            return res
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
            act_ckpt_retried = getattr(target_mod, "_ptopt_act_ckpt_oom_retried", False) or getattr(mod_self, "_ptopt_act_ckpt_oom_retried", False)
            act_ckpt_active = getattr(target_mod, "_ptopt_act_ckpt_active", False) or getattr(mod_self, "_ptopt_act_ckpt_active", False)
            act_ckpt_partial = getattr(target_mod, "_ptopt_act_ckpt_partial", False) or getattr(mod_self, "_ptopt_act_ckpt_partial", False)
            if (
                is_oom
                and is_train
                and getattr(cfg.runtime, "activation_checkpointing", "off") == "auto"
                and not manager.strict
                and getattr(torch_mod, "_ptopt_act_ckpt_runtime", None) is not None
                and os.environ.get("PTOPT_ACT_CKPT_OOM_RETRY", "1").strip() != "0"
                and not act_ckpt_retried
                and (not act_ckpt_active or act_ckpt_partial)
            ):
                raise
            if is_train and getattr(target_mod, "_ptopt_train_cg_active", False) and not getattr(target_mod, "_ptopt_train_cg_failed", False):
                target_mod._ptopt_train_cg_active = False
                target_mod._ptopt_train_cg_failed = True
                rt.fallback_count += 1
                manager.record_event(
                    "compile.train_cuda_graphs",
                    "FALLBACK",
                    f"Training CUDA Graphs runtime error on {type(target_mod).__name__}; reverting to mode=default",
                    str(exc),
                )
                recomp_ok = _attempt_compile_module(
                    target_mod,
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
            if getattr(target_mod, active_attr, False) and not getattr(target_mod, failed_attr, False):
                _clear_active_compile_for_mode(target_mod, is_train)
                setattr(target_mod, failed_attr, True)
                target_mod._ptopt_compile_failed = True
                rt.fallback_count += 1
                manager.record_event(
                    "compile.auto",
                    "FALLBACK",
                    f"Compiled execution failed on {type(target_mod).__name__}; falling back to eager",
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
    rt = getattr(torch_mod, "_ptopt_compile_runtime", None)
    for cfg_obj, attr_name, old_val in getattr(rt, "_restorable_inductor_flags", ()):
        try:
            setattr(cfg_obj, attr_name, old_val)
        except Exception:  # noqa: BLE001
            pass
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
