"""Pass 5.5a: Zero-code VRAM-aware Auto Activation Checkpointing (`use_reentrant=False`)."""

from __future__ import annotations

import functools
import gc
import os
import sys
from typing import TYPE_CHECKING, Any, Callable

from ptopt.config import OptimizeConfig
from ptopt.hooks.module_dispatch import (
    recompute_dispatch_guard,
    register_module_call_handler,
    unregister_module_call_handler,
)

if TYPE_CHECKING:
    from ptopt.hooks.manager import HookManager

_ACT_CKPT_INSTALLED = False
_DEFAULT_MIN_BLOCK_PARAMS = 100_000
_DEFAULT_VRAM_THRESHOLD = 0.82


def _is_leaf_primitive_module(mod: Any, torch_mod: Any) -> bool:
    nn_ns = getattr(torch_mod, "nn", None)
    if nn_ns is None:
        return False
    leaf_names = (
        "Linear",
        "Conv1d",
        "Conv2d",
        "Conv3d",
        "ConvTranspose2d",
        "Embedding",
        "LayerNorm",
        "BatchNorm1d",
        "BatchNorm2d",
        "GroupNorm",
        "Dropout",
        "ReLU",
        "GELU",
        "SiLU",
        "Identity",
    )
    for name in leaf_names:
        cls = getattr(nn_ns, name, None)
        if cls is not None and isinstance(mod, cls):
            return True
    return False


def _is_dynamo_compiling(torch_mod: Any) -> bool:
    compiler = getattr(torch_mod, "compiler", None)
    fn = getattr(compiler, "is_compiling", None)
    if callable(fn):
        try:
            return bool(fn())
        except Exception:  # noqa: BLE001
            return False
    return False


def _module_trainable_params(mod: Any) -> int:
    if not hasattr(mod, "parameters"):
        return 0
    try:
        return sum(int(p.numel()) for p in mod.parameters() if getattr(p, "requires_grad", False))
    except Exception:  # noqa: BLE001
        return 0


def discover_checkpoint_blocks(
    root_module: Any,
    torch_mod: Any,
    *,
    min_block_params: int | None = None,
) -> list[Any]:
    """Find non-overlapping repeated block submodules suitable for `torch.utils.checkpoint`."""
    nn_ns = getattr(torch_mod, "nn", None)
    if nn_ns is None or not hasattr(root_module, "modules"):
        return []
    min_params = (
        int(os.environ.get("PTOPT_ACT_CKPT_MIN_PARAMS", str(_DEFAULT_MIN_BLOCK_PARAMS)))
        if min_block_params is None
        else int(min_block_params)
    )
    mod_list_cls = getattr(nn_ns, "ModuleList", None)
    seq_cls = getattr(nn_ns, "Sequential", None)

    candidates: list[Any] = []
    seen_ids: set[int] = set()

    # 1. Repeated blocks inside ModuleList / Sequential containers (e.g. TransformerBlock, ResidualBlock)
    for container in root_module.modules():
        if container is root_module:
            continue
        is_container = (mod_list_cls is not None and isinstance(container, mod_list_cls)) or (
            seq_cls is not None and isinstance(container, seq_cls)
        )
        if not is_container:
            continue
        children = list(container.children()) if hasattr(container, "children") else []
        if len(children) < 2:
            continue
        effective_min = (
            min(min_params, 10_000)
            if (len(children) >= 4 and min_block_params is None and "PTOPT_ACT_CKPT_MIN_PARAMS" not in os.environ)
            else min_params
        )
        for child in children:
            cid = id(child)
            if cid in seen_ids or child is root_module:
                continue
            if _is_leaf_primitive_module(child, torch_mod):
                continue
            # Require at least one child op inside composite block so checkpoint saves real intermediates
            has_subchildren = hasattr(child, "children") and any(True for _ in child.children())
            if not has_subchildren:
                continue
            if _module_trainable_params(child) >= effective_min:
                seen_ids.add(cid)
                candidates.append(child)

    if candidates:
        return candidates

    # 2. Fallback: direct composite block children with repeated types or >= min_params
    for child in root_module.modules():
        cid = id(child)
        if child is root_module or cid in seen_ids or _is_leaf_primitive_module(child, torch_mod):
            continue
        if (mod_list_cls is not None and isinstance(child, mod_list_cls)) or (
            seq_cls is not None and isinstance(child, seq_cls)
        ):
            continue
        has_subchildren = hasattr(child, "children") and any(True for _ in child.children())
        if has_subchildren and _module_trainable_params(child) >= min_params:
            seen_ids.add(cid)
            candidates.append(child)

    # Filter out nested sub-blocks if their ancestor block is already selected
    ancestors_children: set[int] = set()
    for blk in candidates:
        for desc in blk.modules():
            if desc is not blk:
                ancestors_children.add(id(desc))
    return [blk for blk in candidates if id(blk) not in ancestors_children]


_TOTAL_VRAM_CACHE: dict[int, float] = {}


def _default_vram_ratio(torch_mod: Any) -> float | None:
    try:
        cuda_api = getattr(torch_mod, "cuda", None)
        if not cuda_api or not cuda_api.is_available():
            return None
        idx = int(cuda_api.current_device()) if hasattr(cuda_api, "current_device") else 0
        total = _TOTAL_VRAM_CACHE.get(idx)
        if total is None:
            total = float(cuda_api.get_device_properties(idx).total_memory)
            if total > 0:
                _TOTAL_VRAM_CACHE[idx] = total
        if total <= 0:
            return None
        peak = float(max(cuda_api.memory_allocated(idx), cuda_api.max_memory_allocated(idx)))
        return peak / total
    except Exception:  # noqa: BLE001
        return None


def _is_oom_error(exc: BaseException, torch_mod: Any) -> bool:
    cuda_ns = getattr(torch_mod, "cuda", None)
    for oom_cls in (
        getattr(cuda_ns, "OutOfMemoryError", None) if cuda_ns is not None else None,
        getattr(torch_mod, "OutOfMemoryError", None),
    ):
        if isinstance(oom_cls, type) and isinstance(exc, oom_cls):
            return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _build_sac_context_fn(ckpt_mod: Any, torch_mod: Any) -> Callable[[], Any] | None:
    create_ctx = getattr(ckpt_mod, "create_selective_checkpoint_contexts", None)
    aten_ops = getattr(getattr(torch_mod, "ops", None), "aten", None)
    if not callable(create_ctx) or aten_ops is None:
        return None
    ops_to_save: list[Any] = []
    for op_name in (
        "_scaled_dot_product_flash_attention",
        "_scaled_dot_product_efficient_attention",
        "_scaled_dot_product_cudnn_attention",
    ):
        op_packet = getattr(aten_ops, op_name, None)
        op_default = getattr(op_packet, "default", None) if op_packet is not None else None
        if op_default is not None:
            ops_to_save.append(op_default)
    if not ops_to_save:
        return None
    return functools.partial(create_ctx, ops_to_save)


def patch_module_activation_checkpointing(
    root_module: Any,
    torch_mod: Any,
    manager: "HookManager | None" = None,
    *,
    min_block_params: int | None = None,
    ratio: float = 1.0,
    use_sac: bool = False,
) -> int:
    """Wrap discovered composite blocks on `root_module` with `checkpoint(..., use_reentrant=False)`."""
    ckpt_mod = getattr(getattr(torch_mod, "utils", None), "checkpoint", None)
    ckpt_fn = getattr(ckpt_mod, "checkpoint", None)
    if not callable(ckpt_fn):
        return 0

    blocks = discover_checkpoint_blocks(root_module, torch_mod, min_block_params=min_block_params)
    if 0.0 < ratio < 1.0 and len(blocks) >= 4:
        step = max(2, round(1.0 / max(ratio, 0.1)))
        blocks = blocks[::step]
    sac_context_fn = _build_sac_context_fn(ckpt_mod, torch_mod) if use_sac else None

    patched = 0
    for blk in blocks:
        if getattr(blk, "_ptopt_act_ckpt_patched", False):
            continue
        orig_fwd = blk.forward

        def _make_ckpt_fwd(sub_blk: Any, base_fwd: Any, owner: Any) -> Any:
            @functools.wraps(base_fwd)
            def _guarded_base_fwd(*a: Any, **kw: Any) -> Any:
                # Prevent backward recomputation from re-entering top-level pass hooks
                # (e.g. amp.auto falsely seeing Step-0 autocast as user AMP).
                with recompute_dispatch_guard(torch_mod):
                    return base_fwd(*a, **kw)

            @functools.wraps(base_fwd)
            def _ckpt_forward(*args: Any, **kwargs: Any) -> Any:
                if not (
                    getattr(owner, "_ptopt_act_ckpt_active", False)
                    and getattr(sub_blk, "training", True)
                    and torch_mod.is_grad_enabled()
                    and not getattr(sub_blk, "_ptopt_act_ckpt_failed", False)
                ):
                    return base_fwd(*args, **kwargs)
                live_ckpt = getattr(ckpt_mod, "checkpoint", ckpt_fn)
                ckpt_kw: dict[str, Any] = {"use_reentrant": False}
                if sac_context_fn is not None:
                    ckpt_kw["context_fn"] = sac_context_fn
                ckpt_kw.update(kwargs)
                if _is_dynamo_compiling(torch_mod):
                    return live_ckpt(_guarded_base_fwd, *args, **ckpt_kw)
                try:
                    return live_ckpt(_guarded_base_fwd, *args, **ckpt_kw)
                except Exception as exc:  # noqa: BLE001
                    if _is_oom_error(exc, torch_mod) or type(exc).__name__ == "_StopRecomputationError":
                        raise
                    sub_blk._ptopt_act_ckpt_failed = True
                    sub_blk.forward = base_fwd
                    if manager is not None:
                        manager.record_event(
                            "runtime.activation_checkpointing",
                            "FALLBACK",
                            f"Checkpoint forward failed on {type(sub_blk).__name__}; restored original forward",
                            str(exc),
                        )
                    return base_fwd(*args, **kwargs)

            return _ckpt_forward

        blk._ptopt_act_ckpt_orig_fwd = orig_fwd
        blk._ptopt_act_ckpt_patched = True
        blk.forward = _make_ckpt_fwd(blk, orig_fwd, root_module)
        patched += 1
    return patched


class ActivationCheckpointRuntime:
    """Tracks VRAM-aware auto activation checkpointing state."""

    def __init__(self, cfg: OptimizeConfig, manager: "HookManager", torch_mod: Any, mode: str) -> None:
        self.cfg = cfg
        self.manager = manager
        self.torch = torch_mod
        self.mode = mode
        try:
            self.vram_threshold = float(os.environ.get("PTOPT_ACT_CKPT_VRAM_THRESHOLD", str(_DEFAULT_VRAM_THRESHOLD)))
        except ValueError:
            self.vram_threshold = _DEFAULT_VRAM_THRESHOLD
        self.triggered_modules = 0
        self.patched_blocks = 0


def install_activation_checkpoint_hook(
    cfg: OptimizeConfig,
    manager: "HookManager",
    torch_mod: Any,
    *,
    vram_probe_fn: Callable[[], float | None] | None = None,
) -> ActivationCheckpointRuntime | None:
    """Install lazy top-level `nn.Module.__call__` VRAM-aware activation checkpointing hook."""
    global _ACT_CKPT_INSTALLED  # noqa: PLW0603
    if _ACT_CKPT_INSTALLED or torch_mod is None or not hasattr(torch_mod, "nn"):
        return None

    mode = str(getattr(cfg.runtime, "activation_checkpointing", "off")).strip().lower()
    if mode not in ("auto", "always"):
        return None

    rt = ActivationCheckpointRuntime(cfg, manager, torch_mod, mode)
    probe_vram = vram_probe_fn or (lambda: _default_vram_ratio(torch_mod))
    default_ratio = "0.5" if mode == "auto" else "1.0"
    try:
        auto_ratio = max(0.0, min(1.0, float(os.environ.get("PTOPT_ACT_CKPT_RATIO", default_ratio))))
    except ValueError:
        auto_ratio = 0.5 if mode == "auto" else 1.0
    use_sac = os.environ.get("PTOPT_ACT_CKPT_SAC", "0").strip() == "1"
    enable_triton_fallback = (
        str(getattr(cfg, "profile", "balanced")) not in ("supersafe", "safe")
        and os.environ.get("PTOPT_ACT_CKPT_TRITON", "1").strip() != "0"
    )
    allow_oom_retry = (
        mode == "auto"
        and not bool(getattr(manager, "strict", False))
        and os.environ.get("PTOPT_ACT_CKPT_OOM_RETRY", "1").strip() != "0"
    )

    def _disable_compile_for_ckpt(mod_self: Any) -> bool:
        if os.environ.get("PTOPT_ACT_CKPT_ALLOW_COMPILE", "0").strip() == "1":
            return False
        # Bypass torch.compile on root AND all child submodules (so backward recomputation
        # inside checkpoint(..., use_reentrant=False) never compiles child blocks mid-backward,
        # preventing PyTorch #166926 tensor count mismatch).
        was_compiled = bool(getattr(mod_self, "_ptopt_compile_active", False))
        for sub_m in mod_self.modules():
            sub_m._compiled_call_impl = None
            sub_m._ptopt_train_compiled_impl = None
            sub_m._ptopt_eval_compiled_impl = None
            sub_m._ptopt_compile_active = False
            sub_m._ptopt_train_compile_active = False
            sub_m._ptopt_eval_compile_active = False
            sub_m._ptopt_compile_checked = True
            sub_m._ptopt_train_compile_checked = True
            sub_m._ptopt_eval_compile_checked = True
            sub_m._ptopt_compile_ineligible = True
        return was_compiled

    def _maybe_enable_triton_fallback(mod_self: Any) -> None:
        if not enable_triton_fallback or getattr(mod_self, "_ptopt_act_ckpt_triton_done", False):
            return
        mod_self._ptopt_act_ckpt_triton_done = True
        try:
            from ptopt.kernels.registry import enable_custom_kernels_on_module

            enable_custom_kernels_on_module(mod_self, torch_mod, manager)
        except Exception as exc:  # noqa: BLE001
            if getattr(manager, "strict", False):
                raise
            manager.record_event(
                "runtime.activation_checkpointing",
                "FALLBACK",
                f"Auto-Triton fallback after compile bypass failed on {type(mod_self).__name__}",
                str(exc),
            )

    def _activate_act_ckpt(mod_self: Any, *, ratio: float, reason_detail: str) -> int:
        was_active = bool(getattr(mod_self, "_ptopt_act_ckpt_active", False))
        n_patched = patch_module_activation_checkpointing(
            mod_self,
            torch_mod,
            manager=manager,
            ratio=ratio,
            use_sac=use_sac,
        )
        if n_patched > 0:
            mod_self._ptopt_act_ckpt_active = True
            mod_self._ptopt_act_ckpt_partial = ratio < 1.0
            if ratio < 1.0:
                # Clear pre-checkpoint cumulative peak stats so partial checkpointing is evaluated cleanly
                cuda_ns = getattr(torch_mod, "cuda", None)
                if cuda_ns is not None and getattr(cuda_ns, "is_available", lambda: False)():
                    reset_fn = getattr(cuda_ns, "reset_peak_memory_stats", None)
                    if callable(reset_fn):
                        try:
                            reset_fn()
                        except Exception:  # noqa: BLE001
                            pass
            if not was_active:
                rt.triggered_modules += 1
            rt.patched_blocks += n_patched
            if _disable_compile_for_ckpt(mod_self):
                manager.record_event(
                    "compile.auto",
                    "FALLBACK",
                    f"Bypassed torch.compile on {type(mod_self).__name__} while activation checkpointing active",
                )
            _maybe_enable_triton_fallback(mod_self)
            manager.record_event(
                "runtime.activation_checkpointing",
                "APPLIED",
                f"Enabled activation checkpointing on {n_patched} blocks of {type(mod_self).__name__} ({reason_detail})",
            )
        elif not was_active:
            mod_self._ptopt_act_ckpt_no_blocks = True
        else:
            mod_self._ptopt_act_ckpt_partial = False
        return n_patched

    def _act_ckpt_mod_call(mod_self: Any, args: tuple[Any, ...], kwargs: dict[str, Any], next_fn: Any) -> Any:
        eligible = (
            getattr(mod_self, "training", True)
            and torch_mod.is_grad_enabled()
            and not getattr(mod_self, "_ptopt_act_ckpt_no_blocks", False)
        )
        if eligible:
            if not getattr(mod_self, "_ptopt_act_ckpt_active", False):
                if mode == "always":
                    _activate_act_ckpt(
                        mod_self,
                        ratio=auto_ratio,
                        reason_detail=f"mode=always, ratio={auto_ratio:.2f}",
                    )
                else:
                    step_cnt = int(getattr(mod_self, "_ptopt_act_ckpt_probe_steps", 0)) + 1
                    mod_self._ptopt_act_ckpt_probe_steps = step_cnt
                    should_probe = (vram_probe_fn is not None) or step_cnt <= 2 or (step_cnt % 8 == 0)
                    vram_ratio = probe_vram() if should_probe else None
                    if vram_ratio is not None and vram_ratio >= rt.vram_threshold:
                        _activate_act_ckpt(
                            mod_self,
                            ratio=auto_ratio,
                            reason_detail=(
                                f"mode=auto (VRAM {vram_ratio * 100.0:.1f}% >= {rt.vram_threshold * 100.0:.0f}%, "
                                f"ratio={auto_ratio:.2f})"
                            ),
                        )
            elif getattr(mod_self, "_ptopt_act_ckpt_partial", False):
                step_cnt = int(getattr(mod_self, "_ptopt_act_ckpt_probe_steps", 0)) + 1
                mod_self._ptopt_act_ckpt_probe_steps = step_cnt
                should_probe = (vram_probe_fn is not None) or step_cnt <= 2 or (step_cnt % 8 == 0)
                vram_ratio = probe_vram() if should_probe else None
                if vram_ratio is not None and vram_ratio >= rt.vram_threshold:
                    _activate_act_ckpt(
                        mod_self,
                        ratio=1.0,
                        reason_detail=(
                            f"mode=auto escalate (VRAM {vram_ratio * 100.0:.1f}% >= "
                            f"{rt.vram_threshold * 100.0:.0f}%, ratio=1.00)"
                        ),
                    )
        try:
            return next_fn(mod_self, *args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            can_retry_oom = (
                allow_oom_retry
                and eligible
                and _is_oom_error(exc, torch_mod)
                and not getattr(mod_self, "_ptopt_act_ckpt_oom_retried", False)
                and (
                    not getattr(mod_self, "_ptopt_act_ckpt_active", False)
                    or getattr(mod_self, "_ptopt_act_ckpt_partial", False)
                )
            )
            if can_retry_oom:
                mod_self._ptopt_act_ckpt_oom_retried = True
                gc.collect()
                cuda_ns = getattr(torch_mod, "cuda", None)
                if cuda_ns is not None and hasattr(cuda_ns, "empty_cache"):
                    try:
                        cuda_ns.empty_cache()
                    except Exception:  # noqa: BLE001
                        pass
                n_patched = _activate_act_ckpt(
                    mod_self,
                    ratio=1.0,
                    reason_detail="oom_retry (mode=auto, ratio=1.00)",
                )
                if n_patched > 0:
                    manager.record_event(
                        "runtime.activation_checkpointing",
                        "FALLBACK",
                        f"Caught OutOfMemoryError on {type(mod_self).__name__}; cleared cache, enabled full "
                        "activation checkpointing (ratio=1.00), and retrying forward "
                        "(warning: partial pre-OOM state mutations such as BatchNorm/RNG may remain)",
                        str(exc),
                    )
                    return next_fn(mod_self, *args, **kwargs)
            raise

    register_module_call_handler(torch_mod, "runtime.activation_checkpointing", 20, _act_ckpt_mod_call)
    _ACT_CKPT_INSTALLED = True
    torch_mod._ptopt_act_ckpt_runtime = rt
    return rt


def uninstall_activation_checkpoint_hook(torch_mod: Any) -> None:
    """Restore monkey-patched activation checkpoint hook (for unit tests)."""
    global _ACT_CKPT_INSTALLED  # noqa: PLW0603
    if not _ACT_CKPT_INSTALLED or torch_mod is None:
        return
    unregister_module_call_handler(torch_mod, "runtime.activation_checkpointing")
    if hasattr(torch_mod, "_ptopt_act_ckpt_runtime"):
        delattr(torch_mod, "_ptopt_act_ckpt_runtime")
    _ACT_CKPT_INSTALLED = False


def apply_activation_checkpoint_pass(
    cfg: OptimizeConfig,
    manager: "HookManager",
    torch_mod: Any = None,
) -> str:
    """Install VRAM-aware Auto Activation Checkpointing when configured (`auto` / `always`)."""
    mode = str(getattr(cfg.runtime, "activation_checkpointing", "off")).strip().lower()
    if mode not in ("auto", "always"):
        return "SKIPPED: runtime.activation_checkpointing is off"
    tmod = torch_mod if torch_mod is not None else sys.modules.get("torch")
    if tmod is None or not hasattr(tmod, "nn"):
        return "SKIPPED: torch not loaded"
    rt = install_activation_checkpoint_hook(cfg, manager, tmod)
    if rt is None:
        return "SKIPPED: activation checkpoint hook already active"
    return f"Activation checkpoint hook installed (mode={rt.mode}, vram_threshold={rt.vram_threshold:.2f})"
