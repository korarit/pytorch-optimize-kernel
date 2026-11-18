"""Pass 5.5a: Zero-code VRAM-aware Auto Activation Checkpointing (`use_reentrant=False`)."""

from __future__ import annotations

import functools
import os
import sys
from typing import TYPE_CHECKING, Any, Callable

from src.config import OptimizeConfig

if TYPE_CHECKING:
    from src.hooks.manager import HookManager

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
            if _module_trainable_params(child) >= min_params:
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


def _default_vram_ratio(torch_mod: Any) -> float | None:
    try:
        cuda_api = getattr(torch_mod, "cuda", None)
        if not cuda_api or not cuda_api.is_available():
            return None
        idx = int(cuda_api.current_device()) if hasattr(cuda_api, "current_device") else 0
        total = float(cuda_api.get_device_properties(idx).total_memory)
        if total <= 0:
            return None
        peak = float(max(cuda_api.memory_allocated(idx), cuda_api.max_memory_allocated(idx)))
        return peak / total
    except Exception:  # noqa: BLE001
        return None


def patch_module_activation_checkpointing(
    root_module: Any,
    torch_mod: Any,
    manager: "HookManager | None" = None,
    *,
    min_block_params: int | None = None,
) -> int:
    """Wrap discovered composite blocks on `root_module` with `checkpoint(..., use_reentrant=False)`."""
    ckpt_mod = getattr(getattr(torch_mod, "utils", None), "checkpoint", None)
    ckpt_fn = getattr(ckpt_mod, "checkpoint", None)
    if not callable(ckpt_fn):
        return 0

    blocks = discover_checkpoint_blocks(root_module, torch_mod, min_block_params=min_block_params)
    patched = 0
    for blk in blocks:
        if getattr(blk, "_ptopt_act_ckpt_patched", False):
            continue
        orig_fwd = blk.forward

        def _make_ckpt_fwd(sub_blk: Any, base_fwd: Any, owner: Any) -> Any:
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
                if _is_dynamo_compiling(torch_mod):
                    return live_ckpt(base_fwd, *args, use_reentrant=False, **kwargs)
                try:
                    return live_ckpt(base_fwd, *args, use_reentrant=False, **kwargs)
                except Exception as exc:  # noqa: BLE001
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
    nn_mod = torch_mod.nn.Module
    call_attr = "__call__"
    orig_mod_call = getattr(nn_mod, call_attr)
    depth = [0]
    probe_vram = vram_probe_fn or (lambda: _default_vram_ratio(torch_mod))

    @functools.wraps(orig_mod_call)
    def _act_ckpt_mod_call(mod_self: Any, *args: Any, **kwargs: Any) -> Any:
        depth[0] += 1
        is_top = depth[0] == 1
        try:
            if (
                is_top
                and getattr(mod_self, "training", True)
                and torch_mod.is_grad_enabled()
                and not getattr(mod_self, "_ptopt_act_ckpt_no_blocks", False)
            ):
                if not getattr(mod_self, "_ptopt_act_ckpt_active", False):
                    should_enable = mode == "always"
                    vram_ratio: float | None = None
                    if not should_enable:
                        vram_ratio = probe_vram()
                        if vram_ratio is not None and vram_ratio >= rt.vram_threshold:
                            should_enable = True

                    if should_enable:
                        n_patched = patch_module_activation_checkpointing(mod_self, torch_mod, manager=manager)
                        if n_patched > 0:
                            mod_self._ptopt_act_ckpt_active = True
                            rt.triggered_modules += 1
                            rt.patched_blocks += n_patched
                            # Bypass torch.compile on root AND all child submodules (so backward recomputation
                            # inside checkpoint(..., use_reentrant=False) never compiles child blocks mid-backward,
                            # preventing PyTorch #166926 tensor count mismatch).
                            was_compiled = bool(getattr(mod_self, "_ptopt_compile_active", False))
                            for sub_m in mod_self.modules():
                                sub_m._compiled_call_impl = None
                                sub_m._ptopt_compile_active = False
                                sub_m._ptopt_compile_checked = True
                                sub_m._ptopt_compile_ineligible = True
                            if was_compiled:
                                manager.record_event(
                                    "compile.auto",
                                    "FALLBACK",
                                    f"Bypassed torch.compile on {type(mod_self).__name__} while activation checkpointing active",
                                )
                            detail = (
                                f"mode=always"
                                if mode == "always"
                                else f"mode=auto (VRAM {(vram_ratio or 0.0) * 100.0:.1f}% >= {rt.vram_threshold * 100.0:.0f}%)"
                            )
                            manager.record_event(
                                "runtime.activation_checkpointing",
                                "APPLIED",
                                f"Enabled activation checkpointing on {n_patched} blocks of {type(mod_self).__name__} ({detail})",
                            )
                        else:
                            mod_self._ptopt_act_ckpt_no_blocks = True
            return orig_mod_call(mod_self, *args, **kwargs)
        finally:
            depth[0] -= 1

    nn_mod._ptopt_act_ckpt_call_attr = call_attr
    nn_mod._ptopt_act_ckpt_orig_call = orig_mod_call
    setattr(nn_mod, call_attr, _act_ckpt_mod_call)
    _ACT_CKPT_INSTALLED = True
    torch_mod._ptopt_act_ckpt_runtime = rt
    return rt


def uninstall_activation_checkpoint_hook(torch_mod: Any) -> None:
    """Restore monkey-patched activation checkpoint hook (for unit tests)."""
    global _ACT_CKPT_INSTALLED  # noqa: PLW0603
    if not _ACT_CKPT_INSTALLED or torch_mod is None:
        return
    nn_mod = getattr(torch_mod, "nn", None) and torch_mod.nn.Module
    if nn_mod is not None and hasattr(nn_mod, "_ptopt_act_ckpt_orig_call"):
        setattr(nn_mod, getattr(nn_mod, "_ptopt_act_ckpt_call_attr"), getattr(nn_mod, "_ptopt_act_ckpt_orig_call"))
        delattr(nn_mod, "_ptopt_act_ckpt_call_attr")
        delattr(nn_mod, "_ptopt_act_ckpt_orig_call")
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
