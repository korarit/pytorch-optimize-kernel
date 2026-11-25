"""Pass 3.1c: Safe `channels_last` memory format conversion for Conv2d vision models."""

from __future__ import annotations

import functools
import os
import sys
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

from ptopt.config import OptimizeConfig
from ptopt.hooks.module_dispatch import (
    iter_nested_tensors,
    register_module_call_handler,
    unregister_module_call_handler,
)

if TYPE_CHECKING:
    from ptopt.hooks.manager import HookManager

_CHANNELS_LAST_INSTALLED = False


def has_conv2d(module: Any, torch_mod: Any) -> bool:
    """Return True if `module` contains at least one `nn.Conv2d` layer."""
    conv2d_cls = getattr(getattr(torch_mod, "nn", None), "Conv2d", None)
    if conv2d_cls is None or not hasattr(module, "modules"):
        return False
    return any(isinstance(sub, conv2d_cls) for sub in module.modules())


def has_conv3d(module: Any, torch_mod: Any) -> bool:
    """Return True if `module` contains at least one `nn.Conv3d` layer."""
    conv3d_cls = getattr(getattr(torch_mod, "nn", None), "Conv3d", None)
    if conv3d_cls is None or not hasattr(module, "modules"):
        return False
    return any(isinstance(sub, conv3d_cls) for sub in module.modules())


def _convert_tensor_channels_last(
    obj: Any,
    torch_mod: Any,
    *,
    allow_2d: bool = True,
    allow_3d: bool = False,
) -> Any:
    tensor_cls = getattr(torch_mod, "Tensor", None)
    channels_last_fmt = getattr(torch_mod, "channels_last", None)
    channels_last_3d_fmt = getattr(torch_mod, "channels_last_3d", None)
    if tensor_cls is None:
        return obj
    if isinstance(obj, tensor_cls):
        if allow_2d and obj.dim() == 4 and channels_last_fmt is not None:
            return obj.contiguous(memory_format=channels_last_fmt)
        if allow_3d and obj.dim() == 5 and channels_last_3d_fmt is not None:
            return obj.contiguous(memory_format=channels_last_3d_fmt)
        return obj
    if isinstance(obj, tuple):
        return tuple(
            _convert_tensor_channels_last(item, torch_mod, allow_2d=allow_2d, allow_3d=allow_3d)
            for item in obj
        )
    return obj


def convert_module_channels_last(module: Any, torch_mod: Any) -> bool:
    """Convert a Conv2d/Conv3d module and attach a channels_last input pre-hook once; returns True if applied."""
    if getattr(module, "_ptopt_channels_last_done", False):
        return False
    is_c2d = has_conv2d(module, torch_mod)
    is_c3d = has_conv3d(module, torch_mod)
    if not is_c2d and not is_c3d:
        return False

    applied = False
    channels_last_fmt = getattr(torch_mod, "channels_last", None)
    channels_last_3d_fmt = getattr(torch_mod, "channels_last_3d", None)
    if is_c2d and is_c3d:
        no_grad_fn = getattr(torch_mod, "no_grad", None)
        ctx = no_grad_fn() if callable(no_grad_fn) else nullcontext()
        with ctx:
            for tensor in (*module.parameters(), *module.buffers()):
                if tensor.dim() == 4 and channels_last_fmt is not None:
                    tensor.data = tensor.data.contiguous(memory_format=channels_last_fmt)
                    applied = True
                elif tensor.dim() == 5 and channels_last_3d_fmt is not None:
                    tensor.data = tensor.data.contiguous(memory_format=channels_last_3d_fmt)
                    applied = True
    elif is_c2d and channels_last_fmt is not None:
        module.to(memory_format=channels_last_fmt)
        applied = True
    elif is_c3d and channels_last_3d_fmt is not None:
        module.to(memory_format=channels_last_3d_fmt)
        applied = True
    if not applied:
        return False

    from ptopt.hooks.checkpoint_compat import sync_all_live_optimizers  # noqa: PLC0415

    sync_all_live_optimizers(torch_mod)

    def _input_pre_hook(_mod: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
        return tuple(
            _convert_tensor_channels_last(inp, torch_mod, allow_2d=is_c2d, allow_3d=is_c3d)
            for inp in inputs
        )

    def _input_pre_hook_kw(
        _mod: Any,
        inputs: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        conv_args = tuple(
            _convert_tensor_channels_last(inp, torch_mod, allow_2d=is_c2d, allow_3d=is_c3d)
            for inp in inputs
        )
        conv_kw = (
            {k: _convert_tensor_channels_last(v, torch_mod, allow_2d=is_c2d, allow_3d=is_c3d) for k, v in kwargs.items()}
            if kwargs
            else kwargs
        )
        return conv_args, conv_kw

    if hasattr(module, "register_forward_pre_hook"):
        try:
            module.register_forward_pre_hook(_input_pre_hook_kw, with_kwargs=True)
        except TypeError:
            module.register_forward_pre_hook(_input_pre_hook)
    for sub_m in module.modules():
        sub_m._ptopt_channels_last_done = True
    module._ptopt_channels_last_3d = is_c3d
    return True


def _is_conv_dominated_architecture(mod_self: Any, torch_mod: Any) -> bool:
    """Skip ViT/Attention architectures that only use 1-2 Conv2d layers as patch embeddings."""
    nn_ns = getattr(torch_mod, "nn", None)
    if nn_ns is None or not hasattr(mod_self, "modules"):
        return True
    c2d_cls = getattr(nn_ns, "Conv2d", None)
    c3d_cls = getattr(nn_ns, "Conv3d", None)
    mha_cls = getattr(nn_ns, "MultiheadAttention", None)
    conv_count = 0
    has_attention = False
    for sub in mod_self.modules():
        if (c2d_cls is not None and isinstance(sub, c2d_cls)) or (c3d_cls is not None and isinstance(sub, c3d_cls)):
            conv_count += 1
        elif mha_cls is not None and isinstance(sub, mha_cls):
            has_attention = True
        else:
            cls_name = type(sub).__name__.lower()
            if "attention" in cls_name or "attn" in cls_name:
                has_attention = True
    return not (has_attention and conv_count <= 2)


def _has_large_4d_input(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    torch_mod: Any,
    *,
    min_spatial_area: int,
    require_cuda: bool,
    allow_3d: bool = False,
) -> bool:
    """Return True if any 4D/5D Tensor satisfies device and spatial/volume thresholds."""
    tensor_cls = getattr(torch_mod, "Tensor", None)
    if tensor_cls is None:
        return False
    for item in (*args, *kwargs.values()):
        for tensor in iter_nested_tensors(item, tensor_cls):
            if require_cuda and not getattr(tensor, "is_cuda", False):
                continue
            if tensor.dim() == 4:
                # Skip 1-channel or 1D-height/width EEG/signal tensors (e.g. EEGNet (B, 1, C_eeg, T))
                if int(tensor.shape[1]) <= 1 or int(tensor.shape[-2]) <= 1 or int(tensor.shape[-1]) <= 1:
                    continue
                spatial = int(tensor.shape[-2]) * int(tensor.shape[-1])
                bc = int(tensor.shape[0]) * int(tensor.shape[1])
                if spatial >= min_spatial_area:
                    return True
                # High-channel feature/batch heuristic (e.g. B*C>=4096 & H*W>=4096) for standard auto gate
                if min_spatial_area == 16384 and bc >= 4096 and spatial >= 4096:
                    return True
            elif allow_3d and tensor.dim() == 5:
                vol = int(tensor.shape[-3]) * int(tensor.shape[-2]) * int(tensor.shape[-1])
                if vol >= max(4096, min_spatial_area // 2):
                    return True
    return False


def install_channels_last_hook(
    torch_mod: Any,
    *,
    min_spatial_area: int = 0,
    require_cuda: bool = False,
    manager: "HookManager | None" = None,
) -> bool:
    """Intercept top-level Module forward calls to convert eligible Conv2d/Conv3d models on first execution."""
    global _CHANNELS_LAST_INSTALLED  # noqa: PLW0603
    if _CHANNELS_LAST_INSTALLED or torch_mod is None or not hasattr(torch_mod, "nn"):
        return False

    def _channels_last_call(mod_self: Any, args: tuple[Any, ...], kwargs: dict[str, Any], next_fn: Any) -> Any:
        if not getattr(mod_self, "_ptopt_channels_last_checked", False):
            mod_self._ptopt_channels_last_checked = True
            is_c2d = has_conv2d(mod_self, torch_mod)
            is_c3d = has_conv3d(mod_self, torch_mod)
            if is_c2d or is_c3d:
                if not _is_conv_dominated_architecture(mod_self, torch_mod):
                    if manager is not None:
                        manager.record_event(
                            "runtime.channels_last",
                            "SKIPPED",
                            "Skipped channels_last for Attention/ViT module with <= 2 Conv layers (patch embed)",
                        )
                    return next_fn(mod_self, *args, **kwargs)
                should_convert = min_spatial_area <= 0 or _has_large_4d_input(
                    args,
                    kwargs,
                    torch_mod,
                    min_spatial_area=min_spatial_area,
                    require_cuda=require_cuda,
                    allow_3d=is_c3d,
                )
                if should_convert:
                    if convert_module_channels_last(mod_self, torch_mod) and manager is not None:
                        if is_c2d and is_c3d:
                            kind = "Conv2d+Conv3d channels_last/channels_last_3d"
                        elif is_c3d:
                            kind = "Conv3d channels_last_3d"
                        else:
                            kind = "Conv2d channels_last"
                        manager.record_event(
                            "runtime.channels_last",
                            "APPLIED",
                            f"Converted {kind} module (min_spatial_area={min_spatial_area})",
                        )
                elif manager is not None:
                    manager.record_event(
                        "runtime.channels_last",
                        "SKIPPED",
                        f"Conv spatial/volume input below auto threshold ({min_spatial_area} px)",
                    )
        return next_fn(mod_self, *args, **kwargs)

    register_module_call_handler(torch_mod, "runtime.channels_last", 50, _channels_last_call)
    _CHANNELS_LAST_INSTALLED = True
    return True


def uninstall_channels_last_hook(torch_mod: Any) -> None:
    """Restore original `nn.Module` call wrapper (for unit tests)."""
    global _CHANNELS_LAST_INSTALLED  # noqa: PLW0603
    if not _CHANNELS_LAST_INSTALLED or torch_mod is None or not hasattr(torch_mod, "nn"):
        return
    unregister_module_call_handler(torch_mod, "runtime.channels_last")
    _CHANNELS_LAST_INSTALLED = False


def apply_channels_last_pass(
    cfg: OptimizeConfig,
    manager: "HookManager",
    torch_mod: Any = None,
) -> str:
    """Enable Conv2d channels_last pass when forced (`=1`) or auto-enabled for large CUDA Conv2d."""
    env_mode = os.environ.get("PTOPT_CHANNELS_LAST")
    if env_mode == "0":
        return "SKIPPED: disabled via PTOPT_CHANNELS_LAST=0"

    tmod = torch_mod if torch_mod is not None else sys.modules.get("torch")
    if tmod is None:
        return "SKIPPED: torch not loaded"

    if env_mode == "1":
        installed = install_channels_last_hook(tmod, min_spatial_area=0, require_cuda=False)
        return "channels_last Conv2d hook installed (forced)" if installed else "channels_last hook already active"

    # Auto mode: Phase 3.4 profiler showed 13.7% NCHW<->NHWC cuDNN overhead on large (>=128x128) CUDA CNNs
    has_cuda = bool(getattr(getattr(tmod, "cuda", None), "is_available", lambda: False)())
    if cfg.profile not in ("safe", "balanced", "aggressive") or not has_cuda:
        return "SKIPPED: opt-in only on CPU/supersafe (set PTOPT_CHANNELS_LAST=1)"

    installed = install_channels_last_hook(
        tmod,
        min_spatial_area=128 * 128,
        require_cuda=True,
        manager=manager,
    )
    return (
        "Auto channels_last hook installed (CUDA Conv2d spatial>=128x128)"
        if installed
        else "channels_last hook already active"
    )
