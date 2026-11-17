"""Pass 3.1c: Safe `channels_last` memory format conversion for Conv2d vision models."""

from __future__ import annotations

import functools
import os
import sys
from typing import TYPE_CHECKING, Any

from src.config import OptimizeConfig

if TYPE_CHECKING:
    from src.hooks.manager import HookManager

_CHANNELS_LAST_INSTALLED = False


def has_conv2d(module: Any, torch_mod: Any) -> bool:
    """Return True if `module` contains at least one `nn.Conv2d` layer."""
    conv2d_cls = getattr(getattr(torch_mod, "nn", None), "Conv2d", None)
    if conv2d_cls is None or not hasattr(module, "modules"):
        return False
    return any(isinstance(sub, conv2d_cls) for sub in module.modules())


def _convert_tensor_channels_last(obj: Any, torch_mod: Any) -> Any:
    tensor_cls = getattr(torch_mod, "Tensor", None)
    channels_last_fmt = getattr(torch_mod, "channels_last", None)
    if tensor_cls is None or channels_last_fmt is None:
        return obj
    if isinstance(obj, tensor_cls) and obj.dim() == 4:
        return obj.contiguous(memory_format=channels_last_fmt)
    if isinstance(obj, tuple):
        return tuple(_convert_tensor_channels_last(item, torch_mod) for item in obj)
    return obj


def convert_module_channels_last(module: Any, torch_mod: Any) -> bool:
    """Convert a Conv2d module and attach a 4D input pre-hook once; returns True if applied."""
    if getattr(module, "_ptopt_channels_last_done", False):
        return False
    if not has_conv2d(module, torch_mod):
        return False

    channels_last_fmt = getattr(torch_mod, "channels_last", None)
    if channels_last_fmt is None:
        return False

    module.to(memory_format=channels_last_fmt)

    def _input_pre_hook(_mod: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
        return tuple(_convert_tensor_channels_last(inp, torch_mod) for inp in inputs)

    if hasattr(module, "register_forward_pre_hook"):
        module.register_forward_pre_hook(_input_pre_hook)
    module._ptopt_channels_last_done = True
    return True


def _has_large_4d_input(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    torch_mod: Any,
    *,
    min_spatial_area: int,
    require_cuda: bool,
) -> bool:
    """Return True if any 4D Tensor input satisfies device and spatial size thresholds."""
    tensor_cls = getattr(torch_mod, "Tensor", None)
    if tensor_cls is None:
        return False
    for item in (*args, *kwargs.values()):
        if isinstance(item, tensor_cls) and item.dim() == 4:
            if require_cuda and not getattr(item, "is_cuda", False):
                continue
            if int(item.shape[-2]) * int(item.shape[-1]) >= min_spatial_area:
                return True
    return False


def install_channels_last_hook(
    torch_mod: Any,
    *,
    min_spatial_area: int = 0,
    require_cuda: bool = False,
    manager: "HookManager | None" = None,
) -> bool:
    """Intercept top-level Module forward calls to convert eligible Conv2d models on first execution."""
    global _CHANNELS_LAST_INSTALLED  # noqa: PLW0603
    if _CHANNELS_LAST_INSTALLED or torch_mod is None or not hasattr(torch_mod, "nn"):
        return False

    nn_module = torch_mod.nn.Module
    attr_name = "__call__"
    orig_call = getattr(nn_module, attr_name)
    depth = [0]

    @functools.wraps(orig_call)
    def _channels_last_call(mod_self: Any, *args: Any, **kwargs: Any) -> Any:
        depth[0] += 1
        try:
            if depth[0] == 1 and not getattr(mod_self, "_ptopt_channels_last_checked", False):
                mod_self._ptopt_channels_last_checked = True
                if has_conv2d(mod_self, torch_mod):
                    should_convert = min_spatial_area <= 0 or _has_large_4d_input(
                        args,
                        kwargs,
                        torch_mod,
                        min_spatial_area=min_spatial_area,
                        require_cuda=require_cuda,
                    )
                    if should_convert:
                        if convert_module_channels_last(mod_self, torch_mod) and manager is not None:
                            manager.record_event(
                                "runtime.channels_last",
                                "APPLIED",
                                f"Converted Conv2d module to channels_last (min_spatial_area={min_spatial_area})",
                            )
                    elif manager is not None:
                        manager.record_event(
                            "runtime.channels_last",
                            "SKIPPED",
                            f"Conv2d spatial input below auto threshold ({min_spatial_area} px)",
                        )
            return orig_call(mod_self, *args, **kwargs)
        finally:
            depth[0] -= 1

    setattr(nn_module, attr_name, _channels_last_call)
    nn_module._ptopt_orig_channels_last_attr = attr_name  # type: ignore[attr-defined]
    nn_module._ptopt_orig_channels_last_call = orig_call  # type: ignore[attr-defined]
    _CHANNELS_LAST_INSTALLED = True
    return True


def uninstall_channels_last_hook(torch_mod: Any) -> None:
    """Restore original `nn.Module` call wrapper (for unit tests)."""
    global _CHANNELS_LAST_INSTALLED  # noqa: PLW0603
    if not _CHANNELS_LAST_INSTALLED or torch_mod is None or not hasattr(torch_mod, "nn"):
        return
    nn_module = torch_mod.nn.Module
    if hasattr(nn_module, "_ptopt_orig_channels_last_call"):
        attr = getattr(nn_module, "_ptopt_orig_channels_last_attr")
        setattr(nn_module, attr, getattr(nn_module, "_ptopt_orig_channels_last_call"))
        delattr(nn_module, "_ptopt_orig_channels_last_attr")
        delattr(nn_module, "_ptopt_orig_channels_last_call")
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
