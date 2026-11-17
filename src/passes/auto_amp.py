"""Pass 3.3b: Safe Auto Mixed Precision (bf16 / fp16 + GradScaler) with double-AMP prevention."""

from __future__ import annotations

import functools
import math
import sys
from typing import TYPE_CHECKING, Any

from src.config import OptimizeConfig

if TYPE_CHECKING:
    from src.hooks.manager import HookManager

_AUTO_AMP_INSTALLED = False
_OPTIM_STEP_CLASSES = ("Adam", "AdamW", "SGD", "RMSprop", "Adagrad")


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


def _is_autocast_active(torch_mod: Any) -> bool:
    """Check whether CUDA autocast is already active in user code."""
    fn_new = getattr(torch_mod, "is_autocast_enabled", None)
    if callable(fn_new):
        try:
            return bool(fn_new("cuda"))
        except TypeError:
            return bool(fn_new())
    return False


def _resolve_torch_dtype(torch_mod: Any, dtype_name: str) -> Any:
    if dtype_name == "bf16":
        return getattr(torch_mod, "bfloat16", None)
    if dtype_name == "fp16":
        return getattr(torch_mod, "float16", None)
    return getattr(torch_mod, "float32", None)


class AutoAmpRuntime:
    """Coordinates non-invasive autocast, GradScaler, and user-AMP detection."""

    def __init__(self, cfg: OptimizeConfig, manager: "HookManager", torch_mod: Any) -> None:
        self.cfg = cfg
        self.manager = manager
        self.torch = torch_mod
        self.dtype_name = cfg.amp.dtype
        self.target_dtype = _resolve_torch_dtype(torch_mod, cfg.amp.dtype)
        self.use_scaler = bool(cfg.amp.grad_scaler and cfg.amp.dtype == "fp16")
        self.user_amp_detected = False
        self.disabled_by_watchdog = False
        self._in_internal_scaler_init = False
        self._in_internal_scaler_step = False
        self._step_scaled = False
        self._unscaled_this_step = False
        self._module_depth = 0
        self.scaler: Any = None
        if self.use_scaler:
            self._init_internal_scaler()

    def _init_internal_scaler(self) -> None:
        self._in_internal_scaler_init = True
        try:
            amp_ns = getattr(self.torch, "amp", None)
            scaler_cls = getattr(amp_ns, "GradScaler", None)
            if scaler_cls is not None:
                self.scaler = scaler_cls("cuda", enabled=True)
            else:
                cuda_amp = getattr(getattr(self.torch, "cuda", None), "amp", None)
                legacy_cls = getattr(cuda_amp, "GradScaler", None)
                if legacy_cls is not None:
                    self.scaler = legacy_cls(enabled=True)
        finally:
            self._in_internal_scaler_init = False

    def mark_user_amp(self, source: str) -> None:
        if not self.user_amp_detected:
            self.user_amp_detected = True
            self.manager.record_event(
                "amp.auto",
                "SKIPPED",
                f"Pre-existing user AMP detected ({source}); yielding to user code",
            )

    def check_loss_stability(self, loss_tensor: Any) -> bool:
        """Return False and disable Auto-AMP if non-finite loss is detected."""
        if self.disabled_by_watchdog or not hasattr(loss_tensor, "numel") or loss_tensor.numel() != 1:
            return True
        try:
            val = float(loss_tensor.detach().cpu().item())
        except Exception:  # noqa: BLE001
            return True
        if not math.isfinite(val):
            self.disabled_by_watchdog = True
            self._step_scaled = False
            self.manager.record_event(
                "amp.auto",
                "FALLBACK",
                f"Non-finite loss ({val}) detected under {self.dtype_name}; falling back to fp32",
            )
            return False
        return True


def install_auto_amp_hook(
    cfg: OptimizeConfig,
    manager: "HookManager",
    torch_mod: Any,
) -> AutoAmpRuntime | None:
    """Install idempotent Auto-AMP hooks across Module forward, GradScaler, backward, and Optimizer.step."""
    global _AUTO_AMP_INSTALLED  # noqa: PLW0603
    if _AUTO_AMP_INSTALLED or torch_mod is None or not hasattr(torch_mod, "nn"):
        return None

    rt = AutoAmpRuntime(cfg, manager, torch_mod)
    if rt.target_dtype is None:
        return None

    # 1. Detect user-created GradScaler (amp.GradScaler and legacy cuda.amp.GradScaler)
    scaler_classes: list[Any] = []
    amp_grad_scaler = getattr(getattr(torch_mod, "amp", None), "GradScaler", None)
    if amp_grad_scaler is not None:
        scaler_classes.append(amp_grad_scaler)
    cuda_grad_scaler = getattr(getattr(getattr(torch_mod, "cuda", None), "amp", None), "GradScaler", None)
    if cuda_grad_scaler is not None and cuda_grad_scaler not in scaler_classes:
        scaler_classes.append(cuda_grad_scaler)

    for s_cls in scaler_classes:
        orig_s_init = s_cls.__init__

        @functools.wraps(orig_s_init)
        def _detect_user_scaler(s_self: Any, *args: Any, _base: Any = orig_s_init, **kwargs: Any) -> None:
            _base(s_self, *args, **kwargs)
            if not rt._in_internal_scaler_init and getattr(s_self, "is_enabled", lambda: True)():
                rt.mark_user_amp("GradScaler")

        s_cls._ptopt_orig_init = orig_s_init
        s_cls.__init__ = _detect_user_scaler

    # 2. Wrap top-level Module.__call__ in autocast when active and not already in user autocast
    nn_mod = torch_mod.nn.Module
    call_attr = "__call__"
    orig_mod_call = getattr(nn_mod, call_attr)

    @functools.wraps(orig_mod_call)
    def _amp_mod_call(mod_self: Any, *args: Any, **kwargs: Any) -> Any:
        rt._module_depth += 1
        is_top = rt._module_depth == 1
        try:
            if not is_top or rt.disabled_by_watchdog or rt.user_amp_detected:
                return orig_mod_call(mod_self, *args, **kwargs)
            if _is_autocast_active(torch_mod):
                rt.mark_user_amp("torch.amp.autocast")
                return orig_mod_call(mod_self, *args, **kwargs)
            if not _has_cuda_context(mod_self, args, kwargs, torch_mod):
                return orig_mod_call(mod_self, *args, **kwargs)

            with torch_mod.amp.autocast(device_type="cuda", dtype=rt.target_dtype):
                return orig_mod_call(mod_self, *args, **kwargs)
        finally:
            rt._module_depth -= 1

    nn_mod._ptopt_amp_call_attr = call_attr
    nn_mod._ptopt_amp_orig_call = orig_mod_call
    setattr(nn_mod, call_attr, _amp_mod_call)

    # 3. Wrap Tensor.backward for fp16 GradScaler & stability watchdog
    orig_backward = torch_mod.Tensor.backward

    @functools.wraps(orig_backward)
    def _amp_backward(tensor_self: Any, *args: Any, **kwargs: Any) -> Any:
        if rt.disabled_by_watchdog or rt.user_amp_detected:
            return orig_backward(tensor_self, *args, **kwargs)

        # Only check stability on scalar losses produced when CUDA autocast or fp16 scaler applies
        if getattr(tensor_self, "is_cuda", False) and not rt.check_loss_stability(tensor_self):
            return orig_backward(tensor_self, *args, **kwargs)

        if (
            rt.use_scaler
            and rt.scaler is not None
            and getattr(tensor_self, "is_cuda", False)
            and tensor_self.numel() == 1
            and not args
            and not kwargs
        ):
            rt._step_scaled = True
            rt._unscaled_this_step = False
            scaled = rt.scaler.scale(tensor_self)
            return orig_backward(scaled)

        return orig_backward(tensor_self, *args, **kwargs)

    torch_mod.Tensor._ptopt_amp_orig_backward = orig_backward
    torch_mod.Tensor.backward = _amp_backward

    # 4. Wrap optimizer step() classes for fp16 GradScaler.step() + update()
    optim_mod = getattr(torch_mod, "optim", None)
    if optim_mod is not None and rt.use_scaler and rt.scaler is not None:
        for cls_name in _OPTIM_STEP_CLASSES:
            opt_cls = getattr(optim_mod, cls_name, None)
            if opt_cls is None:
                continue
            orig_step = opt_cls.step

            def _make_amp_step(base_step: Any) -> Any:
                @functools.wraps(base_step)
                def _amp_step(opt_self: Any, *args: Any, **kwargs: Any) -> Any:
                    if rt._in_internal_scaler_step or not rt._step_scaled or rt.user_amp_detected or rt.disabled_by_watchdog:
                        return base_step(opt_self, *args, **kwargs)
                    rt._in_internal_scaler_step = True
                    opt_self._ptopt_in_scaler_step = True
                    try:
                        res = rt.scaler.step(opt_self, *args, **kwargs)
                        rt.scaler.update()
                        rt._step_scaled = False
                        rt._unscaled_this_step = False
                        return res
                    finally:
                        opt_self._ptopt_in_scaler_step = False
                        rt._in_internal_scaler_step = False

                return _amp_step

            opt_cls._ptopt_amp_orig_step = orig_step
            opt_cls.step = _make_amp_step(orig_step)

    _AUTO_AMP_INSTALLED = True
    torch_mod._ptopt_amp_runtime = rt
    return rt


def uninstall_auto_amp_hook(torch_mod: Any) -> None:
    """Restore monkey-patched AMP hooks (for unit tests)."""
    global _AUTO_AMP_INSTALLED  # noqa: PLW0603
    if not _AUTO_AMP_INSTALLED or torch_mod is None:
        return

    for ns_cls in (
        getattr(getattr(torch_mod, "amp", None), "GradScaler", None),
        getattr(getattr(getattr(torch_mod, "cuda", None), "amp", None), "GradScaler", None),
    ):
        if ns_cls is not None and hasattr(ns_cls, "_ptopt_orig_init"):
            ns_cls.__init__ = ns_cls._ptopt_orig_init
            delattr(ns_cls, "_ptopt_orig_init")

    nn_mod = getattr(torch_mod, "nn", None) and torch_mod.nn.Module
    if nn_mod is not None and hasattr(nn_mod, "_ptopt_amp_orig_call"):
        setattr(nn_mod, getattr(nn_mod, "_ptopt_amp_call_attr"), getattr(nn_mod, "_ptopt_amp_orig_call"))
        delattr(nn_mod, "_ptopt_amp_call_attr")
        delattr(nn_mod, "_ptopt_amp_orig_call")

    if hasattr(torch_mod.Tensor, "_ptopt_amp_orig_backward"):
        torch_mod.Tensor.backward = torch_mod.Tensor._ptopt_amp_orig_backward
        delattr(torch_mod.Tensor, "_ptopt_amp_orig_backward")

    optim_mod = getattr(torch_mod, "optim", None)
    if optim_mod is not None:
        for cls_name in _OPTIM_STEP_CLASSES:
            opt_cls = getattr(optim_mod, cls_name, None)
            if opt_cls is not None and hasattr(opt_cls, "_ptopt_amp_orig_step"):
                opt_cls.step = opt_cls._ptopt_amp_orig_step
                delattr(opt_cls, "_ptopt_amp_orig_step")

    if hasattr(torch_mod, "_ptopt_amp_runtime"):
        delattr(torch_mod, "_ptopt_amp_runtime")
    _AUTO_AMP_INSTALLED = False


def apply_auto_amp_pass(
    cfg: OptimizeConfig,
    manager: "HookManager",
    torch_mod: Any = None,
) -> str:
    """Enable Safe Auto-AMP pass when configured and CUDA is available."""
    if not cfg.amp.enabled or cfg.amp.dtype == "fp32":
        return "SKIPPED: amp.enabled is False or dtype is fp32"
    tmod = torch_mod if torch_mod is not None else sys.modules.get("torch")
    if tmod is None:
        return "SKIPPED: torch not loaded"

    cuda_avail = False
    try:
        cuda_avail = bool(getattr(tmod, "cuda", None) and tmod.cuda.is_available())
    except Exception:  # noqa: BLE001
        cuda_avail = False
    if not cuda_avail:
        return "SKIPPED: CUDA unavailable for Auto-AMP"

    rt = install_auto_amp_hook(cfg, manager, tmod)
    if rt is None:
        return "Auto-AMP hook already active"
    return f"Auto-AMP installed (dtype={cfg.amp.dtype}, grad_scaler={rt.use_scaler})"
