"""Pass 3.3b: Safe Auto Mixed Precision (bf16 / fp16 + GradScaler) with double-AMP prevention."""

from __future__ import annotations

import functools
import math
import os
import sys
from typing import TYPE_CHECKING, Any

from ptopt.config import OptimizeConfig
from ptopt.hooks.module_dispatch import (
    has_cuda_context,
    register_module_call_handler,
    unregister_module_call_handler,
)

if TYPE_CHECKING:
    from ptopt.hooks.manager import HookManager

_AUTO_AMP_INSTALLED = False
_OPTIM_STEP_CLASSES = ("Adam", "AdamW", "SGD", "RMSprop", "Adagrad", "NAdam", "RAdam")
_FUNCTIONAL_LOSS_FP32_SPECS: dict[str, bool] = {
    "cross_entropy": False,
    "nll_loss": False,
    "binary_cross_entropy_with_logits": True,
    "kl_div": True,
}


def _has_cuda_context(mod_self: Any, args: tuple[Any, ...], kwargs: dict[str, Any], torch_mod: Any) -> bool:
    """Return True if module parameters or inputs live on CUDA."""
    return has_cuda_context(mod_self, args, kwargs, torch_mod)


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


def _promote_half_cuda_tensor(value: Any, half_dtypes: tuple[Any, ...]) -> Any:
    """Promote CUDA fp16/bf16 tensor to fp32 so functional losses outside autocast stay finite."""
    if hasattr(value, "_ptopt_lazy_linear"):
        return value
    if value is not None and getattr(value, "is_cuda", False) and getattr(value, "dtype", None) in half_dtypes:
        return value.float()
    return value


def _install_functional_loss_fp32_guards(torch_mod: Any, rt: AutoAmpRuntime) -> None:
    """Promote half-precision logits in functional losses invoked outside Module autocast."""
    func_mod = getattr(getattr(torch_mod, "nn", None), "functional", None)
    if func_mod is None:
        return
    half_dtypes = tuple(
        dt for dt in (getattr(torch_mod, "float16", None), getattr(torch_mod, "bfloat16", None)) if dt is not None
    )
    if not half_dtypes:
        return

    orig_losses: dict[str, Any] = {}
    for fn_name, promote_target in _FUNCTIONAL_LOSS_FP32_SPECS.items():
        orig_fn = getattr(func_mod, fn_name, None)
        if orig_fn is None or not callable(orig_fn):
            continue

        def _make_loss_wrapper(base_fn: Any, with_target: bool) -> Any:
            @functools.wraps(base_fn)
            def _fp32_loss_guard(*args: Any, **kwargs: Any) -> Any:
                if (
                    not rt.disabled_by_watchdog
                    and not rt.user_amp_detected
                    and not _is_autocast_active(torch_mod)
                ):
                    if args:
                        args_list = list(args)
                        args_list[0] = _promote_half_cuda_tensor(args_list[0], half_dtypes)
                        if with_target and len(args_list) > 1:
                            args_list[1] = _promote_half_cuda_tensor(args_list[1], half_dtypes)
                        args = tuple(args_list)
                    elif "input" in kwargs:
                        kwargs = dict(kwargs)
                        kwargs["input"] = _promote_half_cuda_tensor(kwargs["input"], half_dtypes)
                    if with_target and "target" in kwargs:
                        kwargs = dict(kwargs)
                        kwargs["target"] = _promote_half_cuda_tensor(kwargs["target"], half_dtypes)
                return base_fn(*args, **kwargs)

            return _fp32_loss_guard

        orig_losses[fn_name] = orig_fn
        setattr(func_mod, fn_name, _make_loss_wrapper(orig_fn, promote_target))

    if orig_losses:
        func_mod._ptopt_amp_orig_losses = orig_losses


def _has_conv_or_attention(mod_self: Any, torch_mod: Any) -> bool:
    nn_ns = getattr(torch_mod, "nn", None)
    if nn_ns is None or not hasattr(mod_self, "modules"):
        return False
    conv_types = tuple(
        t for t in (getattr(nn_ns, "Conv1d", None), getattr(nn_ns, "Conv2d", None), getattr(nn_ns, "Conv3d", None))
        if t is not None
    )
    mha_cls = getattr(nn_ns, "MultiheadAttention", None)
    for sub in mod_self.modules():
        if conv_types and isinstance(sub, conv_types):
            return True
        if mha_cls is not None and isinstance(sub, mha_cls):
            return True
        cls_name = type(sub).__name__.lower()
        if "attention" in cls_name or "attn" in cls_name:
            return True
    return False


def _is_cast_heavy_eager_mlp(mod_self: Any, cfg: OptimizeConfig, torch_mod: Any) -> bool:
    """Detect deep LayerNorm-interleaved MLPs (no Conv2d/Attention) where eager AMP cast churn hurts."""
    if cfg.compile.enabled or os.environ.get("PTOPT_AMP_FORCE") == "1":
        return False
    if _has_conv_or_attention(mod_self, torch_mod):
        return False
    nn_ns = getattr(torch_mod, "nn", None)
    ln_cls = getattr(nn_ns, "LayerNorm", None) if nn_ns is not None else None
    if ln_cls is None or not hasattr(mod_self, "modules"):
        return False
    ln_count = sum(1 for sub in mod_self.modules() if isinstance(sub, ln_cls))
    return ln_count >= 6


def _is_tiny_uncompiled_mlp(
    mod_self: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    torch_mod: Any,
) -> bool:
    """Return True if an uncompiled tiny MLP has too few params/elems to profit from eager autocast."""
    if os.environ.get("PTOPT_AMP_FORCE") == "1":
        return False
    if _has_conv_or_attention(mod_self, torch_mod):
        return False
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
            if isinstance(item, tensor_cls):
                try:
                    input_elems += int(item.numel())
                except Exception:  # noqa: BLE001
                    pass
    return 0 < total_params < 50_000 and input_elems < 32_768


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
        self._last_forward_skipped_amp = False
        self._module_depth = 0
        self.scaler: Any = None
        self._watchdog_buf: Any = None
        self._watchdog_events: list[Any] = [None, None]
        self._watchdog_pending_slot: int | None = None
        self._watchdog_pending_slots: list[int] = []
        self._watchdog_slot_cursor: int = 0
        self._watchdog_sync_mode: bool = (
            os.environ.get("PTOPT_AMP_WATCHDOG_SYNC") == "1"
            or bool(getattr(manager, "strict", False))
        )
        self._async_watchdog_checks: int = 0
        self._sync_watchdog_checks: int = 0
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

    def _trip_watchdog(self, val: float) -> bool:
        self.disabled_by_watchdog = True
        self._step_scaled = False
        self._watchdog_pending_slot = None
        self._watchdog_pending_slots.clear()
        self.manager.record_event(
            "amp.auto",
            "FALLBACK",
            f"Non-finite loss ({val}) detected under {self.dtype_name}; falling back to fp32",
        )
        return False

    def check_loss_stability(self, loss_tensor: Any) -> bool:
        """Return False and disable Auto-AMP if non-finite loss is detected (synchronous)."""
        if self.disabled_by_watchdog or not hasattr(loss_tensor, "numel") or loss_tensor.numel() != 1:
            return True
        self._sync_watchdog_checks += 1
        try:
            val = float(loss_tensor.detach().cpu().item())
        except Exception:  # noqa: BLE001
            return True
        if not math.isfinite(val):
            return self._trip_watchdog(val)
        return True

    def _drain_watchdog_slot(self, slot: int, *, wait: bool) -> bool:
        if self._watchdog_buf is None:
            return True
        ev = self._watchdog_events[slot]
        if ev is not None:
            if wait and callable(getattr(ev, "synchronize", None)):
                try:
                    ev.synchronize()
                except Exception:  # noqa: BLE001
                    return True
            elif not wait and callable(getattr(ev, "query", None)):
                try:
                    if not ev.query():
                        return True
                except Exception:  # noqa: BLE001
                    return True
        if slot in self._watchdog_pending_slots:
            self._watchdog_pending_slots.remove(slot)
        self._watchdog_pending_slot = self._watchdog_pending_slots[-1] if self._watchdog_pending_slots else None
        try:
            val = float(self._watchdog_buf[slot].item())
        except Exception:  # noqa: BLE001
            return True
        if not math.isfinite(val):
            return self._trip_watchdog(val)
        return True

    def _poll_pending_slot_nonblocking(self) -> bool:
        if self._watchdog_buf is None or not self._watchdog_pending_slots:
            return True
        for slot in list(self._watchdog_pending_slots):
            before = len(self._watchdog_pending_slots)
            ok = self._drain_watchdog_slot(slot, wait=False)
            if not ok:
                return False
            # Stop scanning newer slots if oldest event hasn't completed yet
            if len(self._watchdog_pending_slots) == before:
                break
        return True

    def enqueue_loss_stability_check(self, loss_tensor: Any) -> bool:
        """Stage a non-blocking D2H scalar loss check for backward() without forcing CUDA sync."""
        if self.disabled_by_watchdog or not hasattr(loss_tensor, "numel") or loss_tensor.numel() != 1:
            return True
        if self._watchdog_sync_mode or not getattr(loss_tensor, "is_cuda", False):
            return self.check_loss_stability(loss_tensor)

        if not self._poll_pending_slot_nonblocking():
            return False

        cuda_api = getattr(self.torch, "cuda", None)
        event_cls = getattr(cuda_api, "Event", None) if cuda_api is not None else None
        empty_fn = getattr(self.torch, "empty", None)
        fp32_dt = getattr(self.torch, "float32", None)
        if not callable(empty_fn) or event_cls is None or fp32_dt is None:
            return self.check_loss_stability(loss_tensor)

        try:
            if self._watchdog_buf is None:
                try:
                    self._watchdog_buf = empty_fn((2,), dtype=fp32_dt, pin_memory=True)
                except Exception:  # noqa: BLE001
                    self._watchdog_buf = empty_fn((2,), dtype=fp32_dt)
                self._watchdog_events = [event_cls(), event_cls()]

            slot = self._watchdog_slot_cursor
            if slot in self._watchdog_pending_slots:
                if not self._drain_watchdog_slot(slot, wait=True):
                    return False
            scalar = loss_tensor.detach().reshape(())
            if getattr(scalar, "dtype", None) != fp32_dt and hasattr(scalar, "float"):
                scalar = scalar.float()
            self._watchdog_buf[slot].copy_(scalar, non_blocking=True)
            self._watchdog_events[slot].record()
            self._watchdog_pending_slots.append(slot)
            self._watchdog_pending_slot = slot
            self._watchdog_slot_cursor = 1 - slot
            self._async_watchdog_checks += 1
            return True
        except Exception:  # noqa: BLE001
            self._watchdog_pending_slot = None
            self._watchdog_pending_slots.clear()
            return self.check_loss_stability(loss_tensor)

    def flush_pending_stability_check(self) -> bool:
        """Verify any queued scalar loss checks before optimizer.step() updates weights."""
        if self.disabled_by_watchdog:
            return False
        if self._watchdog_buf is None or not self._watchdog_pending_slots:
            return True
        for slot in list(self._watchdog_pending_slots):
            if not self._drain_watchdog_slot(slot, wait=True):
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

    # 2. Register top-level Module.__call__ autocast handler on unified dispatcher
    def _amp_mod_call(mod_self: Any, args: tuple[Any, ...], kwargs: dict[str, Any], next_fn: Any) -> Any:
        if rt.disabled_by_watchdog or rt.user_amp_detected:
            return next_fn(mod_self, *args, **kwargs)
        if _is_autocast_active(torch_mod):
            rt.mark_user_amp("torch.amp.autocast")
            return next_fn(mod_self, *args, **kwargs)
        if not _has_cuda_context(mod_self, args, kwargs, torch_mod):
            return next_fn(mod_self, *args, **kwargs)

        eligible = getattr(mod_self, "_ptopt_amp_eligible", None)
        if eligible is None:
            eligible = not _is_cast_heavy_eager_mlp(mod_self, cfg, torch_mod)
            mod_self._ptopt_amp_eligible = eligible
            if not eligible:
                manager.record_event(
                    "amp.auto",
                    "SKIPPED",
                    "Skipped eager AMP for deep LayerNorm MLP without Conv2d/Attention (cast overhead > compute)",
                )
        if (
            eligible
            and (getattr(mod_self, "_ptopt_compile_ineligible", False) or getattr(mod_self, "_ptopt_compile_failed", False))
            and _is_tiny_uncompiled_mlp(mod_self, args, kwargs, torch_mod)
        ):
            eligible = False
            mod_self._ptopt_amp_eligible = False
            manager.record_event(
                "amp.auto",
                "SKIPPED",
                f"Skipped eager AMP for tiny uncompiled module {type(mod_self).__name__} (autocast overhead > compute)",
            )
        is_train_fwd = bool(getattr(mod_self, "training", True) and torch_mod.is_grad_enabled())
        if not eligible:
            if is_train_fwd:
                rt._last_forward_skipped_amp = True
            return next_fn(mod_self, *args, **kwargs)

        if is_train_fwd:
            rt._last_forward_skipped_amp = False
        with torch_mod.amp.autocast(device_type="cuda", dtype=rt.target_dtype):
            return next_fn(mod_self, *args, **kwargs)

    register_module_call_handler(torch_mod, "amp.auto", 40, _amp_mod_call)

    # 3. Wrap Tensor.backward for fp16 GradScaler & zero-sync stability watchdog
    orig_backward = torch_mod.Tensor.backward

    @functools.wraps(orig_backward)
    def _amp_backward(tensor_self: Any, *args: Any, **kwargs: Any) -> Any:
        if rt.disabled_by_watchdog or rt.user_amp_detected:
            return orig_backward(tensor_self, *args, **kwargs)
        if rt._last_forward_skipped_amp:
            rt._last_forward_skipped_amp = False
            return orig_backward(tensor_self, *args, **kwargs)

        # Enqueue non-blocking stability check so forward->backward never forces a CUDA sync
        if getattr(tensor_self, "is_cuda", False) and not rt.enqueue_loss_stability_check(tensor_self):
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

    # 4. Wrap optimizer step() classes for stability check flush + optional fp16 GradScaler.step()
    optim_mod = getattr(torch_mod, "optim", None)
    if optim_mod is not None:
        for cls_name in _OPTIM_STEP_CLASSES:
            opt_cls = getattr(optim_mod, cls_name, None)
            if opt_cls is None:
                continue
            orig_step = opt_cls.step

            def _make_amp_step(base_step: Any) -> Any:
                @functools.wraps(base_step)
                def _amp_step(opt_self: Any, *args: Any, **kwargs: Any) -> Any:
                    if rt._in_internal_scaler_step or rt.user_amp_detected or rt.disabled_by_watchdog:
                        return base_step(opt_self, *args, **kwargs)
                    if not rt.flush_pending_stability_check():
                        return None
                    if not (rt.use_scaler and rt.scaler is not None and rt._step_scaled):
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

    # 5. Promote half-precision logits in functional losses invoked outside Module autocast
    _install_functional_loss_fp32_guards(torch_mod, rt)

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

    unregister_module_call_handler(torch_mod, "amp.auto")

    if hasattr(torch_mod.Tensor, "_ptopt_amp_orig_backward"):
        torch_mod.Tensor.backward = torch_mod.Tensor._ptopt_amp_orig_backward
        delattr(torch_mod.Tensor, "_ptopt_amp_orig_backward")

    func_mod = getattr(getattr(torch_mod, "nn", None), "functional", None)
    if func_mod is not None and hasattr(func_mod, "_ptopt_amp_orig_losses"):
        for fn_name, orig_fn in getattr(func_mod, "_ptopt_amp_orig_losses").items():
            setattr(func_mod, fn_name, orig_fn)
        delattr(func_mod, "_ptopt_amp_orig_losses")

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
