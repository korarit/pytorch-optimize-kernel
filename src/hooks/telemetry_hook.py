"""Non-invasive runtime telemetry hook capturing step latency, losses, memory, and fallback events."""

from __future__ import annotations

import atexit
import functools
import importlib
import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.hooks.manager import HookManager


class TelemetryCollector:
    """Collects per-step timing, scalar losses, peak memory, and fallback events."""

    def __init__(self, out_path: Path, manager: "HookManager") -> None:
        self.out_path = out_path
        self.manager = manager
        self.t_start = time.perf_counter()
        self.step_times_ms: list[float] = []
        self.losses: list[float] = []
        self._last_step_end: float | None = None
        self._optimizer_used: bool = False
        self._module_depth: int = 0
        self._installed: bool = False
        self._flushed: bool = False
        self._torch: Any = None
        self._opt_hook_handle: Any = None
        self._orig_backward: Any = None
        self._orig_mod_call: Any = None
        self._mod_call_attr: str | None = None

    def install(self, torch_mod: Any) -> None:
        """Install non-invasive telemetry hooks once on Tensor.backward, Optimizer step, and Module eval."""
        if self._installed or torch_mod is None:
            return
        self._installed = True
        self._torch = torch_mod
        collector = self

        # 1. Capture scalar loss on backward()
        self._orig_backward = torch_mod.Tensor.backward
        orig_backward = self._orig_backward

        @functools.wraps(orig_backward)
        def _wrapped_backward(tensor_self: Any, *args: Any, **kwargs: Any) -> Any:
            try:
                if tensor_self.numel() == 1:
                    collector.losses.append(float(tensor_self.detach().cpu().item()))
            except Exception:  # noqa: BLE001
                pass
            return orig_backward(tensor_self, *args, **kwargs)

        torch_mod.Tensor.backward = _wrapped_backward

        # 2. Capture training step duration via PyTorch global optimizer step post-hook
        try:
            opt_mod = importlib.import_module("torch.optim.optimizer")
            reg_post_hook = getattr(opt_mod, "register_optimizer_step_post_hook", None)
            if callable(reg_post_hook):
                def _opt_post_hook(_opt: Any, _args: Any, _kwargs: Any) -> None:
                    collector._optimizer_used = True
                    if collector._last_step_end is None:
                        collector._last_step_end = collector.t_start
                    _sync_if_cuda(torch_mod)
                    now = time.perf_counter()
                    collector.step_times_ms.append(round((now - collector._last_step_end) * 1000.0, 4))
                    collector._last_step_end = now

                self._opt_hook_handle = reg_post_hook(_opt_post_hook)
        except Exception:  # noqa: BLE001
            self._opt_hook_handle = None

        # 3. Capture top-level inference step duration when Optimizer.step is unused
        nn_mod = torch_mod.nn.Module
        attr_name = "_wrapped_call_impl" if hasattr(nn_mod, "_wrapped_call_impl") else "__call__"
        self._mod_call_attr = attr_name
        self._orig_mod_call = getattr(nn_mod, attr_name)
        orig_mod_call = self._orig_mod_call

        @functools.wraps(orig_mod_call)
        def _wrapped_mod_call(mod_self: Any, *args: Any, **kwargs: Any) -> Any:
            collector._module_depth += 1
            is_top = collector._module_depth == 1
            t0 = time.perf_counter() if is_top else 0.0
            try:
                out = orig_mod_call(mod_self, *args, **kwargs)
                if is_top and not collector._optimizer_used and not getattr(mod_self, "training", True):
                    _sync_if_cuda(torch_mod)
                    dt_ms = (time.perf_counter() - t0) * 1000.0
                    collector.step_times_ms.append(round(dt_ms, 4))
                return out
            finally:
                collector._module_depth -= 1

        setattr(nn_mod, attr_name, _wrapped_mod_call)
        atexit.register(self.flush)

    def uninstall(self) -> None:
        """Restore monkey-patched hooks (for clean unit tests)."""
        if not self._installed or self._torch is None:
            return
        if self._orig_backward is not None:
            self._torch.Tensor.backward = self._orig_backward
        if self._opt_hook_handle is not None:
            self._opt_hook_handle.remove()
            self._opt_hook_handle = None
        if self._mod_call_attr and self._orig_mod_call is not None:
            setattr(self._torch.nn.Module, self._mod_call_attr, self._orig_mod_call)
        self._installed = False

    def snapshot(self) -> dict[str, Any]:
        """Build raw telemetry payload dict."""
        torch_mod = self._torch
        peak_alloc_mb = 0.0
        peak_resv_mb = 0.0
        if torch_mod is not None:
            try:
                if torch_mod.cuda.is_available():
                    peak_alloc_mb = round(torch_mod.cuda.max_memory_allocated() / (1024.0 * 1024.0), 2)
                    peak_resv_mb = round(torch_mod.cuda.max_memory_reserved() / (1024.0 * 1024.0), 2)
            except Exception:  # noqa: BLE001
                pass

        return {
            "telemetry_schema_version": 1,
            "mode": self.manager.config.profile if self.manager.config else "unknown",
            "end_to_end_sec": round(time.perf_counter() - self.t_start, 4),
            "step_times_ms": list(self.step_times_ms),
            "losses": [round(x, 6) for x in self.losses],
            "memory_mb": {
                "peak_allocated": peak_alloc_mb,
                "peak_reserved": peak_resv_mb,
            },
            "fallback_events": [ev.to_dict() for ev in self.manager.events],
        }

    def flush(self) -> None:
        """Write collected telemetry JSON once to `self.out_path`."""
        if self._flushed:
            return
        self._flushed = True
        try:
            self.out_path.parent.mkdir(parents=True, exist_ok=True)
            self.out_path.write_text(json.dumps(self.snapshot(), indent=2) + "\n", encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass


def _sync_if_cuda(torch_mod: Any) -> None:
    try:
        if torch_mod.cuda.is_available():
            torch_mod.cuda.synchronize()
    except Exception:  # noqa: BLE001
        pass


def maybe_install_telemetry_hook(torch_mod: Any, manager: "HookManager", env: dict[str, str] | None = None) -> TelemetryCollector | None:
    """Install TelemetryCollector if `PTOPT_TELEMETRY_OUT` is configured."""
    env_map = os.environ if env is None else env
    out_raw = env_map.get("PTOPT_TELEMETRY_OUT")
    if not out_raw:
        return None
    collector = TelemetryCollector(Path(out_raw), manager)
    collector.install(torch_mod)
    return collector
