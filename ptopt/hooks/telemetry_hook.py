"""Non-invasive runtime telemetry hook capturing step latency, losses, memory, fallback events, and opt-in debug profiling."""

from __future__ import annotations

import atexit
import functools
import importlib
import json
import os
import statistics
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ptopt.hooks.manager import HookManager


def _new_step_bucket() -> dict[str, Any]:
    return {
        "in_step": False,
        "data_other_ms": 0.0,
        "forward_ms": 0.0,
        "loss_clip_ms": 0.0,
        "backward_ms": 0.0,
        "optim_ms": 0.0,
        "last_phase_end": None,
        "t_opt_start": None,
    }


def _extract_cuda_memory_debug(torch_mod: Any) -> dict[str, Any] | None:
    """Collect detailed CUDA allocator & VRAM utilization stats if CUDA is active."""
    try:
        if not torch_mod.cuda.is_available():
            return None
        peak_alloc_mb = round(torch_mod.cuda.max_memory_allocated() / (1024.0 * 1024.0), 2)
        peak_resv_mb = round(torch_mod.cuda.max_memory_reserved() / (1024.0 * 1024.0), 2)
        total_vram_mb = round(torch_mod.cuda.get_device_properties(0).total_memory / (1024.0 * 1024.0), 2)
        mem_stats = torch_mod.cuda.memory_stats()
        active_peak_mb = round(float(mem_stats.get("active_bytes.all.peak", 0)) / (1024.0 * 1024.0), 2)
        return {
            "total_vram_mb": total_vram_mb,
            "peak_allocated_mb": peak_alloc_mb,
            "peak_reserved_mb": peak_resv_mb,
            "reserved_overhead_mb": round(max(0.0, peak_resv_mb - peak_alloc_mb), 2),
            "active_peak_mb": active_peak_mb,
            "headroom_mb": round(max(0.0, total_vram_mb - peak_resv_mb), 2),
            "utilization_pct": round((peak_resv_mb / total_vram_mb) * 100.0, 2) if total_vram_mb > 0 else 0.0,
            "num_alloc_retries": int(mem_stats.get("num_alloc_retries", 0)),
            "num_ooms": int(mem_stats.get("num_ooms", 0)),
        }
    except Exception:  # noqa: BLE001
        return None


def _summarize_profiler_ops(prof: Any, has_cuda: bool, limit: int | None = None) -> dict[str, Any]:
    """Extract top CPU and CUDA ops from an exited `torch.profiler.profile` instance."""
    if limit is None:
        try:
            limit = max(1, min(50, int(os.environ.get("PTOPT_TORCH_PROFILE_TOP_K", "20"))))
        except ValueError:
            limit = 20
    events = list(prof.key_averages())
    sort_attr = "self_cuda_time_total" if has_cuda else "self_cpu_time_total"
    # PyTorch 2.x also exposes `self_device_time_total` as the unified CUDA/XPU field
    def _cuda_self_us(ev: Any) -> float:
        return float(getattr(ev, "self_device_time_total", getattr(ev, "self_cuda_time_total", 0.0)) or 0.0)

    def _cuda_total_us(ev: Any) -> float:
        return float(getattr(ev, "device_time_total", getattr(ev, "cuda_time_total", 0.0)) or 0.0)

    def _cpu_self_us(ev: Any) -> float:
        return float(getattr(ev, "self_cpu_time_total", 0.0) or 0.0)

    def _cpu_total_us(ev: Any) -> float:
        return float(getattr(ev, "cpu_time_total", 0.0) or 0.0)

    total_self_cuda_us = sum(_cuda_self_us(ev) for ev in events)
    total_self_cpu_us = sum(_cpu_self_us(ev) for ev in events)

    top_cuda: list[dict[str, Any]] = []
    if has_cuda and total_self_cuda_us > 0:
        ranked_cuda = sorted(events, key=_cuda_self_us, reverse=True)[:limit]
        for ev in ranked_cuda:
            self_us = _cuda_self_us(ev)
            if self_us <= 0:
                continue
            top_cuda.append({
                "name": str(getattr(ev, "key", "unknown"))[:48],
                "calls": int(getattr(ev, "count", 0)),
                "self_cuda_ms": round(self_us / 1000.0, 3),
                "total_cuda_ms": round(_cuda_total_us(ev) / 1000.0, 3),
                "pct": round((self_us / total_self_cuda_us) * 100.0, 1),
            })

    top_cpu: list[dict[str, Any]] = []
    ranked_cpu = sorted(events, key=_cpu_self_us, reverse=True)[:limit]
    for ev in ranked_cpu:
        self_us = _cpu_self_us(ev)
        if self_us <= 0:
            continue
        top_cpu.append({
            "name": str(getattr(ev, "key", "unknown"))[:48],
            "calls": int(getattr(ev, "count", 0)),
            "self_cpu_ms": round(self_us / 1000.0, 3),
            "total_cpu_ms": round(_cpu_total_us(ev) / 1000.0, 3),
            "pct": round((self_us / total_self_cpu_us) * 100.0, 1) if total_self_cpu_us > 0 else 0.0,
        })

    return {
        "sort_by": sort_attr,
        "total_self_cuda_ms": round(total_self_cuda_us / 1000.0, 3),
        "total_self_cpu_ms": round(total_self_cpu_us / 1000.0, 3),
        "top_cuda_ops": top_cuda,
        "top_cpu_ops": top_cpu,
    }


def _aggregate_phase_breakdown(steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize step_0 cold warmup vs steady-state (steps[1:]) per-phase timings."""
    if not steps:
        return {}
    steady = steps[1:] if len(steps) > 1 else steps
    phase_keys = ("data_other_ms", "forward_ms", "loss_clip_ms", "backward_ms", "optim_ms", "total_ms")
    steady_means = {
        k: round(statistics.mean(float(s[k]) for s in steady), 3) for k in phase_keys
    }
    steady_medians = {
        k: round(statistics.median(float(s[k]) for s in steady), 3) for k in phase_keys
    }
    tot = steady_means["total_ms"]
    steady_shares_pct = {
        k.removesuffix("_ms"): round((steady_means[k] / tot) * 100.0, 1) if tot > 0 else 0.0
        for k in ("data_other_ms", "forward_ms", "loss_clip_ms", "backward_ms", "optim_ms")
    }
    return {
        "num_steps": len(steps),
        "step_0_cold": {k: steps[0][k] for k in phase_keys},
        "steady_mean_ms": steady_means,
        "steady_median_ms": steady_medians,
        "steady_shares_pct": steady_shares_pct,
        "per_step_trace": steps[:8],
    }


class TelemetryCollector:
    """Collects per-step timing, scalar losses, peak memory, fallback events, and optional debug profile."""

    def __init__(
        self,
        out_path: Path,
        manager: "HookManager",
        *,
        debug_profile: bool = False,
        torch_profile: bool = False,
        profile_start_step: int = 2,
        profile_num_steps: int = 3,
    ) -> None:
        self.out_path = out_path
        self.manager = manager
        self.debug_profile = debug_profile
        self.torch_profile = torch_profile
        self.profile_start_step = max(0, profile_start_step)
        self.profile_num_steps = max(1, profile_num_steps)
        self.t_start = time.perf_counter()
        self.step_times_ms: list[float] = []
        self.train_step_times_ms: list[float] = []
        self.eval_step_times_ms: list[float] = []
        self.losses: list[float] = []
        self._last_step_end: float | None = None
        self._optimizer_used: bool = False
        self._tls = threading.local()
        self._installed: bool = False
        self._flushed: bool = False
        self._torch: Any = None
        self._opt_pre_hook_handle: Any = None
        self._opt_hook_handle: Any = None
        self._orig_backward: Any = None
        self._orig_mod_call: Any = None
        self._mod_call_attr: str | None = None
        # Debug profiling state (unused unless debug_profile / torch_profile is enabled)
        self._step_idx: int = 0
        self._cur_step: dict[str, Any] = _new_step_bucket()
        self.debug_steps: list[dict[str, Any]] = []
        self._module_stats: dict[str, dict[str, Any]] = {}
        self._profiler: Any = None
        self._profiler_active: bool = False
        self._profiler_summary: dict[str, Any] | None = None

    @property
    def _module_depth(self) -> int:
        return int(getattr(self._tls, "module_depth", 0))

    @_module_depth.setter
    def _module_depth(self, val: int) -> None:
        self._tls.module_depth = max(0, int(val))

    @property
    def _suppress_depth(self) -> int:
        return int(getattr(self._tls, "suppress_depth", 0))

    @_suppress_depth.setter
    def _suppress_depth(self, val: int) -> None:
        self._tls.suppress_depth = max(0, int(val))

    @contextmanager
    def suppress(self) -> Iterator[None]:
        """Temporarily suppress telemetry step counting (e.g. during internal prewarm/calibration)."""
        self._suppress_depth += 1
        try:
            yield
        finally:
            self._suppress_depth -= 1

    def _start_profiler_if_ready(self) -> None:
        if not self.torch_profile or self._profiler_active or self._profiler_summary is not None:
            return
        if self._step_idx != self.profile_start_step or self._torch is None:
            return
        try:
            prof_mod = importlib.import_module("torch.profiler")
            activities = [prof_mod.ProfilerActivity.CPU]
            if self._torch.cuda.is_available():
                activities.append(prof_mod.ProfilerActivity.CUDA)
            record_shapes = os.environ.get("PTOPT_TORCH_PROFILE_SHAPES") == "1"
            self._profiler = prof_mod.profile(activities=activities, record_shapes=record_shapes)
            self._profiler.__enter__()
            self._profiler_active = True
        except Exception:  # noqa: BLE001
            self._profiler = None
            self._profiler_active = False

    def _stop_profiler_if_done(self, force: bool = False) -> None:
        if not self._profiler_active or self._profiler is None:
            return
        end_step = self.profile_start_step + self.profile_num_steps
        if not force and self._step_idx < end_step:
            return
        try:
            self._profiler.__exit__(None, None, None)
            has_cuda = bool(self._torch is not None and self._torch.cuda.is_available())
            self._profiler_summary = _summarize_profiler_ops(self._profiler, has_cuda=has_cuda)
            self._profiler_summary["window_steps"] = [self.profile_start_step, self._step_idx]
        except Exception:  # noqa: BLE001
            self._profiler_summary = None
        finally:
            self._profiler = None
            self._profiler_active = False

    def install(self, torch_mod: Any) -> None:
        """Install non-invasive telemetry hooks once on Tensor.backward, Optimizer step, and Module eval."""
        if self._installed or torch_mod is None:
            return
        self._installed = True
        self._torch = torch_mod
        collector = self

        # 1. Capture scalar loss on backward() (+ optional backward phase timing)
        self._orig_backward = torch_mod.Tensor.backward
        orig_backward = self._orig_backward

        @functools.wraps(orig_backward)
        def _wrapped_backward(tensor_self: Any, *args: Any, **kwargs: Any) -> Any:
            if collector._suppress_depth > 0:
                return orig_backward(tensor_self, *args, **kwargs)
            try:
                if tensor_self.numel() == 1:
                    collector.losses.append(float(tensor_self.detach().cpu().item()))
            except Exception:  # noqa: BLE001
                pass
            if not collector.debug_profile:
                return orig_backward(tensor_self, *args, **kwargs)

            _sync_if_cuda(torch_mod)
            t_bwd_start = time.perf_counter()
            cur = collector._cur_step
            if cur["last_phase_end"] is not None:
                cur["loss_clip_ms"] += (t_bwd_start - cur["last_phase_end"]) * 1000.0
            try:
                return orig_backward(tensor_self, *args, **kwargs)
            finally:
                _sync_if_cuda(torch_mod)
                t_bwd_end = time.perf_counter()
                cur["backward_ms"] += (t_bwd_end - t_bwd_start) * 1000.0
                cur["last_phase_end"] = t_bwd_end

        torch_mod.Tensor.backward = _wrapped_backward

        # 2. Capture training step duration via PyTorch global optimizer step pre/post hooks
        try:
            opt_mod = importlib.import_module("torch.optim.optimizer")
            reg_pre_hook = getattr(opt_mod, "register_optimizer_step_pre_hook", None)
            reg_post_hook = getattr(opt_mod, "register_optimizer_step_post_hook", None)
            if callable(reg_pre_hook) and (collector.debug_profile or collector.torch_profile):
                def _opt_pre_hook(_opt: Any, _args: Any, _kwargs: Any) -> None:
                    if collector._suppress_depth > 0 or not collector.debug_profile:
                        return
                    _sync_if_cuda(torch_mod)
                    t_opt_start = time.perf_counter()
                    cur = collector._cur_step
                    if cur["last_phase_end"] is not None:
                        cur["loss_clip_ms"] += (t_opt_start - cur["last_phase_end"]) * 1000.0
                    cur["t_opt_start"] = t_opt_start

                self._opt_pre_hook_handle = reg_pre_hook(_opt_pre_hook)

            if callable(reg_post_hook):
                def _opt_post_hook(opt: Any, _args: Any, _kwargs: Any) -> None:
                    if collector._suppress_depth > 0:
                        return
                    if getattr(opt, "_ptopt_in_scaler_step", False) and not (collector.debug_profile or collector.torch_profile):
                        return
                    if not collector._optimizer_used:
                        collector._optimizer_used = True
                        # Keep step_times_ms focused on training steps once optimizer activity begins.
                        collector.step_times_ms.clear()
                    if collector._last_step_end is None:
                        collector._last_step_end = collector.t_start
                    _sync_if_cuda(torch_mod)
                    now = time.perf_counter()
                    total_ms = (now - collector._last_step_end) * 1000.0
                    rounded_ms = round(total_ms, 4)
                    collector.train_step_times_ms.append(rounded_ms)
                    collector.step_times_ms.append(rounded_ms)
                    collector._last_step_end = now

                    if collector.debug_profile:
                        cur = collector._cur_step
                        optim_ms = (now - cur["t_opt_start"]) * 1000.0 if cur["t_opt_start"] is not None else 0.0
                        cur["optim_ms"] += optim_ms
                        collector.debug_steps.append({
                            "step": collector._step_idx,
                            "data_other_ms": round(cur["data_other_ms"], 3),
                            "forward_ms": round(cur["forward_ms"], 3),
                            "loss_clip_ms": round(cur["loss_clip_ms"], 3),
                            "backward_ms": round(cur["backward_ms"], 3),
                            "optim_ms": round(cur["optim_ms"], 3),
                            "total_ms": round(total_ms, 3),
                        })
                        collector._cur_step = _new_step_bucket()

                    if collector.torch_profile or collector.debug_profile:
                        collector._step_idx += 1
                        was_prof_active = collector._profiler_active
                        collector._stop_profiler_if_done(force=False)
                        if was_prof_active and not collector._profiler_active:
                            collector._last_step_end = time.perf_counter()

                self._opt_hook_handle = reg_post_hook(_opt_post_hook)
        except Exception:  # noqa: BLE001
            self._opt_pre_hook_handle = None
            self._opt_hook_handle = None

        # 3. Capture top-level forward duration & inference step duration
        nn_mod = torch_mod.nn.Module
        attr_name = "__call__"
        self._mod_call_attr = attr_name
        self._orig_mod_call = getattr(nn_mod, attr_name)
        orig_mod_call = self._orig_mod_call

        @functools.wraps(orig_mod_call)
        def _wrapped_mod_call(mod_self: Any, *args: Any, **kwargs: Any) -> Any:
            if collector._suppress_depth > 0:
                return orig_mod_call(mod_self, *args, **kwargs)
            collector._module_depth += 1
            is_top = collector._module_depth == 1
            is_eval_call = not getattr(mod_self, "training", True)
            is_interleaved_eval = bool(
                is_top and collector._optimizer_used and is_eval_call and not collector._cur_step["in_step"]
            )
            t0 = 0.0
            if is_top:
                if collector.debug_profile:
                    _record_module_trace(collector._module_stats, mod_self, args, kwargs, torch_mod, is_eval_call)
                if is_interleaved_eval:
                    _sync_if_cuda(torch_mod)
                    t0 = time.perf_counter()
                elif collector.debug_profile:
                    _sync_if_cuda(torch_mod)
                    t_pre_prof = time.perf_counter()
                    cur = collector._cur_step
                    if not cur["in_step"]:
                        prev_end = collector._last_step_end if collector._last_step_end is not None else collector.t_start
                        cur["data_other_ms"] += (t_pre_prof - prev_end) * 1000.0
                        cur["in_step"] = True
                    if collector.torch_profile:
                        t_ps = time.perf_counter()
                        collector._start_profiler_if_ready()
                        prof_init_sec = time.perf_counter() - t_ps
                        if collector._last_step_end is not None:
                            collector._last_step_end += prof_init_sec
                        else:
                            collector.t_start += prof_init_sec
                    t0 = time.perf_counter()
                else:
                    if collector.torch_profile:
                        collector._start_profiler_if_ready()
                    if not collector._optimizer_used and is_eval_call:
                        t0 = time.perf_counter()
            try:
                out = orig_mod_call(mod_self, *args, **kwargs)
                if is_interleaved_eval:
                    _sync_if_cuda(torch_mod)
                    t1 = time.perf_counter()
                    collector.eval_step_times_ms.append(round((t1 - t0) * 1000.0, 4))
                    # Advance _last_step_end only for grad-free validation so eval loops don't inflate next train step.
                    if not _output_requires_grad(out, torch_mod):
                        collector._last_step_end = t1
                elif is_top and collector.debug_profile:
                    _sync_if_cuda(torch_mod)
                    t1 = time.perf_counter()
                    dt_ms = (t1 - t0) * 1000.0
                    cur = collector._cur_step
                    cur["forward_ms"] += dt_ms
                    cur["last_phase_end"] = t1
                    if not collector._optimizer_used and is_eval_call:
                        rounded_ms = round(dt_ms, 4)
                        collector.eval_step_times_ms.append(rounded_ms)
                        collector.step_times_ms.append(rounded_ms)
                        total_ms = cur["data_other_ms"] + dt_ms
                        collector.debug_steps.append({
                            "step": collector._step_idx,
                            "data_other_ms": round(cur["data_other_ms"], 3),
                            "forward_ms": round(dt_ms, 3),
                            "loss_clip_ms": 0.0,
                            "backward_ms": 0.0,
                            "optim_ms": 0.0,
                            "total_ms": round(total_ms, 3),
                        })
                        collector._cur_step = _new_step_bucket()
                        collector._last_step_end = t1
                        collector._step_idx += 1
                        was_prof_active = collector._profiler_active
                        collector._stop_profiler_if_done(force=False)
                        if was_prof_active and not collector._profiler_active:
                            collector._last_step_end = time.perf_counter()
                elif is_top and not collector._optimizer_used and is_eval_call:
                    _sync_if_cuda(torch_mod)
                    dt_ms = (time.perf_counter() - t0) * 1000.0
                    rounded_ms = round(dt_ms, 4)
                    collector.eval_step_times_ms.append(rounded_ms)
                    collector.step_times_ms.append(rounded_ms)
                    if collector.torch_profile:
                        collector._step_idx += 1
                        collector._stop_profiler_if_done(force=False)
                return out
            finally:
                collector._module_depth -= 1

        setattr(nn_mod, attr_name, _wrapped_mod_call)
        atexit.register(self.flush)

    def uninstall(self) -> None:
        """Restore monkey-patched hooks (for clean unit tests)."""
        if not self._installed or self._torch is None:
            return
        self._stop_profiler_if_done(force=True)
        if self._orig_backward is not None:
            self._torch.Tensor.backward = self._orig_backward
        if self._opt_pre_hook_handle is not None:
            self._opt_pre_hook_handle.remove()
            self._opt_pre_hook_handle = None
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

        telemetry_only = os.environ.get("PTOPT_TELEMETRY_ONLY") == "1"
        payload: dict[str, Any] = {
            "telemetry_schema_version": 1,
            "mode": (
                "baseline"
                if telemetry_only
                else (self.manager.config.profile if self.manager.config else "unknown")
            ),
            "telemetry_only": telemetry_only,
            "end_to_end_sec": round(time.perf_counter() - self.t_start, 4),
            "step_times_ms": list(self.step_times_ms),
            "train_step_times_ms": list(self.train_step_times_ms),
            "eval_step_times_ms": list(self.eval_step_times_ms),
            "losses": [round(x, 6) for x in self.losses],
            "memory_mb": {
                "peak_allocated": peak_alloc_mb,
                "peak_reserved": peak_resv_mb,
            },
            "fallback_events": [ev.to_dict() for ev in self.manager.events],
        }
        if self.debug_profile or self.torch_profile:
            debug_block: dict[str, Any] = {}
            if self.debug_steps:
                debug_block["phase_breakdown"] = _aggregate_phase_breakdown(self.debug_steps)
            if self._module_stats:
                debug_block["module_summary"] = dict(self._module_stats)
            mem_debug = _extract_cuda_memory_debug(torch_mod) if torch_mod is not None else None
            if mem_debug is not None:
                debug_block["cuda_memory_debug"] = mem_debug
            if self._profiler_summary is not None:
                debug_block["torch_profiler"] = self._profiler_summary
            dynamo_debug = _extract_dynamo_debug()
            if dynamo_debug is not None:
                debug_block["dynamo_compile"] = dynamo_debug
            payload["debug_profile"] = debug_block
        return payload

    def flush(self) -> None:
        """Write collected telemetry JSON once to `self.out_path` (worker & multi-rank safe)."""
        if self._flushed:
            return
        if _is_dataloader_worker(self._torch):
            self._flushed = True
            return
        self._flushed = True
        self._stop_profiler_if_done(force=True)
        target_path = _resolve_rank_output_path(self.out_path)
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(json.dumps(self.snapshot(), indent=2) + "\n", encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass


def _is_dataloader_worker(torch_mod: Any) -> bool:
    """Return True inside a DataLoader worker subprocess so it never overwrites main telemetry."""
    if torch_mod is None:
        return False
    try:
        get_worker = getattr(getattr(getattr(torch_mod, "utils", None), "data", None), "get_worker_info", None)
        return bool(callable(get_worker) and get_worker() is not None)
    except Exception:  # noqa: BLE001
        return False


def _resolve_rank_output_path(out_path: Path) -> Path:
    """Suffix non-zero distributed ranks (`RANK != 0`) so DDP workers don't clobber rank 0."""
    rank = os.environ.get("RANK")
    if rank is not None and rank not in ("", "0"):
        return out_path.with_name(f"{out_path.stem}.rank{rank}{out_path.suffix}")
    return out_path


def _record_module_trace(
    trace_map: dict[str, dict[str, Any]],
    mod_self: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    torch_mod: Any,
    is_eval: bool,
) -> None:
    """Record compact top-level module call counts, parameter count, and distinct input shapes."""
    cls_name = type(mod_self).__qualname__
    entry = trace_map.get(cls_name)
    if entry is None:
        if len(trace_map) >= 16:
            return
        params = 0
        if hasattr(mod_self, "parameters"):
            try:
                params = sum(int(p.numel()) for p in mod_self.parameters())
            except Exception:  # noqa: BLE001
                params = 0
        entry = {"train_calls": 0, "eval_calls": 0, "params": params, "input_shapes": []}
        trace_map[cls_name] = entry

    entry["eval_calls" if is_eval else "train_calls"] += 1
    if len(entry["input_shapes"]) < 8 and torch_mod is not None:
        try:
            from ptopt.hooks.module_dispatch import build_shape_signature  # noqa: PLC0415

            sig_str = str(build_shape_signature(args, kwargs, torch_mod, include_device=True))[:180]
            if sig_str not in entry["input_shapes"]:
                entry["input_shapes"].append(sig_str)
        except Exception:  # noqa: BLE001
            pass


def _extract_dynamo_debug() -> dict[str, Any] | None:
    """Extract TorchDynamo/Inductor compile counters and compile timings if Dynamo was loaded."""
    du = sys.modules.get("torch._dynamo.utils")
    if du is None:
        return None
    raw_counters = getattr(du, "counters", None)
    if not raw_counters:
        return None

    counters: dict[str, dict[str, int]] = {}
    for sec_name, sec_map in raw_counters.items():
        if not sec_map:
            continue
        sorted_items = sorted(sec_map.items(), key=lambda kv: int(kv[1]), reverse=True)[:20]
        counters[str(sec_name)] = {str(k)[:240]: int(v) for k, v in sorted_items}

    if not counters:
        return None

    frames = counters.get("frames", {})
    stats = counters.get("stats", {})
    aot = counters.get("aot_autograd", {})
    ind = counters.get("inductor", {})
    gb_count = sum(counters.get("graph_break", {}).values()) + sum(counters.get("unimplemented", {}).values())
    recomp_count = sum(counters.get("recompiles", {}).values())

    summary = {
        "frames_total": int(frames.get("total", 0)),
        "frames_ok": int(frames.get("ok", 0)),
        "calls_captured": int(stats.get("calls_captured", 0)),
        "unique_graphs": int(stats.get("unique_graphs", 0)),
        "graph_break_count": int(gb_count),
        "recompiles_count": int(recomp_count),
        "inductor_cache_hit": int(ind.get("fxgraph_cache_hit", 0)),
        "inductor_cache_miss": int(ind.get("fxgraph_cache_miss", 0)),
        "aot_cache_hit": int(aot.get("autograd_cache_hit", 0)),
        "aot_cache_miss": int(aot.get("autograd_cache_miss", 0)),
    }

    compile_times_str = ""
    compile_times_fn = getattr(du, "compile_times", None)
    if callable(compile_times_fn):
        try:
            compile_times_str = str(compile_times_fn(repr="str"))[:8000]
        except Exception:  # noqa: BLE001
            compile_times_str = ""

    return {
        "summary": summary,
        "counters": counters,
        "compile_times": compile_times_str,
    }


def _output_requires_grad(out: Any, torch_mod: Any) -> bool:
    """Check whether a forward output (tensor or nested container) carries `requires_grad=True`."""
    tensor_cls = getattr(torch_mod, "Tensor", None) if torch_mod is not None else None
    if tensor_cls is None or out is None:
        return False
    if isinstance(out, tensor_cls):
        return bool(getattr(out, "requires_grad", False))
    try:
        from ptopt.hooks.module_dispatch import iter_nested_tensors  # noqa: PLC0415

        return any(bool(getattr(t, "requires_grad", False)) for t in iter_nested_tensors(out, tensor_cls, max_items=16))
    except Exception:  # noqa: BLE001
        return False


def _sync_if_cuda(torch_mod: Any) -> None:
    try:
        if torch_mod.cuda.is_available():
            torch_mod.cuda.synchronize()
    except Exception:  # noqa: BLE001
        pass


def maybe_install_telemetry_hook(
    torch_mod: Any,
    manager: "HookManager",
    env: dict[str, str] | None = None,
) -> TelemetryCollector | None:
    """Install TelemetryCollector if `PTOPT_TELEMETRY_OUT` is configured."""
    env_map = os.environ if env is None else env
    out_raw = env_map.get("PTOPT_TELEMETRY_OUT")
    if not out_raw:
        return None
    debug_profile = env_map.get("PTOPT_DEBUG_PROFILE") == "1"
    torch_profile = env_map.get("PTOPT_TORCH_PROFILE") == "1"
    try:
        start_step = int(env_map.get("PTOPT_TORCH_PROFILE_START", "2"))
    except ValueError:
        start_step = 2
    try:
        num_steps = int(env_map.get("PTOPT_TORCH_PROFILE_STEPS", "3"))
    except ValueError:
        num_steps = 3

    collector = TelemetryCollector(
        Path(out_raw),
        manager,
        debug_profile=debug_profile,
        torch_profile=torch_profile,
        profile_start_step=start_step,
        profile_num_steps=num_steps,
    )
    collector.install(torch_mod)
    return collector
