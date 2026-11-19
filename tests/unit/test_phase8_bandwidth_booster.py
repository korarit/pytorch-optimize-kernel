"""Unit tests for Phase 8: VRAM-Headroom Bandwidth Booster & Detailed Telemetry."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ptopt.config import ConfigResolver, OptimizeConfig
from ptopt.hooks import HookManager
from ptopt.hooks.telemetry_hook import TelemetryCollector
from ptopt.passes.auto_compile import configure_inductor_bandwidth_tuning
from ptopt.passes.dataloader_tune import (
    _AsyncCudaPrefetchIter,
    compute_adaptive_prefetch_depth,
    install_dataloader_hook,
    uninstall_dataloader_hook,
)
from ptopt.passes.inference_opt import decide_fp8_weight_retention
from ptopt.passes.memory_allocator import probe_cuda_vram_headroom
from tests.unit.test_config_resolver import _make_fp


class _MockCudaTensor:
    def __init__(self, val: int, device_type: str = "cpu") -> None:
        self.val = val
        self.device = SimpleNamespace(type=device_type)
        self.is_cuda = device_type == "cuda"
        self.recorded_streams: list[object] = []

    def to(self, device: object = None, non_blocking: bool = False) -> "_MockCudaTensor":
        t = _MockCudaTensor(self.val, device_type="cuda")
        t.non_blocking = non_blocking
        t.target_device = device
        return t

    def record_stream(self, stream: object) -> None:
        self.recorded_streams.append(stream)


class _MockEvent:
    def __init__(self) -> None:
        self.recorded_streams: list["_MockStream"] = []

    def record(self, stream: "_MockStream") -> None:
        self.recorded_streams.append(stream)


class _MockStream:
    def __init__(self, name: str) -> None:
        self.name = name
        self.waited_on: list[_MockStream] = []
        self.waited_events: list[_MockEvent] = []

    def wait_stream(self, other: "_MockStream") -> None:
        self.waited_on.append(other)

    def wait_event(self, ev: _MockEvent) -> None:
        self.waited_events.append(ev)


class _MockCudaContext:
    def __init__(self, stream_obj: _MockStream) -> None:
        self.stream_obj = stream_obj

    def __enter__(self) -> "_MockStream":
        return self.stream_obj

    def __exit__(self, *_exc: object) -> None:
        return None


def _make_vram_torch(*, total_gb: float, used_ratio: float, initialized: bool = True) -> SimpleNamespace:
    total_bytes = int(total_gb * (1024.0 ** 3))
    state = {"used_bytes": int(total_bytes * used_ratio), "initialized": initialized, "props_calls": 0}
    main_stream = _MockStream("main")
    created_streams: list[_MockStream] = []

    def _mk_stream(device: object = None) -> _MockStream:
        s = _MockStream(f"copy_{len(created_streams)}")
        created_streams.append(s)
        return s

    def _get_props(_idx: int) -> SimpleNamespace:
        state["props_calls"] += 1
        return SimpleNamespace(total_memory=total_bytes, major=8, minor=9)

    cuda_ns = SimpleNamespace(
        is_available=lambda: True,
        is_initialized=lambda: state["initialized"],
        current_device=lambda: 0,
        get_device_properties=_get_props,
        memory_allocated=lambda _idx=0: state["used_bytes"],
        memory_reserved=lambda _idx=0: state["used_bytes"],
        Stream=_mk_stream,
        Event=lambda: _MockEvent(),
        stream=lambda s: _MockCudaContext(s),
        current_stream=lambda _dev=None: main_stream,
        _main_stream=main_stream,
        _state=state,
    )
    return SimpleNamespace(
        Tensor=_MockCudaTensor,
        device=lambda kind, idx=0: f"{kind}:{idx}",
        cuda=cuda_ns,
    )


class TestPhase8AdaptivePrefetch(unittest.TestCase):
    """8.1: Verify adaptive multi-batch GPU prefetch queue and dynamic VRAM back-off."""

    def test_adaptive_depth_scales_with_headroom_and_backs_off(self) -> None:
        t_low = _make_vram_torch(total_gb=24.0, used_ratio=0.20)
        d_low, hr_low = compute_adaptive_prefetch_depth(t_low)
        self.assertEqual(d_low, 3)
        self.assertIsNotNone(hr_low)

        t_mid = _make_vram_torch(total_gb=24.0, used_ratio=0.55)
        d_mid, _ = compute_adaptive_prefetch_depth(t_mid)
        self.assertEqual(d_mid, 2)

        t_tight = _make_vram_torch(total_gb=24.0, used_ratio=0.72)
        d_tight, _ = compute_adaptive_prefetch_depth(t_tight)
        self.assertEqual(d_tight, 1)

        t_full = _make_vram_torch(total_gb=24.0, used_ratio=0.82)
        d_full, _ = compute_adaptive_prefetch_depth(t_full)
        self.assertEqual(d_full, 0)

    def test_prefetch_queue_stages_multiple_batches_and_backs_off_on_pressure(self) -> None:
        mock_t = _make_vram_torch(total_gb=24.0, used_ratio=0.25)
        mgr = HookManager()
        raw_batches = [(_MockCudaTensor(i),) for i in range(6)]
        it = _AsyncCudaPrefetchIter(iter(raw_batches), mock_t, manager=mgr)

        # At 25% VRAM used, queue should pre-stage 3 batches right at __init__!
        self.assertEqual(mock_t._ptopt_prefetch_stats["initial_depth"], 3)
        self.assertEqual(mock_t._ptopt_prefetch_stats["max_queue_depth"], 3)
        self.assertEqual(mock_t._ptopt_prefetch_stats["batches_prefetched"], 3)

        b0 = next(it)
        self.assertTrue(b0[0].is_cuda)
        self.assertEqual(b0[0].val, 0)
        # Per-batch cuda.Event synchronization avoids stalling on later queued copy batches
        self.assertGreaterEqual(len(mock_t.cuda._main_stream.waited_events), 1)
        self.assertEqual(mock_t.cuda._main_stream.waited_on, [])
        # get_device_properties(idx) must be cached per device instead of queried per batch
        self.assertEqual(mock_t.cuda._state["props_calls"], 1)

        # Simulate VRAM spike to 85% -> must back off queue depth to 0 and log FALLBACK event!
        mock_t.cuda._state["used_bytes"] = int(24.0 * (1024.0 ** 3) * 0.85)
        b1 = next(it)
        self.assertEqual(b1[0].val, 1)
        self.assertEqual(mock_t._ptopt_prefetch_stats["current_depth"], 0)
        self.assertEqual(mock_t._ptopt_prefetch_stats["backoff_events"], 1)
        self.assertTrue(
            any(ev.pass_name == "dataloader.cuda_prefetch" and ev.status == "FALLBACK" for ev in mgr.events)
        )

        # Drain remaining batches safely in on-demand mode without dropping items
        rest = [b[0].val for b in it]
        self.assertEqual(rest, [2, 3, 4, 5])
        self.assertEqual(mock_t._ptopt_prefetch_stats["batches_served"], 6)

    def test_auto_prefetch_skips_uninitialized_cuda_for_cpu_scripts(self) -> None:
        cfg = OptimizeConfig(profile="balanced")
        cfg.dataloader.auto_tune = True
        cfg.dataloader.cuda_prefetch = True
        mgr = HookManager()
        ds = torch.utils.data.TensorDataset(torch.randn(6, 4))

        with (
            patch.dict(os.environ, {"PTOPT_ACTIVE": "1"}),
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "is_initialized", return_value=False, create=True),
        ):
            install_dataloader_hook(cfg, mgr, torch)
            try:
                dl = torch.utils.data.DataLoader(ds, batch_size=3, num_workers=0)
                batches = list(dl)
                self.assertEqual(len(batches), 2)
                self.assertFalse(batches[0][0].is_cuda)
                self.assertTrue(
                    any(ev.pass_name == "dataloader.cuda_prefetch" and ev.status == "SKIPPED" for ev in mgr.events)
                )
            finally:
                uninstall_dataloader_hook(torch)
                mgr.reset()


    def test_dataloader_hook_default_depth_1_vs_boost_depth_3(self) -> None:
        class _FakeTensor(_MockCudaTensor):
            pass

        class _FakeDataLoader:
            def __init__(
                self,
                dataset: Any,
                batch_size: int = 1,
                num_workers: int = 0,
                pin_memory: bool = False,
                persistent_workers: bool = False,
                prefetch_factor: int | None = None,
            ) -> None:
                self.dataset = dataset
                self.batch_size = batch_size
                self.num_workers = num_workers
                self.pin_memory = pin_memory
                self.persistent_workers = persistent_workers
                self.prefetch_factor = prefetch_factor

            def __iter__(self):
                return iter([(_FakeTensor(i),) for i in range(4)])

        mock_vram = _make_vram_torch(total_gb=24.0, used_ratio=0.25)
        fake_torch = SimpleNamespace(
            utils=SimpleNamespace(data=SimpleNamespace(DataLoader=_FakeDataLoader)),
            Tensor=_FakeTensor,
            device=mock_vram.device,
            cuda=mock_vram.cuda,
        )

        # Default (bandwidth_boost=False) caps lookahead at depth=1 even on plenty of free VRAM
        cfg_default = OptimizeConfig(profile="balanced")
        cfg_default.dataloader.auto_tune = True
        cfg_default.dataloader.cuda_prefetch = True
        mgr_def = HookManager()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PTOPT_CUDA_PREFETCH_DEPTH", None)
            os.environ.pop("PTOPT_CUDA_PREFETCH_MAX_DEPTH", None)
            install_dataloader_hook(cfg_default, mgr_def, fake_torch)
            try:
                dl_def = _FakeDataLoader([0, 1, 2, 3])
                out_def = [b[0].val for b in dl_def]
                self.assertEqual(out_def, [0, 1, 2, 3])
                self.assertEqual(fake_torch._ptopt_prefetch_stats["initial_depth"], 1)
                self.assertEqual(fake_torch._ptopt_prefetch_stats["max_queue_depth"], 1)
            finally:
                uninstall_dataloader_hook(fake_torch)
                mgr_def.reset()

        # Opt-in bandwidth_boost=True unlocks adaptive depth=3 on free VRAM
        mock_vram2 = _make_vram_torch(total_gb=24.0, used_ratio=0.25)
        fake_torch2 = SimpleNamespace(
            utils=SimpleNamespace(data=SimpleNamespace(DataLoader=_FakeDataLoader)),
            Tensor=_FakeTensor,
            device=mock_vram2.device,
            cuda=mock_vram2.cuda,
        )
        cfg_boost = OptimizeConfig(profile="balanced")
        cfg_boost.runtime.bandwidth_boost = True
        cfg_boost.dataloader.auto_tune = True
        cfg_boost.dataloader.cuda_prefetch = True
        mgr_boost = HookManager()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PTOPT_CUDA_PREFETCH_DEPTH", None)
            os.environ.pop("PTOPT_CUDA_PREFETCH_MAX_DEPTH", None)
            install_dataloader_hook(cfg_boost, mgr_boost, fake_torch2)
            try:
                dl_boost = _FakeDataLoader([0, 1, 2, 3])
                out_boost = [b[0].val for b in dl_boost]
                self.assertEqual(out_boost, [0, 1, 2, 3])
                self.assertEqual(fake_torch2._ptopt_prefetch_stats["initial_depth"], 3)
                self.assertEqual(fake_torch2._ptopt_prefetch_stats["max_queue_depth"], 3)
            finally:
                uninstall_dataloader_hook(fake_torch2)
                mgr_boost.reset()


class TestPhase8ConfigRulesAndInductorFP8(unittest.TestCase):
    """8.2, 8.3, 8.4, 8.5: Verify opt-in bandwidth_boost rules, Inductor BW tuning, FP8 cache policy, and telemetry."""

    def test_vram_bandwidth_boost_opt_in_and_cli_flag(self) -> None:
        from ptopt.cli.main import _build_parser, _collect_overrides

        resolver = ConfigResolver()
        fp_l4 = _make_fp(gpu_name="NVIDIA L4", major=8, minor=9, vram_mb=22563.1, physical_cores=8, logical_cores=16)

        # Default balanced must KEEP Phase 8 disabled (vram_bandwidth_boost not matched)
        res_default = resolver.resolve(fp_l4, profile="balanced", env={})
        self.assertFalse(res_default.config.runtime.bandwidth_boost)
        self.assertNotIn("vram_bandwidth_boost", res_default.matched_rules)
        self.assertFalse(res_default.config.runtime.cudnn_benchmark)
        self.assertFalse(res_default.config.dataloader.cuda_prefetch)
        self.assertEqual(res_default.config.dataloader.prefetch_factor, 2)

        # Opt-in via PTOPT_BANDWIDTH_BOOST=1 activates vram_bandwidth_boost on >=12GB CUDA
        res_env = resolver.resolve(fp_l4, profile="balanced", env={"PTOPT_BANDWIDTH_BOOST": "1"})
        self.assertTrue(res_env.config.runtime.bandwidth_boost)
        self.assertIn("vram_bandwidth_boost", res_env.matched_rules)
        self.assertTrue(res_env.config.runtime.cudnn_benchmark)
        self.assertTrue(res_env.config.dataloader.cuda_prefetch)
        self.assertEqual(res_env.config.dataloader.prefetch_factor, 4)
        self.assertEqual(res_env.config.dataloader.num_workers, 4)

        # CLI --bandwidth-boost translates cleanly to runtime.bandwidth_boost=True override
        parsed, _cmd = _build_parser().parse_known_args(["--profile", "balanced", "--bandwidth-boost", "train.py"])
        cli_overrides = _collect_overrides(parsed)
        self.assertIs(cli_overrides.get("runtime.bandwidth_boost"), True)
        res_cli = resolver.resolve(fp_l4, profile="balanced", overrides=cli_overrides, env={})
        self.assertTrue(res_cli.config.runtime.bandwidth_boost)
        self.assertIn("vram_bandwidth_boost", res_cli.matched_rules)

        # High CPU core + RAM scales workers up to 8 under boost
        fp_big_cpu = _make_fp(gpu_name="NVIDIA A100", major=8, minor=0, vram_mb=40960.0, physical_cores=16, logical_cores=32)
        res_big = resolver.resolve(fp_big_cpu, profile="balanced", env={"PTOPT_BANDWIDTH_BOOST": "1"})
        self.assertEqual(res_big.config.dataloader.num_workers, 8)

        # Explicit user override (--no-dataloader-tune / prefetch_factor) is preserved under --bandwidth-boost
        res_explicit = resolver.resolve(
            fp_big_cpu,
            profile="balanced",
            overrides={"runtime.bandwidth_boost": True, "dataloader.auto_tune": False, "dataloader.num_workers": 0},
            env={},
        )
        self.assertTrue(res_explicit.config.runtime.cudnn_benchmark)
        self.assertFalse(res_explicit.config.dataloader.auto_tune)
        self.assertEqual(res_explicit.config.dataloader.num_workers, 0)

        # Low-shm must keep DataLoader off while still allowing cudnn_benchmark under boost
        fp_low_shm = _make_fp(gpu_name="NVIDIA L4", major=8, minor=9, vram_mb=22563.1, shm_total_mb=512.0)
        res_shm = resolver.resolve(fp_low_shm, profile="balanced", env={"PTOPT_BANDWIDTH_BOOST": "1"})
        self.assertTrue(res_shm.config.runtime.cudnn_benchmark)
        self.assertFalse(res_shm.config.dataloader.cuda_prefetch)
        self.assertEqual(res_shm.config.dataloader.num_workers, 0)

        # Supersafe must NEVER be altered even with --bandwidth-boost
        res_sup = resolver.resolve(fp_l4, profile="supersafe", overrides={"runtime.bandwidth_boost": True}, env={})
        self.assertNotIn("vram_bandwidth_boost", res_sup.matched_rules)
        self.assertFalse(res_sup.config.runtime.bandwidth_boost)
        self.assertFalse(res_sup.config.runtime.cudnn_benchmark)
        self.assertFalse(res_sup.config.dataloader.cuda_prefetch)

    def test_inductor_bandwidth_tuning_gate(self) -> None:
        mock_t_ok = _make_vram_torch(total_gb=24.0, used_ratio=0.30)

        # Default (bandwidth_boost=False) skips Inductor coordinate_descent_tuning even in aggressive
        cfg_default = OptimizeConfig(profile="aggressive")
        mgr_def = HookManager()
        fake_ind_def = SimpleNamespace(config=SimpleNamespace(coordinate_descent_tuning=False, epilogue_fusion=False))
        with patch.dict(sys.modules, {"torch._inductor": fake_ind_def}), patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PTOPT_INDUCTOR_BW_TUNE", None)
            info_def = configure_inductor_bandwidth_tuning(cfg_default, mock_t_ok, mgr_def, chosen_mode="max-autotune")
        self.assertFalse(info_def["enabled"])
        self.assertFalse(fake_ind_def.config.coordinate_descent_tuning)

        # balanced + bandwidth_boost (default compile mode) avoids cold-compile explosion unless aggressive/max-autotune
        cfg_bal_boost = OptimizeConfig(profile="balanced")
        cfg_bal_boost.runtime.bandwidth_boost = True
        fake_ind_bal = SimpleNamespace(config=SimpleNamespace(coordinate_descent_tuning=False, epilogue_fusion=False))
        with patch.dict(sys.modules, {"torch._inductor": fake_ind_bal}), patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PTOPT_INDUCTOR_BW_TUNE", None)
            info_bal = configure_inductor_bandwidth_tuning(cfg_bal_boost, mock_t_ok, HookManager(), chosen_mode="default")
        self.assertFalse(info_bal["enabled"])

        # aggressive (max-autotune) + bandwidth_boost=True with free VRAM applies tuning
        cfg_boost = OptimizeConfig(profile="aggressive")
        cfg_boost.runtime.bandwidth_boost = True
        mgr_ok = HookManager()
        fake_ind = SimpleNamespace(config=SimpleNamespace(coordinate_descent_tuning=False, epilogue_fusion=False))
        with patch.dict(sys.modules, {"torch._inductor": fake_ind}), patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PTOPT_INDUCTOR_BW_TUNE", None)
            info_ok = configure_inductor_bandwidth_tuning(cfg_boost, mock_t_ok, mgr_ok, chosen_mode="max-autotune")
        self.assertTrue(info_ok["enabled"])
        self.assertTrue(fake_ind.config.coordinate_descent_tuning)
        self.assertTrue(fake_ind.config.epilogue_fusion)

        # Tight VRAM (>=75%) skips coordinate descent tuning even when bandwidth_boost=True
        mgr_tight = HookManager()
        mock_t_tight = _make_vram_torch(total_gb=24.0, used_ratio=0.80)
        info_tight = configure_inductor_bandwidth_tuning(cfg_boost, mock_t_tight, mgr_tight, chosen_mode="max-autotune")
        self.assertFalse(info_tight["enabled"])
        self.assertTrue(any(ev.pass_name == "compile.inductor_bw_tune" and ev.status == "SKIPPED" for ev in mgr_tight.events))

    def test_fp8_weight_retention_and_telemetry_snapshot(self) -> None:
        mgr = HookManager()
        mock_free = _make_vram_torch(total_gb=24.0, used_ratio=0.25)
        mock_full = _make_vram_torch(total_gb=24.0, used_ratio=0.86)

        # Default (bandwidth_boost=False, no env) uses dual_resident_default
        cfg_off = OptimizeConfig(profile="aggressive")
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PTOPT_FP8_DROP_BF16_EVAL", None)
            drop_def, pol_def = decide_fp8_weight_retention(mock_full, cfg=cfg_off, manager=mgr)
        self.assertFalse(drop_def)
        self.assertEqual(pol_def["mode"], "dual_resident_default")

        # Opt-in bandwidth_boost=True (no env) keeps dual-resident under 85% and auto-drops at >=85%
        cfg_on = OptimizeConfig(profile="aggressive")
        cfg_on.runtime.bandwidth_boost = True
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PTOPT_FP8_DROP_BF16_EVAL", None)
            drop_bfree, pol_bfree = decide_fp8_weight_retention(mock_free, cfg=cfg_on, manager=mgr)
            drop_bfull, pol_bfull = decide_fp8_weight_retention(mock_full, cfg=cfg_on, manager=mgr)
        self.assertFalse(drop_bfree)
        self.assertEqual(pol_bfree["mode"], "boost_dual_resident")
        self.assertTrue(drop_bfull)
        self.assertEqual(pol_bfull["mode"], "boost_auto_drop")

        # Explicit PTOPT_FP8_DROP_BF16_EVAL=auto still works directly at 0.80 threshold
        with patch.dict(os.environ, {"PTOPT_FP8_DROP_BF16_EVAL": "auto"}):
            drop_free, pol_free = decide_fp8_weight_retention(mock_free, manager=mgr)
            drop_full, pol_full = decide_fp8_weight_retention(mock_full, manager=mgr)
        self.assertFalse(drop_free)
        self.assertEqual(pol_free["mode"], "auto_dual_resident")
        self.assertTrue(drop_full)
        self.assertEqual(pol_full["mode"], "auto_drop")

        # Verify TelemetryCollector captures bandwidth_vram_booster stats cleanly
        with tempfile.TemporaryDirectory() as tmp:
            tel_path = Path(tmp) / "tel.json"
            mgr_t = HookManager()
            mgr_t.config = OptimizeConfig(profile="balanced")
            col = TelemetryCollector(tel_path, mgr_t, debug_profile=True)
            mock_t = _make_vram_torch(total_gb=24.0, used_ratio=0.25)
            mock_t._ptopt_prefetch_stats = {"initial_depth": 3, "max_queue_depth": 3, "backoff_events": 0}
            mock_t._ptopt_inductor_bw_tune = {"enabled": True, "flags": ["coordinate_descent_tuning=True"]}
            mock_t._ptopt_fp8_cache_policy = {"mode": "dual_resident_default", "drop_bf16": False}
            col._torch = mock_t
            snap = col.snapshot()
            self.assertIn("bandwidth_vram_booster", snap)
            self.assertIn("bandwidth_vram_booster", snap["debug_profile"])
            bw_blk = snap["debug_profile"]["bandwidth_vram_booster"]
            self.assertEqual(bw_blk["prefetch_stats"]["initial_depth"], 3)
            self.assertTrue(bw_blk["inductor_bw_tune"]["enabled"])
            self.assertIsNotNone(probe_cuda_vram_headroom(mock_t))
            mgr_t.reset()


if __name__ == "__main__":
    unittest.main()
