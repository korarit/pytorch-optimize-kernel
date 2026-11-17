"""Unit tests for Phase 3.1: runtime flags, memory allocator, and channels_last passes."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import OptimizeConfig
from src.hooks import HookManager
from src.hooks.checkpoint_compat import install_checkpoint_compat_hooks, uninstall_checkpoint_compat_hooks
from src.passes.channels_last import (
    apply_channels_last_pass,
    convert_module_channels_last,
    install_channels_last_hook,
    uninstall_channels_last_hook,
)
from src.passes.memory_allocator import apply_prelaunch_env
from src.passes.runtime_flags import apply_runtime_flags_pass


def _make_mock_cuda_torch(*, major: int, minor: int, cuda_avail: bool = True) -> tuple[SimpleNamespace, dict[str, object]]:
    state: dict[str, object] = {
        "matmul_tf32": False,
        "cudnn_tf32": False,
        "cudnn_benchmark": False,
        "matmul_precision": None,
        "flash_sdp": None,
        "mem_sdp": None,
        "math_sdp": None,
        "cuda_backends_touched": False,
    }

    class _Matmul:
        @property
        def allow_tf32(self) -> bool:
            return bool(state["matmul_tf32"])

        @allow_tf32.setter
        def allow_tf32(self, val: bool) -> None:
            state["cuda_backends_touched"] = True
            state["matmul_tf32"] = val

    class _Cudnn:
        @property
        def allow_tf32(self) -> bool:
            return bool(state["cudnn_tf32"])

        @allow_tf32.setter
        def allow_tf32(self, val: bool) -> None:
            state["cuda_backends_touched"] = True
            state["cudnn_tf32"] = val

        @property
        def benchmark(self) -> bool:
            return bool(state["cudnn_benchmark"])

        @benchmark.setter
        def benchmark(self, val: bool) -> None:
            state["cuda_backends_touched"] = True
            state["cudnn_benchmark"] = val

    def _set_prec(val: str) -> None:
        state["matmul_precision"] = val

    def _set_flash(val: bool) -> None:
        state["cuda_backends_touched"] = True
        state["flash_sdp"] = val

    def _set_mem(val: bool) -> None:
        state["cuda_backends_touched"] = True
        state["mem_sdp"] = val

    def _set_math(val: bool) -> None:
        state["cuda_backends_touched"] = True
        state["math_sdp"] = val

    cuda_ns = SimpleNamespace(
        is_available=lambda: cuda_avail,
        current_device=lambda: 0,
        get_device_properties=lambda _idx: SimpleNamespace(major=major, minor=minor),
    )
    cuda_backend = SimpleNamespace(
        matmul=_Matmul(),
        enable_flash_sdp=_set_flash,
        enable_mem_efficient_sdp=_set_mem,
        enable_math_sdp=_set_math,
    )
    backends = SimpleNamespace(cuda=cuda_backend, cudnn=_Cudnn())
    tmod = SimpleNamespace(
        cuda=cuda_ns,
        backends=backends,
        set_float32_matmul_precision=_set_prec,
    )
    return tmod, state


class TestRuntimeFlagsPass(unittest.TestCase):
    """Tests for architecture guards in runtime_flags."""

    def test_t4_sm75_does_not_enable_tf32_or_flash_sdp(self) -> None:
        tmod, state = _make_mock_cuda_torch(major=7, minor=5, cuda_avail=True)
        cfg = OptimizeConfig(profile="safe")
        cfg.runtime.tf32 = True
        cfg.runtime.cudnn_benchmark = True
        cfg.attention.flash_sdp = True
        cfg.attention.mem_efficient_sdp = True

        mgr = HookManager()
        msg = apply_runtime_flags_pass(cfg, mgr, torch_mod=tmod)
        self.assertFalse(state["matmul_tf32"])
        self.assertFalse(state["cudnn_tf32"])
        self.assertIsNone(state["matmul_precision"])
        self.assertTrue(state["cudnn_benchmark"])
        self.assertFalse(state["flash_sdp"])
        self.assertTrue(state["mem_sdp"])
        self.assertIn("tf32=skipped", msg)

    def test_a100_sm80_enables_tf32_and_flash_sdp(self) -> None:
        tmod, state = _make_mock_cuda_torch(major=8, minor=0, cuda_avail=True)
        cfg = OptimizeConfig(profile="safe")
        cfg.runtime.tf32 = True
        cfg.runtime.cudnn_benchmark = True
        cfg.attention.flash_sdp = True

        mgr = HookManager()
        msg = apply_runtime_flags_pass(cfg, mgr, torch_mod=tmod)
        self.assertTrue(state["matmul_tf32"])
        self.assertTrue(state["cudnn_tf32"])
        self.assertEqual(state["matmul_precision"], "high")
        self.assertTrue(state["cudnn_benchmark"])
        self.assertTrue(state["flash_sdp"])
        self.assertIn("tf32=on", msg)

    def test_cpu_only_never_touches_cuda_backends(self) -> None:
        tmod, state = _make_mock_cuda_torch(major=8, minor=9, cuda_avail=False)
        cfg = OptimizeConfig(profile="safe")
        cfg.runtime.tf32 = True
        mgr = HookManager()
        msg = apply_runtime_flags_pass(cfg, mgr, torch_mod=tmod)
        self.assertTrue(msg.startswith("SKIPPED:"))
        self.assertFalse(state["cuda_backends_touched"])


class TestMemoryAllocatorPrelaunch(unittest.TestCase):
    """Tests for pre-launch allocator and CPU thread env setup."""

    def test_sets_env_without_overwriting_user_vars(self) -> None:
        cfg = OptimizeConfig()
        cfg.runtime.alloc_conf = "expandable_segments:True"
        cfg.runtime.omp_threads = 6
        cfg.runtime.jemalloc = False

        env: dict[str, str] = {}
        apply_prelaunch_env(cfg, env)
        self.assertEqual(env["PYTORCH_CUDA_ALLOC_CONF"], "expandable_segments:True")
        self.assertEqual(env["OMP_NUM_THREADS"], "6")
        self.assertEqual(env["MKL_NUM_THREADS"], "6")

        env_user = {"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128", "OMP_NUM_THREADS": "2"}
        apply_prelaunch_env(cfg, env_user)
        self.assertEqual(env_user["PYTORCH_CUDA_ALLOC_CONF"], "max_split_size_mb:128")
        self.assertEqual(env_user["OMP_NUM_THREADS"], "2")

        cfg.compile.enabled = True
        cfg.compile.cache_dir = ".ptopt_test_cache"
        env_cache: dict[str, str] = {}
        apply_prelaunch_env(cfg, env_cache)
        self.assertIn("TORCHINDUCTOR_CACHE_DIR", env_cache)
        self.assertEqual(env_cache["TORCHINDUCTOR_FX_GRAPH_CACHE"], "1")
        self.assertEqual(env_cache["TORCHINDUCTOR_AUTOGRAD_CACHE"], "1")
        self.assertTrue(env_cache["TRITON_CACHE_DIR"].endswith("triton"))


class TestChannelsLastPass(unittest.TestCase):
    """Tests for Conv2d-only channels_last conversion and contiguous state_dict export."""

    def test_conv2d_converted_and_mlp_skipped_and_state_dict_contiguous(self) -> None:
        cnn = nn.Sequential(nn.Conv2d(3, 8, kernel_size=3, padding=1), nn.ReLU(), nn.Conv2d(8, 4, 3, padding=1))
        mlp = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))

        self.assertFalse(convert_module_channels_last(mlp, torch))
        self.assertTrue(convert_module_channels_last(cnn, torch))
        self.assertTrue(cnn[0].weight.is_contiguous(memory_format=torch.channels_last))
        self.assertFalse(cnn[0].weight.is_contiguous())

        # Hooked forward converts 4D input tensor without error and state_dict returns standard contiguous layout
        install_checkpoint_compat_hooks(torch)
        install_channels_last_hook(torch)
        try:
            x = torch.randn(2, 3, 16, 16)
            out = cnn(x)
            self.assertEqual(out.shape, (2, 4, 16, 16))

            sd = cnn.state_dict()
            for val in sd.values():
                if val.dim() == 4:
                    self.assertTrue(val.is_contiguous())

            # Optimizer 4D buffers (e.g. SGD momentum_buffer) & injected fused flag must also normalize cleanly
            opt = torch.optim.SGD(cnn.parameters(), lr=0.05, momentum=0.9)
            opt._ptopt_injected_fused = True  # type: ignore[attr-defined]
            cnn(torch.randn(2, 3, 16, 16)).sum().backward()
            opt.step()
            osd = opt.state_dict()
            self.assertFalse(osd["param_groups"][0]["fused"])
            for st in osd["state"].values():
                buf = st.get("momentum_buffer")
                if buf is not None and buf.dim() == 4:
                    self.assertTrue(buf.is_contiguous())
        finally:
            uninstall_channels_last_hook(torch)
            uninstall_checkpoint_compat_hooks(torch)

    def test_auto_channels_last_converts_only_large_cuda_conv2d(self) -> None:
        cnn_small = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        cnn_large = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        mgr = HookManager()
        cfg = OptimizeConfig(profile="balanced")

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PTOPT_CHANNELS_LAST", None)
            with patch.object(torch.cuda, "is_available", return_value=True):
                msg = apply_channels_last_pass(cfg, mgr, torch_mod=torch)
            self.assertIn("Auto channels_last hook installed", msg)
            try:
                # 64x64 is below 128x128 spatial threshold -> stays contiguous NCHW
                with patch("src.passes.channels_last._has_large_4d_input", return_value=False):
                    cnn_small(torch.randn(1, 3, 64, 64))
                self.assertTrue(cnn_small.weight.is_contiguous())
                self.assertTrue(any(ev.status == "SKIPPED" and "below auto threshold" in ev.reason for ev in mgr.events))

                # >=128x128 CUDA input -> converts automatically to channels_last
                with patch("src.passes.channels_last._has_large_4d_input", return_value=True):
                    cnn_large(torch.randn(1, 3, 128, 128))
                self.assertTrue(cnn_large.weight.is_contiguous(memory_format=torch.channels_last))
                self.assertTrue(any(ev.status == "APPLIED" and "Converted Conv2d" in ev.reason for ev in mgr.events))
            finally:
                uninstall_channels_last_hook(torch)
                mgr.reset()


if __name__ == "__main__":
    unittest.main()
