"""Regression tests for general PyTorch workloads (LLM/NLP/CV/Regression/Signals) & review fixes."""

from __future__ import annotations

import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F

from ptopt.cli.main import build_child_env
from ptopt.config import ConfigResolver, OptimizeConfig, build_runtime_cache_key, validate_or_scope_cache_dir
from ptopt.detect import GPUDeviceInfo, GPUProbeResult, HardwareFingerprint, HostInfo
from ptopt.hooks.checkpoint_compat import install_checkpoint_compat_hooks, uninstall_checkpoint_compat_hooks
from ptopt.hooks.manager import HookManager
from ptopt.kernels.cross_entropy import triton_cross_entropy
from ptopt.kernels.registry import (
    _patch_module_submodules,
    apply_custom_kernels_pass,
    install_custom_kernels_hooks,
    uninstall_custom_kernels_hooks,
)
from ptopt.passes.auto_amp import (
    _FUNCTIONAL_LOSS_FP32_SPECS,
    install_auto_amp_hook,
    uninstall_auto_amp_hook,
)
from ptopt.passes.auto_compile import (
    _disable_cudnn_benchmark_for_dynamic_conv,
    _has_explicit_dynamic_override,
    _is_worth_static_compile,
)
from ptopt.passes.channels_last import _is_conv_dominated_architecture
from ptopt.passes.dataloader_tune import (
    _PTOPTWorkerInitWrapper,
    decide_dataloader_params,
    install_dataloader_hook,
    uninstall_dataloader_hook,
)
from ptopt.passes.inference_opt import (
    _slice_fp8_calib_inputs,
    install_inference_opt_hook,
    uninstall_inference_opt_hook,
)
from ptopt.passes.memory_allocator import apply_prelaunch_env
from ptopt.passes.optimizer_fusion import install_optimizer_fusion_hook, uninstall_optimizer_fusion_hook
from ptopt.passes.runtime_flags import apply_runtime_flags_pass


class TestGeneralWorkloadsAndFixes(unittest.TestCase):
    def tearDown(self) -> None:
        uninstall_auto_amp_hook(torch)
        uninstall_optimizer_fusion_hook(torch)
        uninstall_dataloader_hook(torch)
        uninstall_inference_opt_hook(torch)
        uninstall_custom_kernels_hooks(torch)
        uninstall_checkpoint_compat_hooks(torch)

    def test_amp_clip_grad_norm_unscales_before_clipping(self) -> None:
        cfg = OptimizeConfig(profile="balanced")
        cfg.amp.enabled = True
        cfg.amp.dtype = "fp16"
        cfg.amp.grad_scaler = True
        mgr = HookManager()

        class _FakeScaler:
            def __init__(self, *_a: object, **_kw: object) -> None:
                self.unscale_calls = 0
                self.step_calls = 0
                self.update_calls = 0

            def is_enabled(self) -> bool:
                return True

            def scale(self, t: torch.Tensor) -> torch.Tensor:
                return t * 65536.0

            def unscale_(self, opt: torch.optim.Optimizer) -> None:
                self.unscale_calls += 1
                for grp in opt.param_groups:
                    for p in grp["params"]:
                        if p.grad is not None:
                            p.grad.div_(65536.0)

            def step(self, opt: torch.optim.Optimizer, *a: object, **kw: object) -> object:
                self.step_calls += 1
                return opt.step(*a, **kw)

            def update(self) -> None:
                self.update_calls += 1

        with patch.object(torch.amp, "GradScaler", _FakeScaler):
            rt = install_auto_amp_hook(cfg, mgr, torch)
            assert rt is not None
            try:
                p = nn.Parameter(torch.tensor([2.0, 0.0]))
                opt = torch.optim.SGD([p], lr=0.1)
                loss = p[0] * 1.0  # true grad wrt p is [1.0, 0.0]

                orig_is_cuda = torch.Tensor.is_cuda
                with patch.object(torch.Tensor, "is_cuda", new_callable=lambda: property(lambda _s: True)):
                    loss.backward()
                    total_norm = torch.nn.utils.clip_grad_norm_([p], max_norm=10.0)
                    opt.step()

                self.assertEqual(rt.scaler.unscale_calls, 1)
                self.assertEqual(rt.scaler.step_calls, 1)
                self.assertAlmostEqual(float(total_norm), 1.0, places=4)
                # p[0] should step by 0.1 * 1.0 = 0.1 -> 1.9 (not 0.1 * 10.0 or 65536x)
                self.assertAlmostEqual(float(p[0].item()), 1.9, places=4)
            finally:
                uninstall_auto_amp_hook(torch)

    def test_amp_and_optimizer_fusion_support_custom_optimizer_subclasses(self) -> None:
        class _CustomForeachSGD(torch.optim.Optimizer):
            def __init__(self, params: object, lr: float = 0.05, foreach: bool | None = None) -> None:
                super().__init__(params, {"lr": lr, "foreach": foreach})
                self.steps_run = 0

            def step(self, closure: object = None) -> None:
                self.steps_run += 1

        cfg = OptimizeConfig(profile="balanced")
        cfg.optimizer.fused = False
        cfg.optimizer.foreach = True
        cfg.amp.enabled = True
        cfg.amp.dtype = "bf16"
        mgr = HookManager()

        ok = install_optimizer_fusion_hook(cfg, mgr, torch)
        self.assertTrue(ok)
        rt = install_auto_amp_hook(cfg, mgr, torch)
        assert rt is not None
        try:
            p = nn.Parameter(torch.randn(4))
            opt = _CustomForeachSGD([p], lr=0.05)
            self.assertTrue(opt.defaults["foreach"])
            opt.step()
            self.assertEqual(opt.steps_run, 1)
            self.assertEqual(rt._amp_step_count, 1)
        finally:
            uninstall_auto_amp_hook(torch)
            uninstall_optimizer_fusion_hook(torch)

    def test_amp_fft_stft_fp32_guard_and_float64_skip(self) -> None:
        cfg = OptimizeConfig(profile="balanced")
        cfg.amp.enabled = True
        cfg.amp.dtype = "bf16"
        mgr = HookManager()

        rt = install_auto_amp_hook(cfg, mgr, torch)
        assert rt is not None
        autocast_calls: list[tuple[str, bool, object]] = []

        class _FakeAutocast:
            def __init__(self, device_type: str = "cuda", dtype: object = None, enabled: bool = True) -> None:
                self.args = (device_type, enabled, dtype)

            def __enter__(self) -> "_FakeAutocast":
                autocast_calls.append(self.args)
                return self

            def __exit__(self, *_a: object) -> None:
                return None

        class _SignalFFTNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w = nn.Parameter(torch.randn(8))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                spec = torch.fft.rfft(x, dim=-1)
                return spec.abs().sum() + self.w.sum()

        class _Float64SciNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin = nn.Linear(4, 4, dtype=torch.float64)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.lin(x)

        sig_net = _SignalFFTNet()
        sci_net = _Float64SciNet()
        try:
            with (
                patch.object(torch.amp, "autocast", _FakeAutocast),
                patch.object(torch.Tensor, "is_cuda", new_callable=lambda: property(lambda _s: True)),
            ):
                _ = sig_net(torch.randn(2, 16))
                self.assertIn(("cuda", True, torch.bfloat16), autocast_calls)
                self.assertIn(("cuda", False, None), autocast_calls)

                autocast_calls.clear()
                _ = sci_net(torch.randn(2, 4, dtype=torch.float64))
                self.assertEqual(autocast_calls, [])
        finally:
            uninstall_auto_amp_hook(torch)

    def test_amp_regression_and_signal_loss_fp32_specs(self) -> None:
        for loss_name in (
            "mse_loss",
            "l1_loss",
            "smooth_l1_loss",
            "huber_loss",
            "ctc_loss",
            "binary_cross_entropy",
            "gaussian_nll_loss",
        ):
            self.assertIn(loss_name, _FUNCTIONAL_LOSS_FP32_SPECS)

    def test_amp_multi_model_forward_skip_does_not_clobber_scaled_step(self) -> None:
        cfg = OptimizeConfig(profile="balanced")
        cfg.amp.enabled = True
        cfg.amp.dtype = "bf16"
        mgr = HookManager()
        rt = install_auto_amp_hook(cfg, mgr, torch)
        assert rt is not None

        class _FakeAutocast:
            def __init__(self, **_kw: object) -> None:
                pass

            def __enter__(self) -> None:
                return None

            def __exit__(self, *_a: object) -> None:
                return None

        big = nn.Sequential(nn.Linear(64, 512), nn.ReLU(), nn.Linear(512, 64))
        tiny_post = nn.Linear(4, 4)
        tiny_post._ptopt_compile_ineligible = True
        try:
            with (
                patch.object(torch.amp, "autocast", _FakeAutocast),
                patch.object(torch.Tensor, "is_cuda", new_callable=lambda: property(lambda _s: True)),
            ):
                _ = big(torch.randn(4, 64))
                self.assertTrue(rt._any_amp_forward_this_step)
                _ = tiny_post(torch.randn(2, 4))
                self.assertTrue(rt._last_forward_skipped_amp)
                self.assertTrue(rt._any_amp_forward_this_step)
                # Backward should still run AMP watchdog (resetting flags) because big ran AMP this step
                out = torch.tensor(1.0, requires_grad=True)
                out.sum().backward()
                self.assertFalse(rt._any_amp_forward_this_step)
                self.assertFalse(rt._last_forward_skipped_amp)
        finally:
            uninstall_auto_amp_hook(torch)

    def test_dataloader_iterable_tensordataset_worker_init_and_remainder_scoping(self) -> None:
        cfg = OptimizeConfig(profile="balanced")
        mgr = HookManager()

        class _StreamDS(torch.utils.data.IterableDataset):
            def __iter__(self):
                yield torch.tensor([1.0])

        tds = torch.utils.data.TensorDataset(torch.randn(32, 4))
        tuned_it, notes_it = decide_dataloader_params(
            cfg=cfg,
            dataset=_StreamDS(),
            collate_fn=None,
            worker_init_fn=None,
            orig_num_workers=0,
            orig_pin_memory=False,
            orig_persistent_workers=False,
            orig_prefetch_factor=None,
            torch_mod=torch,
            shm_mb=4096.0,
            cpu_cores=8,
        )
        self.assertEqual(tuned_it["num_workers"], 0)
        self.assertTrue(any("IterableDataset" in r for _s, r in notes_it))

        tuned_tds, notes_tds = decide_dataloader_params(
            cfg=cfg,
            dataset=tds,
            collate_fn=None,
            worker_init_fn=None,
            orig_num_workers=0,
            orig_pin_memory=False,
            orig_persistent_workers=False,
            orig_prefetch_factor=None,
            torch_mod=torch,
            shm_mb=4096.0,
            cpu_cores=8,
        )
        self.assertEqual(tuned_tds["num_workers"], 0)
        self.assertTrue(any("TensorDataset" in r for _s, r in notes_tds))

        # Worker init wrapper must be picklable and set OMP_NUM_THREADS=1
        wrapper = _PTOPTWorkerInitWrapper(None)
        reloaded = pickle.loads(pickle.dumps(wrapper))
        with patch.dict(os.environ, {}, clear=True):
            reloaded(0)
            self.assertEqual(os.environ.get("OMP_NUM_THREADS"), "1")

        # Per-loader remainder flag toggles on __iter__
        install_dataloader_hook(cfg, mgr, torch, shm_probe_fn=lambda: 4096.0)
        try:
            dl_odd = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.arange(10)), batch_size=4)
            dl_even = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.arange(8)), batch_size=4)
            for _ in dl_odd:
                self.assertTrue(getattr(torch, "_ptopt_dataloader_has_remainder", False))
                break
            for _ in dl_even:
                self.assertFalse(getattr(torch, "_ptopt_dataloader_has_remainder", True))
                break
        finally:
            uninstall_dataloader_hook(torch)

    def test_channels_last_skips_vit_and_converts_cnn(self) -> None:
        class _MiniViT(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.patch = nn.Conv2d(3, 32, kernel_size=4, stride=4)
                self.attn = nn.MultiheadAttention(32, num_heads=4, batch_first=True)

        class _MiniCNN(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.c1 = nn.Conv2d(3, 16, 3, padding=1)
                self.c2 = nn.Conv2d(16, 32, 3, padding=1)

        self.assertFalse(_is_conv_dominated_architecture(_MiniViT(), torch))
        self.assertTrue(_is_conv_dominated_architecture(_MiniCNN(), torch))

    def test_rocm_profile_rules_and_cpu_flush_denormal(self) -> None:
        rocm_host = HostInfo(
            hostname="rocm-host",
            os_name="Linux",
            os_release="6.6.0",
            cpu_arch="x86_64",
            cpu_model="EPYC",
            physical_cores=16,
            logical_cores=32,
            isa_flags=["avx2"],
            total_ram_mb=131072.0,
            available_ram_mb=100000.0,
            shm_total_mb=8192.0,
            shm_free_mb=8192.0,
            is_colab=False,
            python_version="3.12.0",
            torch_version="2.6.0+rocm6.2",
            cuda_version=None,
            triton_version="3.1.0",
            jemalloc_path=None,
        )
        rocm_dev = GPUDeviceInfo(
            index=0,
            name="AMD Instinct MI210",
            compute_capability="9.0",
            major=9,
            minor=0,
            sm_int=90,
            sm_name="gfx90a",
            total_vram_mb=65536.0,
            sm_count=104,
            bf16_supported=True,
            tf32_supported=False,
            fp8_supported=False,
            flash_sdp_supported=True,
            mem_efficient_sdp_supported=True,
        )
        rocm_fp = HardwareFingerprint(
            host=rocm_host,
            gpu=GPUProbeResult(backend="rocm", device_count=1, selected_index=0, cuda_visible_devices=None, devices=[rocm_dev]),
        )
        res = ConfigResolver().resolve(rocm_fp, profile="balanced")
        self.assertTrue(res.config.amp.enabled)
        self.assertEqual(res.config.amp.dtype, "fp16")
        self.assertTrue(res.config.compile.enabled)
        self.assertFalse(res.config.runtime.tf32)
        self.assertFalse(res.config.optimizer.fused)
        self.assertFalse(res.config.runtime.cudnn_benchmark)

        mgr = HookManager()
        cfg_bal = OptimizeConfig(profile="balanced")
        msg_bal = apply_runtime_flags_pass(cfg_bal, mgr, torch)
        self.assertIn("flush_denormal=on", msg_bal)
        cfg_safe = OptimizeConfig(profile="safe")
        msg_safe = apply_runtime_flags_pass(cfg_safe, mgr, torch)
        self.assertTrue(msg_safe.startswith("SKIPPED"))

    def test_explicit_override_propagation_compile_depth_and_conv1d(self) -> None:
        cfg = OptimizeConfig(profile="balanced")
        env = build_child_env(
            cfg.to_dict(),
            explicit_overrides=["compile.dynamic", "kernels.enabled"],
        )
        self.assertEqual(env["PTOPT_EXPLICIT_OVERRIDES"], "compile.dynamic,kernels.enabled")

        with patch.dict(os.environ, {"PTOPT_EXPLICIT_OVERRIDES": "compile.dynamic"}):
            self.assertTrue(_has_explicit_dynamic_override(cfg))

        # Deep small signal model (depth >= 8) should be eligible for static compile
        deep_signal = nn.Sequential(*[nn.Conv1d(8, 8, 3, padding=1) for _ in range(8)])
        ok_deep, reason_deep = _is_worth_static_compile(deep_signal, (torch.randn(2, 8, 32),), {}, torch, False)
        self.assertTrue(ok_deep)
        self.assertIn("depth=", reason_deep)

        # Dynamic Conv1d disables cudnn.benchmark cleanly
        mgr = HookManager()
        fake_cudnn = SimpleNamespace(benchmark=True)
        fake_torch = SimpleNamespace(nn=nn, backends=SimpleNamespace(cudnn=fake_cudnn))
        _disable_cudnn_benchmark_for_dynamic_conv(deep_signal, fake_torch, mgr, "conv1d dynamic")
        self.assertFalse(fake_cudnn.benchmark)

    def test_inference_cudagraph_mark_step_and_1d_fp8_slice(self) -> None:
        # 1D batch tensor in FP8 calibration must be sliced along dim 0
        calib_args, calib_kw = _slice_fp8_calib_inputs(
            (torch.randn(32, 64),),
            {"timesteps": torch.arange(32)},
            torch,
            step_idx=0,
            min_rows=8,
        )
        self.assertEqual(calib_args[0].shape[0], 16)
        self.assertEqual(calib_kw["timesteps"].shape[0], 16)

        cfg = OptimizeConfig(profile="balanced")
        cfg.inference.cuda_graphs = True
        mgr = HookManager()
        mark_calls = 0

        def _fake_mark_step_begin() -> None:
            nonlocal mark_calls
            mark_calls += 1

        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.Tensor, "is_cuda", new_callable=lambda: property(lambda _s: True)),
            patch.object(torch, "compile", side_effect=lambda fn, **_kw: fn),
            patch.object(torch.compiler, "cudagraph_mark_step_begin", _fake_mark_step_begin),
        ):
            install_inference_opt_hook(cfg, mgr, torch)
            try:
                m = nn.Linear(8, 4).eval()
                loss_mod = nn.MSELoss().eval()
                _ = m(torch.randn(2, 8))
                _ = m(torch.randn(2, 8))
                _ = m(torch.randn(2, 8))
                self.assertGreaterEqual(mark_calls, 1)
                _ = loss_mod(torch.randn(2, 4), torch.randn(2, 4))
                _ = loss_mod(torch.randn(2, 4), torch.randn(2, 4))
                self.assertFalse(getattr(loss_mod, "_ptopt_infer_cg_active", False))
            finally:
                uninstall_inference_opt_hook(torch)

    def test_kernel_gemma_skip_variance_epsilon_no_kernels_and_soft_ce(self) -> None:
        class _GemmaRMSNorm(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.ones(8))
                self.eps = 1e-6

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x * (1.0 + self.weight)

        class _HFRMSNorm(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.ones(8))
                self.variance_epsilon = 1e-5

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x * self.weight

        gemma = _GemmaRMSNorm()
        hf_rms = _HFRMSNorm()
        mgr = HookManager()
        called_eps: list[float] = []

        with (
            patch.object(torch.Tensor, "is_cuda", new_callable=lambda: property(lambda _s: True)),
            patch(
                "ptopt.kernels.rmsnorm.triton_rmsnorm",
                side_effect=lambda x, w, eps=1e-6: called_eps.append(eps) or x,
            ),
        ):
            patched = _patch_module_submodules(
                nn.Sequential(gemma, hf_rms),
                torch,
                {"rmsnorm": True},
                mgr,
                min_lce_vocab=4096,
                fallback_on_error=True,
            )
            self.assertEqual(patched, 1)
            self.assertFalse(getattr(gemma, "_ptopt_kern_patched", False))
            self.assertTrue(getattr(hf_rms, "_ptopt_kern_patched", False))

            _ = hf_rms(torch.randn(2, 8))
            self.assertEqual(called_eps, [1e-5])

        # --no-kernels honors PTOPT_EXPLICIT_OVERRIDES
        cfg_no_k = OptimizeConfig(profile="balanced")
        cfg_no_k.kernels.enabled = False
        cfg_no_k.compile.enabled = True
        with patch.dict(os.environ, {"PTOPT_EXPLICIT_OVERRIDES": "kernels.enabled"}):
            msg = apply_custom_kernels_pass(cfg_no_k, mgr, torch)
            self.assertTrue(msg.startswith("SKIPPED"))

        # Soft targets (Mixup float target) skip Triton CE cleanly; hard targets pass COUNT_VALID=False
        captured_kw: list[dict[str, object]] = []

        class _FakeKernel:
            def __getitem__(self, _grid: object):
                return lambda *a, **kw: captured_kw.append(kw)

        fake_kernel = _FakeKernel()
        with (
            patch.object(torch.Tensor, "is_cuda", new_callable=lambda: property(lambda _s: True)),
            patch("ptopt.kernels.cross_entropy.HAS_TRITON", True),
            patch("ptopt.kernels.cross_entropy._cross_entropy_fwd_kernel", fake_kernel, create=True),
            patch("ptopt.kernels.cross_entropy.triton", SimpleNamespace(next_power_of_2=lambda n: 128)),
        ):
            _ = triton_cross_entropy(torch.randn(4, 16), torch.tensor([0, 1, 2, 3]), reduction="mean")
            self.assertEqual(len(captured_kw), 1)
            self.assertFalse(captured_kw[0]["COUNT_VALID"])

        # Soft target via F.cross_entropy with installed kernel hook falls back to native CE
        with patch("ptopt.kernels.registry.check_kernel_capability", return_value=(True, "ok")):
            cfg_ce = OptimizeConfig(profile="aggressive")
            cfg_ce.kernels.enabled = True
            install_custom_kernels_hooks(cfg_ce, mgr, torch, triton_available=True, sm_override=89)
            try:
                logits = torch.randn(3, 5)
                soft_t = torch.softmax(torch.randn(3, 5), dim=-1)
                with patch.object(torch.Tensor, "is_cuda", new_callable=lambda: property(lambda _s: True)):
                    out = F.cross_entropy(logits, soft_t)
                self.assertEqual(out.shape, ())
            finally:
                uninstall_custom_kernels_hooks(torch)

    def test_checkpoint_compat_restores_dropped_fp8_weights(self) -> None:
        install_checkpoint_compat_hooks(torch)
        try:
            lin = nn.Linear(4, 3, bias=False)
            w_ref = torch.tensor(
                [[1.0, 2.0, -1.0, 0.5], [0.0, -2.0, 3.0, 1.0], [1.5, 0.5, 0.25, -1.0]],
                dtype=torch.float32,
            )
            lin._ptopt_fp8_w_cache = (None, w_ref.t().contiguous(), torch.tensor(0.5, dtype=torch.float32))
            # Simulate PTOPT_FP8_DROP_BF16_EVAL=1 emptying .weight.data
            lin.weight.data = torch.empty(0, dtype=torch.bfloat16)
            sd = lin.state_dict()
            self.assertEqual(sd["weight"].shape, (3, 4))
            torch.testing.assert_close(sd["weight"], (w_ref * 0.5).to(torch.bfloat16))
        finally:
            uninstall_checkpoint_compat_hooks(torch)

    def test_memory_allocator_preserves_custom_inductor_cache_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "default_cache"
            custom = Path(tmp) / "user_custom_cache"
            k_old = build_runtime_cache_key("sm_75", "2.4.0+cu121", "3.0.0")
            k_new = build_runtime_cache_key("sm_89", "2.5.1+cu121", "3.1.0")
            validate_or_scope_cache_dir(base, k_old, write_meta=True, emit_warnings=False)

            cfg = OptimizeConfig(profile="balanced")
            cfg.compile.enabled = True
            cfg.compile.cache_dir = str(base)
            env = {"TORCHINDUCTOR_CACHE_DIR": str(custom)}
            with patch("ptopt.passes.memory_allocator.detect_runtime_cache_key", return_value=k_new):
                apply_prelaunch_env(cfg, env)
            self.assertEqual(env["TORCHINDUCTOR_CACHE_DIR"], str(custom))


if __name__ == "__main__":
    unittest.main()
