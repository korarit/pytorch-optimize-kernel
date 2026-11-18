"""Unit tests for Phase 6.3 features & Guardrails #1 and #5."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import OptimizeConfig  # noqa: E402
from src.detect.fingerprint import collect_fingerprint  # noqa: E402
from src.hooks.checkpoint_compat import (  # noqa: E402
    install_checkpoint_compat_hooks,
    uninstall_checkpoint_compat_hooks,
)
from src.hooks.manager import HookManager  # noqa: E402
from src.kernels.registry import (  # noqa: E402
    _is_verified_rope_inputs,
    _match_swiglu_mlp,
)
from src.passes.channels_last import (  # noqa: E402
    has_conv3d,
    install_channels_last_hook,
    uninstall_channels_last_hook,
)
from src.passes.inference_opt import (  # noqa: E402
    _estimate_token_rows,
    _quantize_fp8_static_traceable,
    _update_fp8_static_scale,
    apply_fp8_linear_eval_hooks,
)
from src.passes.optimizer_fusion import (  # noqa: E402
    install_optimizer_fusion_hook,
    uninstall_optimizer_fusion_hook,
)


class _LlamaStyleSwiGLU(nn.Module):
    def __init__(self, d: int = 16, h: int = 32) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d, h, bias=False)
        self.up_proj = nn.Linear(d, h, bias=False)
        self.down_proj = nn.Linear(h, d, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class _GemmaStyleGeGLU(nn.Module):
    def __init__(self, d: int = 16, h: int = 32) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d, h, bias=False)
        self.up_proj = nn.Linear(d, h, bias=False)
        self.down_proj = nn.Linear(h, d, bias=False)
        self.act_fn = nn.GELU(approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class _SwiGLUWithDropout(nn.Module):
    def __init__(self, d: int = 16, h: int = 32) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d, h, bias=False)
        self.up_proj = nn.Linear(d, h, bias=False)
        self.down_proj = nn.Linear(h, d, bias=False)
        self.act_fn = nn.SiLU()
        self.drop = nn.Dropout(0.1)


class _FunctionalMysteryMLP(nn.Module):
    def __init__(self, d: int = 16, h: int = 32) -> None:
        super().__init__()
        self.w_gate = nn.Linear(d, h, bias=False)
        self.w_up = nn.Linear(d, h, bias=False)
        self.w_down = nn.Linear(h, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(torch.relu(self.w_gate(x)) * self.w_up(x))


class TestPhase63Features(unittest.TestCase):
    """Tests for Phase 6.3 LLM semantic matching, FP8 static/MHA, heuristic & cache polish."""

    def test_guardrail1_strict_swiglu_and_rope_matching(self) -> None:
        # Matches HF Llama/Qwen style SwiGLU
        llama = _LlamaStyleSwiGLU()
        matched = _match_swiglu_mlp(llama, torch)
        self.assertIsNotNone(matched)
        assert matched is not None
        self.assertIs(matched[0], llama.gate_proj)

        # Negative tests: MUST reject GeGLU (GELU), Dropout-containing MLP, and functional ReGLU
        self.assertIsNone(_match_swiglu_mlp(_GemmaStyleGeGLU(), torch))
        self.assertIsNone(_match_swiglu_mlp(_SwiGLUWithDropout(), torch))
        self.assertIsNone(_match_swiglu_mlp(_FunctionalMysteryMLP(), torch))

        # RoPE semantic layout verifier rejects interleaved [c0, c0, c1, c1] tables
        q = SimpleNamespace(is_cuda=True, dim=lambda: 4, shape=(2, 4, 8, 8))
        k = SimpleNamespace(is_cuda=True, dim=lambda: 4, shape=(2, 4, 8, 8))
        half_cos = torch.cat([torch.arange(4.0), torch.arange(4.0)], dim=-1).view(1, 1, 1, 8)
        inter_cos = torch.tensor([0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]).view(1, 1, 1, 8)
        with patch.object(torch.Tensor, "is_cuda", True) if False else patch("src.kernels.registry.getattr", create=True):
            pass

        class _MockCudaTensor:
            def __init__(self, t: torch.Tensor) -> None:
                self._t = t
                self.is_cuda = True
                self.shape = t.shape

            def dim(self) -> int:
                return self._t.dim()

            def __getitem__(self, item: object) -> torch.Tensor:
                return self._t[item]

        self.assertTrue(
            _is_verified_rope_inputs(_MockCudaTensor(torch.zeros(2, 4, 8, 8)), _MockCudaTensor(torch.zeros(2, 4, 8, 8)), _MockCudaTensor(half_cos), _MockCudaTensor(half_cos))
        )
        self.assertFalse(
            _is_verified_rope_inputs(_MockCudaTensor(torch.zeros(2, 4, 8, 8)), _MockCudaTensor(torch.zeros(2, 4, 8, 8)), _MockCudaTensor(inter_cos), _MockCudaTensor(inter_cos))
        )

    def test_2d_token_has_large_m_and_static_fp8_scale_calibration(self) -> None:
        # 2D integer token IDs (16, 64) -> M = 1024 >= 512, whereas 2D float (16, 64) -> M = 16
        tok_ids = torch.randint(0, 1000, (16, 64), dtype=torch.long)
        feat_2d = torch.randn(16, 64, dtype=torch.float32)
        self.assertEqual(_estimate_token_rows(tok_ids), 1024)
        self.assertEqual(_estimate_token_rows(feat_2d), 16)

        if not hasattr(torch, "float8_e4m3fn"):
            self.skipTest("float8_e4m3fn unavailable")

        # Static scale calibration updates EMA with headroom and quantizes cleanly in 1 pass
        lin = nn.Linear(128, 128).eval()
        _update_fp8_static_scale(lin, torch.tensor(0.05), torch.float32, headroom=1.08, calib_steps=2)
        self.assertFalse(getattr(lin, "_ptopt_fp8_static_ready", False))
        _update_fp8_static_scale(lin, torch.tensor(0.06), torch.float32, headroom=1.08, calib_steps=2)
        self.assertTrue(getattr(lin, "_ptopt_fp8_static_ready", False))

        x = torch.randn(32, 128, dtype=torch.bfloat16)
        q = _quantize_fp8_static_traceable(x, lin._ptopt_fp8_x_inv_scale, torch.float8_e4m3fn, torch.float32)
        self.assertEqual(q.dtype, torch.float8_e4m3fn)
        self.assertTrue(q.is_contiguous())

        # Verify nn.MultiheadAttention in_proj_weight triggers FP8 MHA patching
        mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True).eval()
        with patch("src.passes.inference_opt._is_cuda_tensor", return_value=True):
            patched = apply_fp8_linear_eval_hooks(mha, torch, min_work=512 * 512)
        self.assertEqual(patched, 1)
        self.assertTrue(getattr(mha, "_ptopt_fp8_mha_patched", False))

    def test_channels_last_volume_heuristic_and_conv3d(self) -> None:
        mgr = HookManager()
        # Case 1: High-volume 4D (B=64, C=64, H=64, W=64) -> B*C=4096 & H*W=4096 < 16384 -> NOW converts!
        cnn = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1))
        install_channels_last_hook(torch, min_spatial_area=128 * 128, require_cuda=False, manager=mgr)
        try:
            _ = cnn(torch.randn(64, 64, 64, 64))
            self.assertTrue(getattr(cnn, "_ptopt_channels_last_done", False))
            self.assertTrue(cnn[0].weight.is_contiguous(memory_format=torch.channels_last))
        finally:
            uninstall_channels_last_hook(torch)
            mgr.reset()

        # Case 2: Conv3d detection and channels_last_3d
        cnn3d = nn.Sequential(nn.Conv3d(4, 4, 3, padding=1))
        self.assertTrue(has_conv3d(cnn3d, torch))
        install_channels_last_hook(torch, min_spatial_area=0, require_cuda=False, manager=mgr)
        try:
            _ = cnn3d(torch.randn(2, 4, 8, 8, 8))
            self.assertTrue(getattr(cnn3d, "_ptopt_channels_last_done", False))
            self.assertTrue(getattr(cnn3d, "_ptopt_channels_last_3d", False))
            self.assertTrue(cnn3d[0].weight.is_contiguous(memory_format=torch.channels_last_3d))
        finally:
            uninstall_channels_last_hook(torch)
            mgr.reset()

    def test_optimizer_foreach_extended_and_capturable_checkpoint_compat(self) -> None:
        cfg = OptimizeConfig(profile="balanced")
        cfg.optimizer.fused = False
        cfg.optimizer.foreach = True
        mgr = HookManager()
        install_checkpoint_compat_hooks(torch)
        install_optimizer_fusion_hook(cfg, mgr, torch)
        try:
            for opt_cls in (torch.optim.RMSprop, torch.optim.Adagrad, torch.optim.NAdam, torch.optim.RAdam):
                p = nn.Parameter(torch.randn(4, 4))
                opt = opt_cls([p], lr=1e-3)
                self.assertTrue(opt.param_groups[0].get("foreach", False), f"{opt_cls.__name__} missing foreach=True")

            # Verify capturable=False checkpoint step normalization on reload (Guardrail #5a)
            p_a = nn.Parameter(torch.randn(4, 4))
            opt_a = torch.optim.AdamW([p_a], lr=1e-3)
            opt_a._ptopt_injected_capturable = True  # simulate CUDA opt-in injection
            loss = (p_a**2).sum()
            loss.backward()
            opt_a.step()
            sd = opt_a.state_dict()
            self.assertFalse(sd["param_groups"][0].get("capturable", False))

            p_b = nn.Parameter(torch.randn(4, 4))
            opt_b = torch.optim.AdamW([p_b], lr=1e-3)
            opt_b.load_state_dict(sd)
            step_t = opt_b.state[p_b]["step"]
            self.assertEqual(str(step_t.device), "cpu")

            # Verify fused=True (capturable=False) keeps step on param.device instead of moving to CPU
            class _FakeTensor:
                def __init__(self, dev: str) -> None:
                    self.device = dev

                def dim(self) -> int:
                    return 2

                def to(self, device: object = None) -> "_FakeTensor":
                    return _FakeTensor(str(device))

            p_fake = _FakeTensor("cuda:0")
            s_cpu = _FakeTensor("cpu")

            class _FakeMod:
                def state_dict(self) -> dict[str, object]:
                    return {}

                def load_state_dict(self, sd: object) -> None:
                    pass

            class _FakeOpt:
                def __init__(self) -> None:
                    self.param_groups = [{"params": [p_fake], "fused": True, "capturable": False}]
                    self.state = {p_fake: {"step": s_cpu}}

                def state_dict(self) -> dict[str, object]:
                    return {}

                def load_state_dict(self, _sd: object) -> None:
                    pass

            fake_torch = SimpleNamespace(
                nn=SimpleNamespace(Module=_FakeMod),
                optim=SimpleNamespace(Optimizer=_FakeOpt),
                Tensor=_FakeTensor,
                channels_last="channels_last",
                contiguous_format="contiguous",
            )
            import src.hooks.checkpoint_compat as cc

            prev_flag = cc._COMPAT_INSTALLED
            cc._COMPAT_INSTALLED = False
            try:
                install_checkpoint_compat_hooks(fake_torch)
                opt_fake = _FakeOpt()
                opt_fake.load_state_dict({})
                self.assertEqual(str(opt_fake.state[p_fake]["step"].device), "cuda:0")
            finally:
                uninstall_checkpoint_compat_hooks(fake_torch)
                cc._COMPAT_INSTALLED = prev_flag
        finally:
            uninstall_optimizer_fusion_hook(torch)
            uninstall_checkpoint_compat_hooks(torch)
            mgr.reset()

    def test_fingerprint_disk_cache_invalidation_and_live_shm_guardrail5b(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            shm_vals = iter([(256.0, 128.0), (128.0, 64.0)])
            probe_host_calls = 0

            def _fake_probe_host(**_kw: object) -> object:
                nonlocal probe_host_calls
                probe_host_calls += 1
                from src.detect.host_probe import HostInfo

                return HostInfo(
                    hostname="node1",
                    os_name="Linux",
                    os_release="6.1",
                    cpu_arch="x86_64",
                    cpu_model="MockCPU",
                    physical_cores=4,
                    logical_cores=8,
                    shm_total_mb=1024.0,
                    shm_free_mb=512.0,
                    python_version="3.12",
                )

            def _fake_probe_gpu(**_kw: object) -> object:
                from src.detect.gpu_probe import GPUProbeResult

                return GPUProbeResult(backend="cpu", device_count=0, selected_index=0, cuda_visible_devices=None)

            env_patch = {"PTOPT_FP_CACHE_DIR": tmp, "PTOPT_NO_FP_CACHE": "0", "CUDA_VISIBLE_DEVICES": "0"}
            with patch.dict(os.environ, env_patch, clear=False), patch(
                "src.detect.fingerprint.probe_host", side_effect=_fake_probe_host
            ), patch("src.detect.fingerprint.probe_gpu", side_effect=_fake_probe_gpu), patch(
                "src.detect.fingerprint._probe_live_shm", side_effect=lambda: next(shm_vals)
            ):
                fp1 = collect_fingerprint()
                self.assertEqual(probe_host_calls, 1)
                self.assertEqual(fp1.host.shm_total_mb, 1024.0)

                # Cache hit skips probe_host/probe_gpu, but STILL probes /dev/shm live!
                fp2 = collect_fingerprint()
                self.assertEqual(probe_host_calls, 1)
                self.assertEqual(fp2.host.shm_total_mb, 256.0)

                # Changing CUDA_VISIBLE_DEVICES invalidates cache key
                os.environ["CUDA_VISIBLE_DEVICES"] = "1"
                _ = collect_fingerprint()
                self.assertEqual(probe_host_calls, 2)

                # PTOPT_NO_FP_CACHE=1 bypasses cache cleanly
                os.environ["PTOPT_NO_FP_CACHE"] = "1"
                _ = collect_fingerprint()
                self.assertEqual(probe_host_calls, 3)


if __name__ == "__main__":
    unittest.main()
