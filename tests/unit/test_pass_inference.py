"""Unit tests for Phase 3.6: Safe Inference Optimization (`inference_mode`, grads, FP8/INT8 gates, CUDA Graphs)."""

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

from ptopt.config import OptimizeConfig
from ptopt.hooks import HookManager
from ptopt.passes.inference_opt import (
    apply_inference_opt_pass,
    check_fp8_eligibility,
    install_inference_opt_hook,
    uninstall_inference_opt_hook,
)


class TestInferenceOptPass(unittest.TestCase):
    """Tests for train/eval switching, requires_grad guards, FP8 arch gate, and CUDA Graph fallback."""

    def test_train_eval_switching_preserves_backward_and_uses_inference_mode(self) -> None:
        cfg = OptimizeConfig(profile="safe")
        cfg.inference.inference_mode = True
        mgr = HookManager()
        rt = install_inference_opt_hook(cfg, mgr, torch)
        assert rt is not None

        seen_infer_flags: list[bool] = []

        class _SwitchingNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin = nn.Linear(8, 4)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                seen_infer_flags.append(torch.is_inference_mode_enabled())
                return self.lin(x)

        model = _SwitchingNet()
        opt = torch.optim.SGD(model.parameters(), lr=0.05)
        try:
            # Train step 1
            model.train()
            opt.zero_grad()
            loss1 = model(torch.randn(4, 8)).sum()
            loss1.backward()
            self.assertIsNotNone(model.lin.weight.grad)
            opt.step()

            # Eval step enters inference_mode automatically
            model.eval()
            with torch.no_grad():
                eval_out = model(torch.randn(2, 8))
            self.assertEqual(eval_out.shape, (2, 4))

            # Train step 2 back from eval still computes gradients cleanly
            model.train()
            opt.zero_grad()
            loss2 = model(torch.randn(4, 8)).sum()
            loss2.backward()
            self.assertIsNotNone(model.lin.weight.grad)
            opt.step()

            self.assertEqual(seen_infer_flags, [False, True, False])
            self.assertEqual(rt.inference_calls, 1)
        finally:
            uninstall_inference_opt_hook(torch)
            mgr.reset()

    def test_requires_grad_and_enable_grad_guards_skip_inference_mode(self) -> None:
        cfg = OptimizeConfig(profile="safe")
        cfg.inference.inference_mode = True
        mgr = HookManager()
        rt = install_inference_opt_hook(cfg, mgr, torch)
        assert rt is not None

        model = nn.Sequential(nn.Linear(6, 6), nn.Tanh(), nn.Linear(6, 2))
        model.eval()
        try:
            # 1) Input requires_grad=True during eval() (e.g. adversarial / PINN)
            x_adv = torch.randn(3, 6, requires_grad=True)
            out_adv = model(x_adv)
            self.assertTrue(out_adv.requires_grad)
            out_adv.sum().backward()
            self.assertIsNotNone(x_adv.grad)

            # 2) Explicit torch.enable_grad() during eval()
            x_plain = torch.randn(3, 6)
            with torch.enable_grad():
                out_en = model(x_plain)
                self.assertTrue(out_en.requires_grad)

            self.assertGreaterEqual(rt.grad_guard_skips, 2)
            self.assertEqual(rt.inference_calls, 0)
            self.assertTrue(any(ev.status == "SKIPPED" and "preserving autograd" in ev.reason for ev in mgr.events))
        finally:
            uninstall_inference_opt_hook(torch)
            mgr.reset()

    def test_fp8_hardware_gate_t4_skips_and_l4_allows(self) -> None:
        def _mock_cuda_torch(major: int, minor: int) -> SimpleNamespace:
            return SimpleNamespace(
                cuda=SimpleNamespace(
                    is_available=lambda: True,
                    current_device=lambda: 0,
                    get_device_properties=lambda _i: SimpleNamespace(major=major, minor=minor),
                ),
                float8_e4m3fn=object(),
                _scaled_mm=lambda *a, **kw: None,
                nn=torch.nn,
                enable_grad=torch.enable_grad,
                inference_mode=torch.inference_mode,
                Tensor=torch.Tensor,
            )

        t4_torch = _mock_cuda_torch(7, 5)
        ok_t4, reason_t4 = check_fp8_eligibility(t4_torch)
        self.assertFalse(ok_t4)
        self.assertIn("sm_75 < 89", reason_t4)

        l4_torch = _mock_cuda_torch(8, 9)
        ok_l4, reason_l4 = check_fp8_eligibility(l4_torch)
        self.assertTrue(ok_l4)
        self.assertIn("sm_89", reason_l4)

        cfg = OptimizeConfig(profile="aggressive")
        cfg.inference.inference_mode = True
        mgr_t4 = HookManager()
        with patch.dict(os.environ, {"PTOPT_INFERENCE_FP8": "1"}):
            install_inference_opt_hook(cfg, mgr_t4, t4_torch)
            try:
                self.assertTrue(any(ev.pass_name == "inference.fp8" and ev.status == "SKIPPED" for ev in mgr_t4.events))
            finally:
                uninstall_inference_opt_hook(t4_torch)
                mgr_t4.reset()

        mgr_l4 = HookManager()
        with patch.dict(os.environ, {"PTOPT_INFERENCE_FP8": "1"}):
            install_inference_opt_hook(cfg, mgr_l4, l4_torch)
            try:
                self.assertTrue(any(ev.pass_name == "inference.fp8" and ev.status == "APPLIED" for ev in mgr_l4.events))
            finally:
                uninstall_inference_opt_hook(l4_torch)
                mgr_l4.reset()

    def test_cuda_graphs_reduce_overhead_and_dynamic_shape_fallback(self) -> None:
        cfg = OptimizeConfig(profile="aggressive")
        cfg.inference.inference_mode = True
        cfg.inference.cuda_graphs = True
        mgr = HookManager()
        compile_modes: list[object] = []

        def _spy_compile(fn: object, **kwargs: object) -> object:
            compile_modes.append(kwargs.get("mode"))
            return fn

        model = nn.Linear(8, 4)
        model.eval()
        with patch("ptopt.passes.inference_opt._has_cuda_context", return_value=True), patch.object(
            torch, "compile", side_effect=_spy_compile
        ):
            rt = install_inference_opt_hook(cfg, mgr, torch)
            assert rt is not None
            try:
                # 1st eval call stays eager (prevents 1-shot eval modules like VAE Decoder from compiling);
                # 2nd static-shape eval call compiles with mode="reduce-overhead".
                model(torch.randn(1, 8))
                self.assertEqual(compile_modes, [])
                model(torch.randn(1, 8))
                self.assertEqual(compile_modes, ["reduce-overhead"])
                self.assertEqual(rt.cuda_graph_compiles, 1)

                # Changing input batch size triggers CUDA Graph dynamic-shape fallback cleanly
                out_dyn = model(torch.randn(3, 8))
                self.assertEqual(out_dyn.shape, (3, 4))
                self.assertIsNone(model._compiled_call_impl)
                self.assertEqual(rt.cuda_graph_fallbacks, 1)
                self.assertTrue(any(ev.pass_name == "inference.cuda_graphs" and ev.status == "FALLBACK" for ev in mgr.events))
            finally:
                uninstall_inference_opt_hook(torch)
                mgr.reset()

    def test_cuda_graphs_skip_embedding_only_and_clone_cuda_outputs(self) -> None:
        from ptopt.passes.inference_opt import _clone_cudagraph_outputs, _is_embedding_only_module

        emb = nn.Sequential(nn.Embedding(16, 8)).eval()
        idx = torch.tensor([1, 2, 3], dtype=torch.long)
        self.assertTrue(_is_embedding_only_module(emb, (idx,), {}, torch))

        cfg = OptimizeConfig(profile="aggressive")
        cfg.inference.inference_mode = True
        cfg.inference.cuda_graphs = True
        mgr = HookManager()
        compile_modes: list[object] = []
        with (
            patch("ptopt.passes.inference_opt._has_cuda_context", return_value=True),
            patch.object(torch, "compile", side_effect=lambda fn, **kw: (compile_modes.append(kw.get("mode")), fn)[1]),
        ):
            install_inference_opt_hook(cfg, mgr, torch)
            try:
                _ = emb(idx)
                _ = emb(idx)
                self.assertEqual(compile_modes, [])
                self.assertTrue(
                    any(
                        ev.pass_name == "inference.cuda_graphs" and ev.status == "SKIPPED" and "embedding" in ev.reason.lower()
                        for ev in mgr.events
                    )
                )
            finally:
                uninstall_inference_opt_hook(torch)
                mgr.reset()

        # Verify CUDA output cloning detaches CUDAGraph output storage and preserves namedtuple & OrderedDict subclasses
        from collections import OrderedDict, namedtuple

        _OutTuple = namedtuple("_OutTuple", ["logits", "aux"])

        class _FakeModelOutput(OrderedDict):
            pass

        orig_out = torch.randn(2, 4)
        named_in = _OutTuple(logits=orig_out, aux=torch.randn(2, 2))
        mo_in = _FakeModelOutput(logits=orig_out, meta="ok")
        with patch.object(torch.Tensor, "is_cuda", new_callable=lambda: property(lambda _s: True)):
            cloned = _clone_cudagraph_outputs((orig_out,), torch, max_bytes=1024)
            self.assertIsNot(cloned[0], orig_out)
            torch.testing.assert_close(cloned[0], orig_out)

            cloned_nt = _clone_cudagraph_outputs(named_in, torch, max_bytes=1024)
            self.assertIsInstance(cloned_nt, _OutTuple)
            self.assertIsNot(cloned_nt.logits, orig_out)
            torch.testing.assert_close(cloned_nt.logits, orig_out)

            cloned_mo = _clone_cudagraph_outputs(mo_in, torch, max_bytes=1024)
            self.assertIsInstance(cloned_mo, _FakeModelOutput)
            self.assertEqual(cloned_mo["meta"], "ok")
            self.assertIsNot(cloned_mo["logits"], orig_out)

            # Large outputs above cap skip cloning
            uncloned = _clone_cudagraph_outputs((orig_out,), torch, max_bytes=4)
            self.assertIs(uncloned[0], orig_out)

        # Verify reduce-overhead skips when regional blocks are already compiled on target_mod
        reg_root = nn.Linear(8, 4).eval()
        reg_root._ptopt_train_regional_active = True
        mgr_reg = HookManager()
        reg_modes: list[object] = []
        with (
            patch("ptopt.passes.inference_opt._has_cuda_context", return_value=True),
            patch.object(torch, "compile", side_effect=lambda fn, **kw: (reg_modes.append(kw.get("mode")), fn)[1]),
        ):
            install_inference_opt_hook(cfg, mgr_reg, torch)
            try:
                _ = reg_root(torch.randn(1, 8))
                _ = reg_root(torch.randn(1, 8))
                self.assertEqual(reg_modes, [])
                self.assertTrue(
                    any(ev.pass_name == "inference.cuda_graphs" and ev.status == "SKIPPED" and "Regional" in ev.reason for ev in mgr_reg.events)
                )
            finally:
                uninstall_inference_opt_hook(torch)
                mgr_reg.reset()

    def test_fp8_static_calib_skips_modules_with_zero_fp8_layers_and_respects_eval_calls(self) -> None:
        patch_embed_like = nn.Sequential(nn.Conv2d(4, 32, kernel_size=2, stride=2)).eval()
        cfg_bal = OptimizeConfig(profile="balanced")
        cfg_bal.amp.enabled = True
        cfg_bal.amp.dtype = "bf16"
        cfg_bal.compile.enabled = True
        cfg_bal.inference.inference_mode = True
        mgr = HookManager()
        with (
            patch.dict(os.environ, {"PTOPT_INFERENCE_FP8": "1", "PTOPT_FP8_STATIC_SCALE": "1"}),
            patch("ptopt.passes.inference_opt.check_fp8_eligibility", return_value=(True, "mock_sm89")),
            patch("ptopt.passes.inference_opt.prewarm_fp8_weight_caches", return_value=0) as mock_prewarm,
        ):
            install_inference_opt_hook(cfg_bal, mgr, torch)
            try:
                _ = patch_embed_like(torch.randn(2, 4, 32, 32))
                _ = patch_embed_like(torch.randn(2, 4, 32, 32))
                mock_prewarm.assert_not_called()
                self.assertFalse(any(ev.pass_name == "inference.fp8_static" for ev in mgr.events))
            finally:
                uninstall_inference_opt_hook(torch)
                mgr.reset()

        # Safe mode (eager): no wasted pre-calib passes on 1st call; balanced (compiled): defers to 2nd eval call
        cfg_safe = OptimizeConfig(profile="safe")
        cfg_safe.inference.inference_mode = True
        lin_safe = nn.Sequential(nn.Linear(128, 128)).eval()
        mgr_safe = HookManager()
        with (
            patch.dict(os.environ, {"PTOPT_INFERENCE_FP8": "1", "PTOPT_FP8_STATIC_SCALE": "1", "PTOPT_FP8_MIN_WORK": "1"}),
            patch("ptopt.passes.inference_opt.check_fp8_eligibility", return_value=(True, "mock_sm89")),
            patch("ptopt.passes.inference_opt._has_cuda_context", return_value=True),
            patch("ptopt.passes.inference_opt._is_cuda_tensor", return_value=True),
            patch("ptopt.passes.inference_opt.prewarm_fp8_weight_caches", return_value=1) as mock_prewarm_safe,
            patch.object(torch, "_scaled_mm", new=lambda x_q, w_t, **kw: x_q.float() @ w_t.float(), create=True),
        ):
            install_inference_opt_hook(cfg_safe, mgr_safe, torch)
            try:
                _ = lin_safe(torch.randn(512, 128))
                _ = lin_safe(torch.randn(512, 128))
                mock_prewarm_safe.assert_not_called()
            finally:
                uninstall_inference_opt_hook(torch)
                mgr_safe.reset()

        lin_comp = nn.Sequential(nn.Linear(128, 128)).eval()
        mgr_comp = HookManager()
        with (
            patch.dict(os.environ, {"PTOPT_INFERENCE_FP8": "1", "PTOPT_FP8_STATIC_SCALE": "1", "PTOPT_FP8_MIN_WORK": "1"}),
            patch("ptopt.passes.inference_opt.check_fp8_eligibility", return_value=(True, "mock_sm89")),
            patch("ptopt.passes.inference_opt._has_cuda_context", return_value=True),
            patch("ptopt.passes.inference_opt._is_cuda_tensor", return_value=True),
            patch("ptopt.passes.inference_opt.prewarm_fp8_weight_caches", return_value=1) as mock_prewarm_comp,
            patch.object(torch, "_scaled_mm", new=lambda x_q, w_t, **kw: x_q.float() @ w_t.float(), create=True),
        ):
            install_inference_opt_hook(cfg_bal, mgr_comp, torch)
            try:
                _ = lin_comp(torch.randn(512, 128))
                mock_prewarm_comp.assert_not_called()
                _ = lin_comp(torch.randn(512, 128))
                mock_prewarm_comp.assert_called_once()
                self.assertTrue(any(ev.pass_name == "inference.fp8_static" and ev.status == "APPLIED" for ev in mgr_comp.events))
            finally:
                uninstall_inference_opt_hook(torch)
                mgr_comp.reset()

    def test_cpu_dynamic_quantization_opt_in(self) -> None:
        cfg = OptimizeConfig(profile="safe")
        cfg.inference.inference_mode = True
        mgr = HookManager()
        model = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 4))
        model.eval()

        with patch.dict(os.environ, {"PTOPT_INFERENCE_QUANT": "int8"}):
            msg = apply_inference_opt_pass(cfg, mgr, torch_mod=torch)
            self.assertIn("Inference opt hook installed", msg)
            try:
                out = model(torch.randn(2, 16))
                self.assertEqual(out.shape, (2, 4))
                self.assertTrue(any(ev.pass_name == "inference.quant" and ev.status == "APPLIED" for ev in mgr.events))

                # TransformerEncoderLayer fastpath fails under dynamic Linear quantization -> falls back cleanly
                enc = nn.TransformerEncoderLayer(d_model=16, nhead=2, dim_feedforward=32, batch_first=True)
                enc.eval()
                enc_out = enc(torch.randn(2, 4, 16))
                self.assertEqual(enc_out.shape, (2, 4, 16))
                self.assertIsNone(getattr(enc, "_ptopt_quantized_mod", None))
                self.assertTrue(any(ev.pass_name == "inference.quant" and ev.status == "FALLBACK" for ev in mgr.events))
            finally:
                uninstall_inference_opt_hook(torch)
                mgr.reset()

    def test_fp8_prequant_cache_preserves_model_parameters_and_state_dict(self) -> None:
        from ptopt.passes.inference_opt import apply_fp8_linear_eval_hooks

        if not hasattr(torch, "float8_e4m3fn"):
            self.skipTest("float8_e4m3fn unavailable in this PyTorch build")

        model = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 64))
        orig_sd = {k: v.clone() for k, v in model.state_dict().items()}
        orig_param_ids = {name: id(p) for name, p in model.named_parameters()}

        # Verify small-M eval skips FP8 weight caching (no VRAM overhead on bs=1) and state_dict stays intact
        for m in model.modules():
            if isinstance(m, nn.Linear):
                with patch.object(torch.Tensor, "is_cuda", new=True):
                    n_patched = apply_fp8_linear_eval_hooks(m, torch)
                    self.assertEqual(n_patched, 1)
                    m.eval()
                    _ = m(torch.randn(4, 128))
                    self.assertIsNone(getattr(m, "_ptopt_fp8_w_cache", None))
                break

        after_sd = model.state_dict()
        self.assertEqual(list(after_sd.keys()), list(orig_sd.keys()))
        for name, param in model.named_parameters():
            self.assertEqual(id(param), orig_param_ids[name])
            self.assertEqual(param.dtype, torch.float32)
            torch.testing.assert_close(after_sd[name], orig_sd[name])

    def test_fp8_compilable_traceable_path_zero_graph_breaks(self) -> None:
        from ptopt.passes.inference_opt import (
            _quantize_fp8_traceable,
            apply_fp8_linear_eval_hooks,
            prewarm_fp8_weight_caches,
        )

        if not hasattr(torch, "float8_e4m3fn"):
            self.skipTest("float8_e4m3fn unavailable in this PyTorch build")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # 1. Verify _quantize_fp8_traceable compiles under fullgraph=True with 0 graph breaks
        compiled_quant = torch.compile(
            lambda t: _quantize_fp8_traceable(t, torch.float8_e4m3fn, torch.float32),
            fullgraph=True,
        )
        x_sample = torch.randn(512, 128, device=device, dtype=torch.bfloat16)
        q_c, s_c = compiled_quant(x_sample)
        q_e, s_e = _quantize_fp8_traceable(x_sample, torch.float8_e4m3fn, torch.float32)
        self.assertEqual(q_c.dtype, torch.float8_e4m3fn)
        self.assertEqual(s_c.dtype, torch.float32)
        torch.testing.assert_close(s_c, s_e)
        # Allow 1-ULP E4M3 (3-bit mantissa = 12.5%) rounding differences under Inductor fastmath
        torch.testing.assert_close(q_c.float(), q_e.float(), rtol=0.15, atol=1e-2)

        # 2. Verify full patched Linear module compiles under fullgraph=True without graph breaks
        class _Fp8Block(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin = nn.Linear(512, 512, bias=True)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.lin(torch.nn.functional.gelu(x))

        mod = _Fp8Block().to(device=device, dtype=torch.bfloat16).eval()
        orig_lin = nn.Linear(512, 512, bias=True).to(device=device, dtype=torch.bfloat16).eval()
        orig_lin.load_state_dict(mod.lin.state_dict())

        def _fake_scaled_mm(
            a: torch.Tensor,
            b: torch.Tensor,
            *,
            scale_a: torch.Tensor,
            scale_b: torch.Tensor,
            out_dtype: torch.dtype,
            bias: torch.Tensor | None = None,
        ) -> torch.Tensor:
            out = (a.float() @ b.float()) * (scale_a.float() * scale_b.float())
            if bias is not None:
                out = out + bias.float()
            return out.to(out_dtype)

        if device == "cuda":
            n_patched = apply_fp8_linear_eval_hooks(mod, torch, min_work=512 * 512)
            self.assertEqual(n_patched, 1)
            prewarm_fp8_weight_caches(mod, torch, min_work=512 * 512)
            self.assertIsNotNone(getattr(mod.lin, "_ptopt_fp8_w_cache", None))
            compiled_mod = torch.compile(mod, fullgraph=True)
            with torch.inference_mode():
                x_in = torch.randn(512, 512, device=device, dtype=torch.bfloat16)
                out = compiled_mod(x_in)
                ref = orig_lin(torch.nn.functional.gelu(x_in))
            self.assertEqual(out.shape, ref.shape)
            rel_err = (out.float() - ref.float()).abs().max() / ref.float().abs().max().clamp(min=1e-6)
            self.assertLess(float(rel_err.item()), 0.05)
        else:
            with patch("ptopt.passes.inference_opt._is_cuda_tensor", new=lambda _t: True), patch.object(
                torch, "_scaled_mm", new=_fake_scaled_mm, create=True
            ):
                n_patched = apply_fp8_linear_eval_hooks(mod, torch, min_work=512 * 512)
                self.assertEqual(n_patched, 1)
                prewarm_fp8_weight_caches(mod, torch, min_work=512 * 512)
                self.assertIsNotNone(getattr(mod.lin, "_ptopt_fp8_w_cache", None))
                w_t_cached = mod.lin._ptopt_fp8_w_cache[1]
                self.assertEqual(w_t_cached.stride(), (1, 512))
                compiled_mod = torch.compile(mod, fullgraph=True)
                with torch.inference_mode():
                    x_in = torch.randn(512, 512, dtype=torch.bfloat16)
                    out = compiled_mod(x_in)
                    ref = orig_lin(torch.nn.functional.gelu(x_in))
                max_diff = float((out.float() - ref.float()).abs().max().item())
                self.assertGreater(max_diff, 0.0)
                rel_err = max_diff / float(ref.float().abs().max().clamp(min=1e-6).item())
                self.assertLess(rel_err, 0.05)

    def test_fp8_min_work_gate_and_opt_in_weight_drop(self) -> None:
        from ptopt.passes.inference_opt import apply_fp8_linear_eval_hooks, prewarm_fp8_weight_caches

        if not hasattr(torch, "float8_e4m3fn"):
            self.skipTest("float8_e4m3fn unavailable in this PyTorch build")

        small = nn.Linear(128, 128).eval()
        large = nn.Linear(512, 512).eval()
        with patch.object(torch.Tensor, "is_cuda", new=True):
            apply_fp8_linear_eval_hooks(small, torch, min_work=512 * 512)
            apply_fp8_linear_eval_hooks(large, torch, min_work=512 * 512)
            # Small layer skips prewarm due to min_work gate; large layer builds strict col-major cache
            built_small = prewarm_fp8_weight_caches(small, torch, min_work=512 * 512)
            with patch.dict(os.environ, {"PTOPT_FP8_DROP_BF16_EVAL": "0"}):
                built_large = prewarm_fp8_weight_caches(large, torch, min_work=512 * 512)
            self.assertEqual(built_small, 0)
            self.assertEqual(built_large, 1)
            self.assertEqual(large.weight.numel(), 512 * 512)

            # Opt-in PTOPT_FP8_DROP_BF16_EVAL=1 clears source weight storage once cached
            large2 = nn.Linear(512, 512).eval()
            apply_fp8_linear_eval_hooks(large2, torch, min_work=512 * 512)
            with patch.dict(os.environ, {"PTOPT_FP8_DROP_BF16_EVAL": "1"}):
                built_drop = prewarm_fp8_weight_caches(large2, torch, min_work=512 * 512)
            self.assertEqual(built_drop, 1)
            self.assertEqual(large2.weight.numel(), 0)
            self.assertIsNotNone(getattr(large2, "_ptopt_fp8_w_cache", None))

            # TransformerEncoderLayer's MHA out_proj accesses weight directly in F.multi_head_attention_forward;
            # verify out_proj & in_proj_weight are preserved cleanly while linear1/linear2 drop and self_attn is FP8-patched.
            enc = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=1024, batch_first=True).eval()
            n_enc = apply_fp8_linear_eval_hooks(enc, torch, min_work=512 * 512)
            self.assertEqual(n_enc, 3)
            self.assertTrue(getattr(enc.self_attn, "_ptopt_fp8_mha_patched", False))
            with patch.dict(os.environ, {"PTOPT_FP8_DROP_BF16_EVAL": "1"}):
                built_enc = prewarm_fp8_weight_caches(enc, torch, min_work=512 * 512)
            self.assertEqual(built_enc, 4)
            self.assertIsNotNone(getattr(enc.self_attn, "_ptopt_fp8_inproj_cache", None))
            self.assertIsNotNone(getattr(enc.self_attn.out_proj, "_ptopt_fp8_w_cache", None))
            self.assertEqual(enc.self_attn.out_proj.weight.numel(), 512 * 512)
            self.assertEqual(enc.self_attn.in_proj_weight.numel(), 3 * 512 * 512)
            self.assertEqual(enc.linear1.weight.numel(), 0)
            self.assertEqual(enc.linear2.weight.numel(), 0)

    def test_nested_dict_requires_grad_skips_inference_mode(self) -> None:
        cfg = OptimizeConfig(profile="safe")
        cfg.inference.inference_mode = True
        mgr = HookManager()
        rt = install_inference_opt_hook(cfg, mgr, torch)
        assert rt is not None

        class _DictInferNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin = nn.Linear(6, 2)

            def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
                return self.lin(batch["x"])

        model = _DictInferNet().eval()
        try:
            x_req = torch.randn(2, 6, requires_grad=True)
            out = model({"x": x_req})
            self.assertTrue(out.requires_grad)
            out.sum().backward()
            self.assertIsNotNone(x_req.grad)
            self.assertEqual(rt.grad_guard_skips, 1)
        finally:
            uninstall_inference_opt_hook(torch)
            mgr.reset()


if __name__ == "__main__":
    unittest.main()
