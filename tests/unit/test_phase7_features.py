"""Comprehensive unit & integration tests for Phase 7 hardening, calibration, and namespace isolation."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ptopt.cli.main import parse_ptopt_argv  # noqa: E402
from ptopt.cli.overrides import parse_override_items  # noqa: E402
from ptopt.config import ConfigResolver, OptimizeConfig  # noqa: E402
from ptopt.detect import collect_fingerprint  # noqa: E402
from ptopt.detect.fingerprint import _fingerprint_cache_path  # noqa: E402
from ptopt.hooks.checkpoint_compat import (  # noqa: E402
    install_checkpoint_compat_hooks,
    sync_optimizer_state_memory_format,
    uninstall_checkpoint_compat_hooks,
)
from ptopt.hooks.manager import HookManager  # noqa: E402
from ptopt.hooks.module_dispatch import (  # noqa: E402
    get_module_call_dispatcher,
    register_module_call_handler,
    unregister_module_call_handler,
)
from ptopt.hooks.telemetry_hook import TelemetryCollector  # noqa: E402
from ptopt.kernels.fused_linear_cross_entropy import LazyLinearLogits  # noqa: E402
from ptopt.kernels.registry import _patch_module_submodules  # noqa: E402
from ptopt.kernels.rope_or_swiglu import ref_rope  # noqa: E402
from ptopt.passes.activation_checkpoint import (  # noqa: E402
    install_activation_checkpoint_hook,
    uninstall_activation_checkpoint_hook,
)
from ptopt.passes.auto_amp import (  # noqa: E402
    _OPTIM_STEP_CLASSES,
    install_auto_amp_hook,
    uninstall_auto_amp_hook,
)
from ptopt.passes.auto_compile import (  # noqa: E402
    install_auto_compile_hook,
    uninstall_auto_compile_hook,
)
from ptopt.passes.channels_last import convert_module_channels_last  # noqa: E402
from ptopt.passes.inference_opt import (  # noqa: E402
    _update_fp8_static_scale,
    install_inference_opt_hook,
    uninstall_inference_opt_hook,
)


class _MixedConv2d3d(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c2 = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.c3 = nn.Conv3d(2, 4, kernel_size=3, padding=1)

    def forward(self, x2: torch.Tensor, x3: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.c2(x2), self.c3(x3)


class _TwoBlockNet(nn.Module):
    def __init__(self, dim: int = 32) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim)),
            nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim)),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = b(x)
        return x


class TestPhase7Features(unittest.TestCase):
    """Covers every High, Medium, and Low fix from Phase 7."""

    def test_user_project_with_local_src_package_runs_under_ptopt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proj = Path(tmp)
            pkg = proj / "src"
            pkg.mkdir(parents=True, exist_ok=True)
            (pkg / "__init__.py").write_text("", encoding="utf-8")
            (pkg / "model.py").write_text(
                "import torch.nn as nn\n"
                "class TinyUserNet(nn.Module):\n"
                "    def __init__(self):\n"
                "        super().__init__()\n"
                "        self.fc = nn.Linear(8, 4)\n"
                "    def forward(self, x):\n"
                "        return self.fc(x)\n",
                encoding="utf-8",
            )
            (proj / "train.py").write_text(
                "import torch\n"
                "from src.model import TinyUserNet\n"
                "m = TinyUserNet()\n"
                "y = m(torch.randn(2, 8))\n"
                "print('USER_SRC_OK', tuple(y.shape))\n",
                encoding="utf-8",
            )
            env = dict(os.environ)
            old_pp = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{old_pp}" if old_pp else str(REPO_ROOT)
            proc = subprocess.run(
                [sys.executable, "-m", "ptopt.cli.main", "--profile", "safe", "--", sys.executable, "train.py"],
                cwd=str(proj),
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}")
            self.assertIn("USER_SRC_OK (2, 4)", proc.stdout)

    def test_fp8_static_calibration_no_phantom_steps_and_respects_calib_steps(self) -> None:
        class _FourSubmoduleModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.in_proj = nn.Linear(128, 128)
                self.block1 = nn.Linear(128, 128)
                self.norm = nn.LayerNorm(128)
                self.head = nn.Linear(128, 128)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.head(self.norm(self.block1(self.in_proj(x))))

        model = _FourSubmoduleModel().eval()
        # Fake FP8 Linear calibration counter verification on CPU via _update_fp8_static_scale
        probe_mod = SimpleNamespace()
        _update_fp8_static_scale(probe_mod, torch.tensor(0.5), torch.float32, calib_steps=3)
        self.assertFalse(getattr(probe_mod, "_ptopt_fp8_static_ready", False))
        _update_fp8_static_scale(probe_mod, torch.tensor(0.6), torch.float32, calib_steps=3)
        self.assertFalse(getattr(probe_mod, "_ptopt_fp8_static_ready", False))
        _update_fp8_static_scale(probe_mod, torch.tensor(0.4), torch.float32, calib_steps=3)
        self.assertTrue(getattr(probe_mod, "_ptopt_fp8_static_ready", False))

        with tempfile.TemporaryDirectory() as tmp:
            tel_path = Path(tmp) / "tel.json"
            mgr = HookManager()
            cfg = OptimizeConfig(profile="balanced")
            cfg.amp.enabled = True
            cfg.amp.dtype = "bf16"
            cfg.compile.enabled = True

            # Verify TelemetryCollector sits outer to ModuleCallDispatcher when applied via HookManager
            # and suppress() shields internal call_impl steps regardless of hook ordering.
            collector = TelemetryCollector(tel_path, mgr, debug_profile=True, torch_profile=False)
            with patch.dict(
                os.environ,
                {
                    "PTOPT_INFERENCE_FP8": "1",
                    "PTOPT_FP8_STATIC_SCALE": "1",
                    "PTOPT_FP8_CALIB_STEPS": "3",
                    "PTOPT_FP8_MIN_WORK": "1",
                },
            ):
                rt = install_inference_opt_hook(cfg, mgr, torch)
                # Deliberately install telemetry INNER first too, proving suppress() prevents fake s0..s3 steps
                collector.install(torch)
                try:
                    with collector.suppress():
                        for _ in range(3):
                            _ = model._call_impl(torch.randn(32, 128))
                    for _ in range(5):
                        with torch.no_grad():
                            _ = model(torch.randn(32, 128))
                    self.assertEqual(len(collector.step_times_ms), 5)
                    self.assertEqual(len(collector.debug_steps), 5)
                finally:
                    collector.uninstall()
                    if rt is not None:
                        uninstall_inference_opt_hook(torch)

        # Also verify HookManager.apply_torch_hooks() registers TelemetryCollector OUTER to ModuleCallDispatcher
        # and runs static FP8 multi-step calibration without phantom submodule steps.
        with tempfile.TemporaryDirectory() as tmp2:
            env_patch = {
                "PTOPT_TELEMETRY_OUT": str(Path(tmp2) / "tel_hm.json"),
                "PTOPT_DEBUG_PROFILE": "1",
                "PTOPT_INFERENCE_FP8": "1",
                "PTOPT_FP8_STATIC_SCALE": "1",
                "PTOPT_FP8_CALIB_STEPS": "2",
                "PTOPT_FP8_MIN_WORK": "1",
            }
            model2 = _FourSubmoduleModel().eval()
            mgr2 = HookManager()
            mgr2.config = cfg
            mgr2.register_pass("inference.opt", lambda c, m: (install_inference_opt_hook(c, m, torch), "applied")[1])
            fp8_dt = getattr(torch, "float8_e4m3fn", None)

            def _fake_scaled_mm(x_q: torch.Tensor, w_t: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor, out_dtype: torch.dtype, bias: torch.Tensor | None = None) -> torch.Tensor:
                res = (x_q.to(torch.float32) * scale_a) @ (w_t.to(torch.float32) * scale_b)
                if bias is not None:
                    res = res + bias.to(torch.float32)
                return res.to(out_dtype)

            with (
                patch.dict(os.environ, env_patch),
                patch("ptopt.passes.inference_opt.check_fp8_eligibility", return_value=(fp8_dt is not None, "mock_sm89")),
                patch("ptopt.passes.inference_opt._has_cuda_context", return_value=True),
                patch("ptopt.passes.inference_opt._is_cuda_tensor", return_value=True),
                patch("ptopt.passes.inference_opt.prewarm_fp8_weight_caches", return_value=1),
                patch.object(torch, "_scaled_mm", _fake_scaled_mm, create=True),
            ):
                mgr2.apply_torch_hooks(torch)
                hm_collector = getattr(mgr2, "_telemetry_collector", None)
                self.assertIsNotNone(hm_collector)
                try:
                    assert hm_collector is not None
                    for _ in range(4):
                        with torch.no_grad():
                            _ = model2(torch.randn(512, 128))
                    self.assertEqual(len(hm_collector.step_times_ms), 4)
                    self.assertEqual(len(hm_collector.debug_steps), 4)
                    if fp8_dt is not None:
                        self.assertEqual(getattr(model2.in_proj, "_ptopt_fp8_calib_seen", 0), 2)
                        self.assertTrue(getattr(model2.in_proj, "_ptopt_fp8_static_ready", False))
                finally:
                    if hm_collector is not None:
                        hm_collector.uninstall()
                    uninstall_inference_opt_hook(torch)

    def test_hybrid_lce_skips_wide_intermediate_ffn_layer(self) -> None:
        class _FakeLinear:
            def __init__(self, in_f: int, out_f: int) -> None:
                self.in_features = in_f
                self.out_features = out_f
                self.weight = SimpleNamespace(is_cuda=True, dtype=torch.bfloat16)
                self.bias = None

            def forward(self, x: object) -> object:
                return x

        class _WideFFNTransformer:
            def __init__(self) -> None:
                # 8x intermediate FFN expansion (>= min_lce_vocab) must NOT be classified as LCE head
                self.ffn_gate = _FakeLinear(512, 8192)
                self.ffn_up = _FakeLinear(512, 8192)
                self.ffn_down = _FakeLinear(8192, 512)
                self.lm_head = _FakeLinear(512, 32000)

            def modules(self) -> list[object]:
                return [self, self.ffn_gate, self.ffn_up, self.ffn_down, self.lm_head]

        fake_torch = SimpleNamespace(
            bfloat16=torch.bfloat16,
            float16=torch.float16,
            float32=torch.float32,
            nn=SimpleNamespace(Linear=_FakeLinear),
            compiler=SimpleNamespace(disable=lambda fn: fn),
        )
        m = _WideFFNTransformer()
        mgr = HookManager()
        patched = _patch_module_submodules(
            m,
            fake_torch,
            {"fused_linear_cross_entropy": True},
            mgr,
            min_lce_vocab=8192,
            fallback_on_error=True,
            hybrid_compile_lce=True,
        )
        self.assertEqual(patched, 1)
        self.assertFalse(getattr(m.ffn_gate, "_ptopt_kern_patched", False))
        self.assertFalse(getattr(m.ffn_up, "_ptopt_kern_patched", False))
        self.assertFalse(getattr(m.ffn_down, "_ptopt_kern_patched", False))
        self.assertTrue(getattr(m.lm_head, "_ptopt_kern_patched", False))

    def test_lazy_linear_logits_inside_compiled_forward_no_dynamo_error(self) -> None:
        h = torch.randn(2, 4, 16, requires_grad=True)
        w = torch.randn(32, 16, requires_grad=True)
        lazy = LazyLinearLogits(h, w, None, (2, 4, 32), torch.float32)
        self.assertFalse(lazy._is_view())
        self.assertEqual(lazy.device, h.device)
        self.assertLessEqual(lazy.untyped_storage().nbytes(), 4)

        class _LossInsideForwardLM(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.proj = nn.Linear(16, 16)
                self.head = nn.Linear(16, 32)

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                hid = torch.relu(self.proj(x))
                logits = self.head(hid)
                return F.cross_entropy(logits.view(-1, 32), y.view(-1))

        torch._dynamo.reset()
        prev_suppress = torch._dynamo.config.suppress_errors
        torch._dynamo.config.suppress_errors = False
        ref_m = _LossInsideForwardLM()
        test_m = _LossInsideForwardLM()
        test_m.load_state_dict(ref_m.state_dict())

        @torch.compiler.disable
        def _dis_head(inp: torch.Tensor) -> LazyLinearLogits:
            return LazyLinearLogits(inp, test_m.head.weight, test_m.head.bias, (*inp.shape[:-1], 32), inp.dtype)

        test_m.head.forward = _dis_head  # type: ignore[method-assign]
        orig_ce = F.cross_entropy
        ce_hits: list[tuple[int, ...]] = []

        def _hook_ce(inp: torch.Tensor, target: torch.Tensor, **kw: object) -> torch.Tensor:
            if hasattr(inp, "_ptopt_lazy_linear"):
                ce_hits.append(tuple(inp.shape))
                return orig_ce(inp.materialize(), target, **kw)
            return orig_ce(inp, target, **kw)

        F.cross_entropy = _hook_ce  # type: ignore[assignment]
        try:
            x = torch.randn(2, 4, 16)
            y = torch.randint(0, 32, (2, 4))
            ref_loss = ref_m(x, y)
            ref_loss.backward()

            compiled = torch.compile(test_m, backend="eager")
            loss = compiled(x, y)
            loss.backward()
            self.assertGreaterEqual(len(ce_hits), 1)
            self.assertAlmostEqual(float(loss.detach()), float(ref_loss.detach()), places=5)
            self.assertIsNotNone(test_m.head.weight.grad)
            self.assertIsNotNone(test_m.proj.weight.grad)
            self.assertTrue(torch.allclose(test_m.head.weight.grad, ref_m.head.weight.grad, atol=1e-5))
        finally:
            F.cross_entropy = orig_ce  # type: ignore[assignment]
            torch._dynamo.config.suppress_errors = prev_suppress
            torch._dynamo.reset()

    def test_auto_compile_defers_oom_to_activation_checkpoint_rescue(self) -> None:
        mgr = HookManager()
        cfg = OptimizeConfig(profile="balanced")
        cfg.runtime.activation_checkpointing = "auto"
        cfg.compile.enabled = True
        cfg.amp.enabled = False
        model = _TwoBlockNet(32)

        calls = {"n": 0}
        orig_fwd = model.forward

        def _oom_first_fwd(x: torch.Tensor) -> torch.Tensor:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("CUDA out of memory. Tried to allocate 256.00 MiB")
            return orig_fwd(x)

        model.forward = _oom_first_fwd  # type: ignore[method-assign]
        act_rt = install_activation_checkpoint_hook(cfg, mgr, torch, vram_probe_fn=lambda: 0.20)
        comp_rt = install_auto_compile_hook(cfg, mgr, torch)
        try:
            with patch.dict(os.environ, {"PTOPT_ACT_CKPT_MIN_PARAMS": "64"}):
                out = model(torch.randn(2, 32))
            self.assertEqual(tuple(out.shape), (2, 32))
            self.assertTrue(getattr(model, "_ptopt_act_ckpt_active", False))
            self.assertFalse(getattr(model, "_ptopt_compile_failed", False))
            self.assertEqual(calls["n"], 2)
        finally:
            uninstall_auto_compile_hook(torch)
            uninstall_activation_checkpoint_hook(torch)
            self.assertIsNotNone(act_rt)
            self.assertIsNotNone(comp_rt)

    def test_activation_checkpoint_resets_peak_memory_stats_on_partial(self) -> None:
        mgr = HookManager()
        cfg = OptimizeConfig(profile="balanced")
        cfg.runtime.activation_checkpointing = "always"
        model = _TwoBlockNet(32)
        reset_calls: list[str] = []

        orig_cuda = torch.cuda
        fake_cuda = SimpleNamespace(
            is_available=lambda: True,
            reset_peak_memory_stats=lambda: reset_calls.append("reset"),
            empty_cache=lambda: None,
            OutOfMemoryError=getattr(orig_cuda, "OutOfMemoryError", RuntimeError),
        )
        with (
            patch.dict(os.environ, {"PTOPT_ACT_CKPT_RATIO": "0.5", "PTOPT_ACT_CKPT_MIN_PARAMS": "64"}),
            patch.object(torch, "cuda", fake_cuda),
        ):
            install_activation_checkpoint_hook(cfg, mgr, torch, vram_probe_fn=lambda: 0.90)
            try:
                _ = model(torch.randn(2, 32))
                self.assertEqual(len(reset_calls), 1)
            finally:
                uninstall_activation_checkpoint_hook(torch)

    def test_checkpoint_compat_injected_fused_and_mixed_conv2d_conv3d(self) -> None:
        install_checkpoint_compat_hooks(torch)
        try:
            # 1) Mixed Conv2d + Conv3d conversion and submodule state_dict contiguous save
            model = _MixedConv2d3d()
            applied = convert_module_channels_last(model, torch)
            self.assertTrue(applied)
            self.assertTrue(model.c2.weight.is_contiguous(memory_format=torch.channels_last))
            self.assertTrue(model.c3.weight.is_contiguous(memory_format=torch.channels_last_3d))

            opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
            x2 = torch.randn(2, 3, 8, 8)
            x3 = torch.randn(2, 2, 4, 6, 6)
            y2, y3 = model(x2, x3)
            loss = y2.sum() + y3.sum()
            loss.backward()
            # Mutate optimizer state buffers into mismatching contiguous strides to verify 4D + 5D alignment
            opt.step()
            for p in model.parameters():
                if p.dim() in (4, 5):
                    opt.state[p]["momentum_buffer"] = opt.state[p]["momentum_buffer"].contiguous()
            aligned = sync_optimizer_state_memory_format(opt, torch)
            self.assertEqual(aligned, 2)
            self.assertEqual(opt.state[model.c2.weight]["momentum_buffer"].stride(), model.c2.weight.stride())
            self.assertEqual(opt.state[model.c3.weight]["momentum_buffer"].stride(), model.c3.weight.stride())

            # Submodule AND parent state_dict must serialize strictly contiguous tensors
            parent_sd = model.state_dict()
            sub_sd = model.c3.state_dict()
            self.assertTrue(parent_sd["c2.weight"].is_contiguous())
            self.assertTrue(parent_sd["c3.weight"].is_contiguous())
            self.assertTrue(sub_sd["weight"].is_contiguous())

            # 2) injected_fused moves fake CUDA step tensor to CPU on state_dict()
            class _FakeCudaStep:
                is_cuda = True

                def cpu(self) -> torch.Tensor:
                    return torch.tensor(5.0)

            p = nn.Parameter(torch.randn(4, 4))
            opt2 = torch.optim.SGD([p], lr=0.01)
            opt2._ptopt_injected_fused = True  # type: ignore[attr-defined]
            opt2.param_groups[0]["fused"] = True
            opt2.state[p]["step"] = _FakeCudaStep()
            clean_opt_sd = opt2.state_dict()
            first_state = next(iter(clean_opt_sd["state"].values()))
            self.assertIsInstance(first_state["step"], torch.Tensor)
            self.assertFalse(clean_opt_sd["param_groups"][0]["fused"])
        finally:
            uninstall_checkpoint_compat_hooks(torch)

    def test_auto_amp_nadam_radam_eval_isolation_and_watchdog_queue(self) -> None:
        self.assertIn("NAdam", _OPTIM_STEP_CLASSES)
        self.assertIn("RAdam", _OPTIM_STEP_CLASSES)

        mgr = HookManager()
        cfg = OptimizeConfig(profile="balanced")
        cfg.amp.enabled = True
        cfg.amp.dtype = "fp16"
        cfg.amp.grad_scaler = True
        rt = install_auto_amp_hook(cfg, mgr, torch)
        self.assertIsNotNone(rt)
        assert rt is not None
        try:
            # 1) NAdam step calls flush_pending_stability_check & trips cleanly if nan queued
            p = nn.Parameter(torch.tensor([1.0, 2.0]))
            opt = torch.optim.NAdam([p], lr=0.01)
            self.assertFalse(rt.check_loss_stability(torch.tensor(float("nan"))))
            self.assertTrue(rt.disabled_by_watchdog)
            rt.disabled_by_watchdog = False

            # 2) Async watchdog queue does not drop an older still-pending slot when overwritten
            class _FakeEvent:
                def __init__(self, complete: bool = False) -> None:
                    self.complete = complete

                def record(self) -> None:
                    pass

                def query(self) -> bool:
                    return self.complete

                def synchronize(self) -> None:
                    self.complete = True

            class _FakeLoss(torch.Tensor):
                pass

            fake_cuda_loss_nan = torch.tensor(float("nan"))
            fake_cuda_loss_ok = torch.tensor(1.5)
            fake_events = [_FakeEvent(False), _FakeEvent(False)]
            fake_cuda_ns = SimpleNamespace(Event=lambda: fake_events.pop(0) if fake_events else _FakeEvent(False))
            with (
                patch.object(rt, "torch", SimpleNamespace(cuda=fake_cuda_ns, empty=torch.empty, float32=torch.float32)),
                patch.object(torch.Tensor, "is_cuda", new_callable=lambda: property(lambda _s: True)),
            ):
                # Enqueue NaN into slot 0 (event stays pending), then enqueue 1.5 into slot 1, then trigger 3rd write -> slot 0 MUST drain NaN and trip!
                self.assertTrue(rt.enqueue_loss_stability_check(fake_cuda_loss_nan))
                self.assertTrue(rt.enqueue_loss_stability_check(fake_cuda_loss_ok))
                ok_third = rt.enqueue_loss_stability_check(fake_cuda_loss_ok)
                self.assertFalse(ok_third)
                self.assertTrue(rt.disabled_by_watchdog)

            # 3) Eval forward must not clobber _last_forward_skipped_amp for training step
            rt.disabled_by_watchdog = False
            rt._last_forward_skipped_amp = False
            eval_tiny = nn.Linear(4, 4).eval()
            with torch.no_grad():
                _ = eval_tiny(torch.randn(2, 4))
            self.assertFalse(rt._last_forward_skipped_amp)

            # Verify RAdam wraps step too
            p2 = nn.Parameter(torch.tensor([1.0]))
            opt2 = torch.optim.RAdam([p2], lr=0.01)
            (p2.sum()).backward()
            opt2.step()
        finally:
            uninstall_auto_amp_hook(torch)

    def test_module_dispatcher_thread_local_depth(self) -> None:
        observed_depths: list[int] = []
        barrier = threading.Barrier(2)
        m = nn.Linear(4, 4)
        disp = get_module_call_dispatcher(torch, create=True)
        assert disp is not None

        def _handler(mod_self: object, args: tuple[object, ...], kwargs: dict[str, object], next_fn: object) -> object:
            observed_depths.append(disp._depth)
            barrier.wait(timeout=3.0)
            return next_fn(mod_self, *args, **kwargs)  # type: ignore[operator]

        register_module_call_handler(torch, "test.tls", 10, _handler)
        try:
            errs: list[Exception] = []

            def _worker() -> None:
                try:
                    _ = m(torch.randn(2, 4))
                except Exception as exc:  # noqa: BLE001
                    errs.append(exc)

            t1 = threading.Thread(target=_worker)
            t2 = threading.Thread(target=_worker)
            t1.start()
            t2.start()
            t1.join(timeout=5.0)
            t2.join(timeout=5.0)
            self.assertFalse(errs, f"Thread error: {errs}")
            self.assertEqual(observed_depths, [1, 1])
            self.assertEqual(disp._depth, 0)
        finally:
            unregister_module_call_handler(torch, "test.tls")

    def test_rope_3d_broadcast_and_cli_allow_list_and_fp_cache_key(self) -> None:
        # 1) RoPE 3D cos/sin (1, S, D) & (B, S, D) against 4D (B, H, S, D)
        q = torch.randn(2, 4, 6, 8)
        k = torch.randn(2, 4, 6, 8)
        cos3 = torch.randn(2, 6, 8)
        sin3 = torch.randn(2, 6, 8)
        qo, ko = ref_rope(q, k, cos3, sin3)
        q_exp, k_exp = ref_rope(q, k, cos3.unsqueeze(1), sin3.unsqueeze(1))
        self.assertTrue(torch.allclose(qo, q_exp))
        self.assertTrue(torch.allclose(ko, k_exp))

        # 2) Comma-separated list overrides (kernels.allow_list)
        parsed = parse_override_items(["kernels.allow_list=rmsnorm, layernorm", "compile.enabled=false"])
        self.assertEqual(parsed["kernels.allow_list"], ["rmsnorm", "layernorm"])
        self.assertIs(parsed["compile.enabled"], False)
        cli = parse_ptopt_argv(["--override", "kernels.allow_list=[rmsnorm,cross_entropy]", "--", "python3", "x.py"])
        self.assertEqual(cli.ptopt_args.override, ["kernels.allow_list=[rmsnorm,cross_entropy]"])
        res = ConfigResolver().resolve(
            collect_fingerprint(),
            overrides=parse_override_items(cli.ptopt_args.override),
        )
        self.assertEqual(res.config.kernels.allow_list, ["rmsnorm", "cross_entropy"])

        # 3) Consistent fingerprint cache path with torch_mod=None vs torch_mod=torch
        p_none = _fingerprint_cache_path(None, os.environ)
        p_torch = _fingerprint_cache_path(torch, os.environ)
        self.assertEqual(p_none, p_torch)


if __name__ == "__main__":
    unittest.main()
