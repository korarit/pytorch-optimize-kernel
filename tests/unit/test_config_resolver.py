"""Unit tests for Phase 1.2: Config Schema v1 and Layered Rule Engine."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ptopt.config import ConfigResolver, ConfigValidationError, OptimizeConfig, validate_config
from ptopt.detect import GPUDeviceInfo, GPUProbeResult, HardwareFingerprint, HostInfo


def _make_fp(
    *,
    backend: str = "cuda",
    os_name: str = "Linux",
    cpu_arch: str = "x86_64",
    physical_cores: int | None = 6,
    logical_cores: int | None = 12,
    shm_total_mb: float | None = 26155.0,
    is_colab: bool = True,
    triton_version: str | None = "3.8.0",
    jemalloc_path: str | None = None,
    gpu_name: str = "NVIDIA L4",
    major: int = 8,
    minor: int = 9,
    vram_mb: float = 22563.1,
    sm_count: int = 58,
) -> HardwareFingerprint:
    host = HostInfo(
        hostname="test-host",
        os_name=os_name,
        os_release="6.6.0",
        cpu_arch=cpu_arch,
        cpu_model="Test CPU",
        physical_cores=physical_cores,
        logical_cores=logical_cores,
        isa_flags=["avx2"],
        total_ram_mb=32768.0,
        available_ram_mb=24000.0,
        shm_total_mb=shm_total_mb,
        shm_free_mb=shm_total_mb,
        is_colab=is_colab,
        python_version="3.13.0",
        torch_version="2.14.1+cu130" if backend == "cuda" else "2.15.0+cpu",
        cuda_version="13.0" if backend == "cuda" else None,
        triton_version=triton_version,
        jemalloc_path=jemalloc_path,
    )
    if backend == "cuda":
        sm_int = major * 10 + minor
        dev = GPUDeviceInfo(
            index=0,
            name=gpu_name,
            compute_capability=f"{major}.{minor}",
            major=major,
            minor=minor,
            sm_int=sm_int,
            sm_name=f"sm_{sm_int}",
            total_vram_mb=vram_mb,
            sm_count=sm_count,
            bf16_supported=sm_int >= 80,
            tf32_supported=sm_int >= 80,
            fp8_supported=sm_int >= 89,
            flash_sdp_supported=sm_int >= 80,
            mem_efficient_sdp_supported=sm_int >= 70,
        )
        gpu = GPUProbeResult(backend="cuda", device_count=1, selected_index=0, cuda_visible_devices=None, devices=[dev])
    else:
        gpu = GPUProbeResult(backend=backend, device_count=0, selected_index=0, cuda_visible_devices=None, devices=[])
    return HardwareFingerprint(host=host, gpu=gpu)


class TestConfigResolverMatrix(unittest.TestCase):
    """Tests covering GPU/Host architecture resolution across profiles."""

    def setUp(self) -> None:
        self.resolver = ConfigResolver()

    def test_t4_sm75_balanced_and_safe(self) -> None:
        fp = _make_fp(gpu_name="Tesla T4", major=7, minor=5, vram_mb=15360.0, physical_cores=2, logical_cores=2)
        res_bal = self.resolver.resolve(fp, profile="balanced", env={})
        cfg = res_bal.config
        self.assertFalse(cfg.runtime.tf32)
        self.assertTrue(cfg.amp.enabled)
        self.assertEqual(cfg.amp.dtype, "fp16")
        self.assertTrue(cfg.amp.grad_scaler)
        self.assertFalse(cfg.attention.flash_sdp)
        self.assertTrue(cfg.attention.mem_efficient_sdp)
        self.assertTrue(cfg.optimizer.fused)
        self.assertTrue(cfg.compile.enabled)
        self.assertFalse(cfg.kernels.enabled)

        res_safe = self.resolver.resolve(fp, profile="safe", env={})
        self.assertFalse(res_safe.config.amp.enabled)
        self.assertFalse(res_safe.config.amp.grad_scaler)
        self.assertFalse(res_safe.config.compile.enabled)
        self.assertFalse(res_safe.config.runtime.tf32)
        self.assertFalse(res_safe.config.runtime.cudnn_benchmark)
        self.assertTrue(res_safe.config.optimizer.fused)

    def test_a100_sm80_supersafe_safe_and_balanced_tf32(self) -> None:
        fp = _make_fp(gpu_name="NVIDIA A100-SXM4-40GB", major=8, minor=0, vram_mb=40960.0)
        res_super = self.resolver.resolve(fp, profile="supersafe", env={})
        self.assertFalse(res_super.config.runtime.tf32)
        self.assertFalse(res_super.config.runtime.cudnn_benchmark)
        self.assertFalse(res_super.config.amp.enabled)
        self.assertFalse(res_super.config.compile.enabled)
        self.assertFalse(res_super.config.dataloader.auto_tune)

        res_safe = self.resolver.resolve(fp, profile="safe", env={})
        self.assertTrue(res_safe.config.runtime.tf32)
        self.assertFalse(res_safe.config.runtime.cudnn_benchmark)
        self.assertFalse(res_safe.config.amp.enabled)
        self.assertFalse(res_safe.config.compile.enabled)
        self.assertTrue(res_safe.config.dataloader.auto_tune)

        res = self.resolver.resolve(fp, profile="balanced", env={})
        cfg = res.config
        self.assertTrue(cfg.runtime.tf32)
        self.assertFalse(cfg.runtime.bandwidth_boost)
        self.assertFalse(cfg.runtime.cudnn_benchmark)
        self.assertFalse(cfg.dataloader.cuda_prefetch)
        self.assertEqual(cfg.dataloader.prefetch_factor, 2)
        self.assertNotIn("vram_bandwidth_boost", res.matched_rules)
        self.assertTrue(cfg.amp.enabled)
        self.assertEqual(cfg.amp.dtype, "bf16")
        self.assertFalse(cfg.amp.grad_scaler)
        self.assertTrue(cfg.attention.flash_sdp)
        self.assertTrue(cfg.compile.enabled)
        self.assertEqual(cfg.compile.mode, "default")

        # Opt-in Phase 8 bandwidth boost enables high-VRAM throughput settings
        res_boost = self.resolver.resolve(
            fp,
            profile="balanced",
            env={"PTOPT_BANDWIDTH_BOOST": "1"},
        )
        self.assertTrue(res_boost.config.runtime.bandwidth_boost)
        self.assertTrue(res_boost.config.runtime.cudnn_benchmark)
        self.assertTrue(res_boost.config.dataloader.cuda_prefetch)
        self.assertEqual(res_boost.config.dataloader.prefetch_factor, 4)
        self.assertIn("vram_bandwidth_boost", res_boost.matched_rules)

        res_cli = self.resolver.resolve(
            fp,
            profile="balanced",
            overrides={"runtime.bandwidth_boost": True},
            env={},
        )
        self.assertTrue(res_cli.config.runtime.bandwidth_boost)
        self.assertIn("vram_bandwidth_boost", res_cli.matched_rules)

        # Guards: supersafe, <12GB VRAM, and CPU keep Phase 8 boost inactive
        res_sup = self.resolver.resolve(
            fp,
            profile="supersafe",
            overrides={"runtime.bandwidth_boost": True},
            env={},
        )
        self.assertNotIn("vram_bandwidth_boost", res_sup.matched_rules)
        self.assertFalse(res_sup.config.runtime.cudnn_benchmark)
        self.assertFalse(res_sup.config.dataloader.cuda_prefetch)

        fp_small = _make_fp(gpu_name="GTX 1070", major=6, minor=1, vram_mb=8192.0)
        res_small = self.resolver.resolve(
            fp_small,
            profile="balanced",
            overrides={"runtime.bandwidth_boost": True},
            env={},
        )
        self.assertNotIn("vram_bandwidth_boost", res_small.matched_rules)
        self.assertFalse(res_small.config.runtime.cudnn_benchmark)
        self.assertFalse(res_small.config.dataloader.cuda_prefetch)

        fp_cpu = _make_fp(backend="cpu", gpu_name="cpu", major=0, minor=0, vram_mb=0.0)
        res_cpu = self.resolver.resolve(
            fp_cpu,
            profile="balanced",
            overrides={"runtime.bandwidth_boost": True},
            env={},
        )
        self.assertNotIn("vram_bandwidth_boost", res_cpu.matched_rules)
        self.assertFalse(res_cpu.config.runtime.cudnn_benchmark)
        self.assertFalse(res_cpu.config.dataloader.cuda_prefetch)

    def test_l4_sm89_aggressive(self) -> None:
        fp = _make_fp(gpu_name="NVIDIA L4", major=8, minor=9, vram_mb=22563.1, sm_count=58, triton_version="3.8.0")
        res = self.resolver.resolve(fp, profile="aggressive", env={})
        cfg = res.config
        self.assertTrue(cfg.runtime.tf32)
        self.assertEqual(cfg.amp.dtype, "bf16")
        self.assertEqual(cfg.compile.mode, "default")
        self.assertTrue(cfg.compile.train_cuda_graphs)
        self.assertTrue(cfg.inference.cuda_graphs)
        self.assertTrue(cfg.kernels.enabled)
        self.assertIn("gpu_cuda_ada_sm89", res.matched_rules)
        self.assertIn("profile_aggressive_low_sm_cuda", res.matched_rules)
        self.assertNotIn("profile_aggressive_high_vram_cuda", res.matched_rules)
        self.assertIn("profile_aggressive_triton_kernels", res.matched_rules)

    def test_a100_and_sm68_aggressive_use_max_autotune(self) -> None:
        for name, maj, min_, vram, sms in (
            ("NVIDIA A100-SXM4-40GB", 8, 0, 40960.0, 108),
            ("NVIDIA GeForce RTX 3080 16GB", 8, 6, 20480.0, 68),
        ):
            fp = _make_fp(gpu_name=name, major=maj, minor=min_, vram_mb=vram, sm_count=sms, triton_version="3.8.0")
            res = self.resolver.resolve(fp, profile="aggressive", env={})
            cfg = res.config
            self.assertEqual(cfg.compile.mode, "max-autotune")
            self.assertTrue(cfg.compile.train_cuda_graphs)
            self.assertTrue(cfg.inference.cuda_graphs)
            self.assertIn("profile_aggressive_high_vram_cuda", res.matched_rules)

    def test_low_vram_disables_cuda_graphs_and_keeps_default_compile(self) -> None:
        fp = _make_fp(gpu_name="RTX 3060 8GB", major=8, minor=6, vram_mb=8192.0)
        res = self.resolver.resolve(fp, profile="aggressive", env={})
        self.assertEqual(res.config.compile.mode, "default")
        self.assertFalse(res.config.inference.cuda_graphs)
        self.assertIn("vram_lt_12gb", res.matched_rules)

    def test_low_shm_disables_dataloader_autotune(self) -> None:
        fp = _make_fp(shm_total_mb=512.0)
        res = self.resolver.resolve(fp, profile="balanced", env={})
        self.assertFalse(res.config.dataloader.auto_tune)
        self.assertEqual(res.config.dataloader.num_workers, 0)
        self.assertFalse(res.config.dataloader.persistent_workers)
        self.assertIsNone(res.config.dataloader.prefetch_factor)
        self.assertIn("host_low_shm_guard", res.matched_rules)

    def test_cpu_only_and_mps(self) -> None:
        fp_cpu = _make_fp(backend="cpu", physical_cores=6, logical_cores=12, jemalloc_path="libjemalloc.so.2")
        res_cpu = self.resolver.resolve(fp_cpu, profile="aggressive", env={})
        self.assertFalse(res_cpu.config.amp.enabled)
        self.assertFalse(res_cpu.config.optimizer.fused)
        self.assertFalse(res_cpu.config.runtime.tf32)
        self.assertEqual(res_cpu.config.runtime.omp_threads, 6)
        self.assertTrue(res_cpu.config.runtime.jemalloc)

        fp_mps = _make_fp(backend="mps", os_name="Darwin", cpu_arch="arm64", shm_total_mb=None, is_colab=False)
        res_mps = self.resolver.resolve(fp_mps, profile="balanced", env={})
        self.assertFalse(res_mps.config.dataloader.pin_memory)
        self.assertFalse(res_mps.config.optimizer.fused)

    def test_overrides_and_trace(self) -> None:
        fp = _make_fp(gpu_name="NVIDIA L4", major=8, minor=9, vram_mb=22563.1)
        res = self.resolver.resolve(
            fp,
            env={"PTOPT_PROFILE": "balanced", "PTOPT_COMPILE": "0", "PTOPT_AMP_DTYPE": "fp16"},
            overrides={"amp.enabled": False, "runtime.omp_threads": 4},
        )
        self.assertEqual(res.config.profile, "balanced")
        self.assertFalse(res.config.compile.enabled)
        self.assertFalse(res.config.amp.enabled)
        self.assertFalse(res.config.amp.grad_scaler)
        self.assertEqual(res.config.runtime.omp_threads, 4)
        self.assertTrue(any(entry.rule_name == "cli_overrides" for entry in res.trace))


class TestConfigValidation(unittest.TestCase):
    """Tests for Schema v1 validation errors and edge constraints."""

    def test_invalid_profile_and_env(self) -> None:
        fp = _make_fp(backend="cpu")
        resolver = ConfigResolver()
        with self.assertRaises(ConfigValidationError):
            resolver.resolve(fp, profile="turbo", env={})
        with self.assertRaises(ConfigValidationError):
            resolver.resolve(fp, env={"PTOPT_COMPILE": "maybe"})

    def test_unknown_keys_and_invalid_constraints(self) -> None:
        with self.assertRaises(ConfigValidationError):
            OptimizeConfig.from_dict({"unknown_top": 1})
        with self.assertRaises(ConfigValidationError):
            OptimizeConfig.from_dict({"amp": {"unknown_field": True}})
        with self.assertRaises(ConfigValidationError):
            OptimizeConfig.from_dict({"amp": {"enabled": True, "dtype": "bfloat8"}})
        with self.assertRaises(ConfigValidationError):
            OptimizeConfig.from_dict({"amp": {"enabled": True, "dtype": "bf16", "grad_scaler": True}})
        with self.assertRaises(ConfigValidationError):
            OptimizeConfig.from_dict({"dataloader": {"num_workers": 0, "persistent_workers": True, "prefetch_factor": None}})

        cfg = OptimizeConfig()
        cfg.runtime.tf32 = True
        with self.assertRaises(ConfigValidationError):
            validate_config(cfg, backend="cpu")


if __name__ == "__main__":
    unittest.main()
