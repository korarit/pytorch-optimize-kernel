"""Built-in layered rules for hardware-aware config resolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from src.detect.fingerprint import HardwareFingerprint

LAYER_ORDER: tuple[str, ...] = (
    "base",
    "host_arch",
    "gpu_arch",
    "vram_tier",
    "user_profile",
)


@dataclass(frozen=True)
class ConfigRule:
    """A single declarative/conditional rule in a resolution layer."""

    name: str
    layer: str
    reason: str
    condition: Callable[[HardwareFingerprint, str], bool]
    updates: Callable[[HardwareFingerprint, str], dict[str, Any]] | dict[str, Any]

    def evaluate(self, fp: HardwareFingerprint, profile: str) -> dict[str, Any] | None:
        if not self.condition(fp, profile):
            return None
        if callable(self.updates):
            return self.updates(fp, profile)
        return dict(self.updates)


def _is_cuda(fp: HardwareFingerprint) -> bool:
    return fp.effective_backend == "cuda" and fp.primary_gpu is not None


def _recommended_workers(fp: HardwareFingerprint) -> int:
    cores = fp.host.physical_cores or fp.host.logical_cores or 2
    return min(4, max(1, cores // 2))


def get_builtin_rules() -> list[ConfigRule]:
    """Return ordered built-in rules across all hardware and profile layers."""
    return [
        # 1. base
        ConfigRule(
            name="base_safe_defaults",
            layer="base",
            reason="Initialize conservative Schema v1 defaults",
            condition=lambda _fp, _p: True,
            updates={
                "runtime.tf32": False,
                "runtime.cudnn_benchmark": False,
                "runtime.alloc_conf": "expandable_segments:True",
                "runtime.omp_threads": 0,
                "runtime.jemalloc": False,
                "amp.enabled": False,
                "amp.dtype": "fp32",
                "amp.grad_scaler": False,
                "compile.enabled": False,
                "compile.mode": "default",
                "compile.dynamic": False,
                "compile.cache_dir": ".torch_inductor_cache",
                "compile.train_cuda_graphs": False,
                "optimizer.fused": False,
                "optimizer.foreach": True,
                "attention.flash_sdp": False,
                "attention.mem_efficient_sdp": True,
                "attention.math_sdp": True,
                "dataloader.auto_tune": False,
                "dataloader.pin_memory": False,
                "dataloader.persistent_workers": False,
                "dataloader.num_workers": None,
                "dataloader.prefetch_factor": 2,
                "inference.inference_mode": False,
                "inference.cuda_graphs": False,
                "kernels.enabled": False,
                "kernels.allow_list": [],
                "kernels.fallback_on_error": True,
            },
        ),
        # 2. host_arch
        ConfigRule(
            name="host_linux_x86_64",
            layer="host_arch",
            reason="Tune CPU threads and jemalloc for Linux x86_64",
            condition=lambda fp, _p: fp.host.os_name.lower() == "linux" and fp.host.cpu_arch == "x86_64",
            updates=lambda fp, _p: {
                "runtime.omp_threads": fp.host.physical_cores or fp.host.logical_cores or 0,
                "runtime.jemalloc": fp.host.jemalloc_path is not None,
            },
        ),
        ConfigRule(
            name="host_linux_aarch64",
            layer="host_arch",
            reason="Tune CPU threads and jemalloc for Linux aarch64",
            condition=lambda fp, _p: fp.host.os_name.lower() == "linux" and fp.host.cpu_arch in ("aarch64", "arm64"),
            updates=lambda fp, _p: {
                "runtime.omp_threads": fp.host.physical_cores or fp.host.logical_cores or 0,
                "runtime.jemalloc": fp.host.jemalloc_path is not None,
            },
        ),
        ConfigRule(
            name="host_darwin_mps",
            layer="host_arch",
            reason="Tune threads and disable CUDA pin_memory on macOS",
            condition=lambda fp, _p: fp.host.os_name.lower() == "darwin",
            updates=lambda fp, _p: {
                "runtime.omp_threads": fp.host.physical_cores or fp.host.logical_cores or 0,
                "dataloader.pin_memory": False,
            },
        ),
        ConfigRule(
            name="host_colab_runtime",
            layer="host_arch",
            reason="Google Colab runtime detected: keep persistent inductor cache path",
            condition=lambda fp, _p: fp.is_colab,
            updates={"compile.cache_dir": ".torch_inductor_cache"},
        ),
        ConfigRule(
            name="host_low_shm_guard",
            layer="host_arch",
            reason="/dev/shm < 2048 MB: disable multiprocessing DataLoader auto-tuning",
            condition=lambda fp, _p: fp.is_low_shm,
            updates={
                "dataloader.auto_tune": False,
                "dataloader.num_workers": 0,
                "dataloader.persistent_workers": False,
                "dataloader.prefetch_factor": None,
            },
        ),
        # 3. gpu_arch
        ConfigRule(
            name="gpu_cpu_only",
            layer="gpu_arch",
            reason="CPU-only backend: disable CUDA/GPU flags",
            condition=lambda fp, _p: fp.effective_backend == "cpu",
            updates={
                "runtime.tf32": False,
                "runtime.cudnn_benchmark": False,
                "runtime.alloc_conf": None,
                "amp.enabled": False,
                "amp.dtype": "fp32",
                "amp.grad_scaler": False,
                "optimizer.fused": False,
                "attention.flash_sdp": False,
                "attention.mem_efficient_sdp": False,
                "dataloader.pin_memory": False,
                "inference.cuda_graphs": False,
                "compile.train_cuda_graphs": False,
                "kernels.enabled": False,
            },
        ),
        ConfigRule(
            name="gpu_mps",
            layer="gpu_arch",
            reason="Apple MPS backend: disable CUDA-specific flags",
            condition=lambda fp, _p: fp.effective_backend == "mps",
            updates={
                "runtime.tf32": False,
                "runtime.cudnn_benchmark": False,
                "runtime.alloc_conf": None,
                "optimizer.fused": False,
                "attention.flash_sdp": False,
                "attention.mem_efficient_sdp": False,
                "dataloader.pin_memory": False,
                "inference.cuda_graphs": False,
                "compile.train_cuda_graphs": False,
                "kernels.enabled": False,
            },
        ),
        ConfigRule(
            name="gpu_rocm",
            layer="gpu_arch",
            reason="ROCm backend: default to fp16 AMP without NVIDIA TF32",
            condition=lambda fp, _p: fp.effective_backend == "rocm",
            updates={
                "runtime.tf32": False,
                "amp.dtype": "fp16",
                "amp.grad_scaler": True,
                "attention.flash_sdp": False,
            },
        ),
        ConfigRule(
            name="gpu_cuda_volta_turing_sm70_79",
            layer="gpu_arch",
            reason="CUDA sm_70-79 (e.g. T4): no TF32/BF16 hardware; prefer FP16 + Mem-Efficient SDPA",
            condition=lambda fp, _p: _is_cuda(fp) and 70 <= fp.sm_int < 80,
            updates={
                "runtime.tf32": False,
                "amp.dtype": "fp16",
                "amp.grad_scaler": True,
                "attention.flash_sdp": False,
                "attention.mem_efficient_sdp": True,
                "compile.train_cuda_graphs": False,
            },
        ),
        ConfigRule(
            name="gpu_cuda_ampere_sm80_86",
            layer="gpu_arch",
            reason="CUDA Ampere sm_80-86 (e.g. A100/RTX 30xx): BF16 & Flash SDPA capable",
            condition=lambda fp, _p: _is_cuda(fp) and 80 <= fp.sm_int <= 86,
            updates={
                "amp.dtype": "bf16",
                "amp.grad_scaler": False,
                "attention.flash_sdp": True,
                "attention.mem_efficient_sdp": True,
            },
        ),
        ConfigRule(
            name="gpu_cuda_ada_sm89",
            layer="gpu_arch",
            reason="CUDA Ada Lovelace sm_89 (e.g. Colab L4/RTX 40xx): BF16 & Flash SDPA capable",
            condition=lambda fp, _p: _is_cuda(fp) and fp.sm_int == 89,
            updates={
                "amp.dtype": "bf16",
                "amp.grad_scaler": False,
                "attention.flash_sdp": True,
                "attention.mem_efficient_sdp": True,
            },
        ),
        ConfigRule(
            name="gpu_cuda_hopper_sm90_plus",
            layer="gpu_arch",
            reason="CUDA Hopper+ sm_90+ (e.g. H100): BF16 & Flash SDPA capable",
            condition=lambda fp, _p: _is_cuda(fp) and fp.sm_int >= 90,
            updates={
                "amp.dtype": "bf16",
                "amp.grad_scaler": False,
                "attention.flash_sdp": True,
                "attention.mem_efficient_sdp": True,
            },
        ),
        # 4. vram_tier
        ConfigRule(
            name="vram_lt_12gb",
            layer="vram_tier",
            reason="VRAM < 12 GB: favor memory-saving allocator & conservative compile",
            condition=lambda fp, _p: _is_cuda(fp) and 0 < fp.vram_mb < 12288,
            updates={
                "runtime.alloc_conf": "expandable_segments:True",
                "compile.mode": "default",
                "inference.cuda_graphs": False,
                "compile.train_cuda_graphs": False,
            },
        ),
        ConfigRule(
            name="vram_12_24gb",
            layer="vram_tier",
            reason="VRAM 12-24 GB (e.g. T4/L4): use expandable_segments allocator",
            condition=lambda fp, _p: _is_cuda(fp) and 12288 <= fp.vram_mb < 32768,
            updates={"runtime.alloc_conf": "expandable_segments:True"},
        ),
        ConfigRule(
            name="vram_ge_40gb",
            layer="vram_tier",
            reason="VRAM >= 40 GB (e.g. A100/H100): headroom available for aggressive tuning",
            condition=lambda fp, _p: _is_cuda(fp) and fp.vram_mb >= 40960,
            updates={"runtime.alloc_conf": "expandable_segments:True"},
        ),
        # 5. user_profile
        ConfigRule(
            name="profile_supersafe_cuda",
            layer="user_profile",
            reason="Supersafe profile on CUDA: strict FP32 SIMT parity (0.0 drift; no TF32/AMP/compile), fused optimizer, inference_mode",
            condition=lambda fp, p: p == "supersafe" and _is_cuda(fp),
            updates={
                "runtime.tf32": False,
                "runtime.cudnn_benchmark": False,
                "optimizer.fused": True,
                "inference.inference_mode": True,
            },
        ),
        ConfigRule(
            name="profile_supersafe_non_cuda",
            layer="user_profile",
            reason="Supersafe profile on CPU/MPS: strict FP32 with inference_mode only",
            condition=lambda fp, p: p == "supersafe" and not _is_cuda(fp),
            updates={"inference.inference_mode": True},
        ),
        ConfigRule(
            name="profile_safe_cuda",
            layer="user_profile",
            reason="Safe profile on CUDA: FP32 weights + TF32 Tensor Core matmul (sm>=80), fused optimizer, inference_mode (no AMP/compile)",
            condition=lambda fp, p: p == "safe" and _is_cuda(fp),
            updates=lambda fp, _p: {
                "runtime.tf32": fp.sm_int >= 80,
                "runtime.cudnn_benchmark": False,
                "optimizer.fused": True,
                "inference.inference_mode": True,
            },
        ),
        ConfigRule(
            name="profile_safe_non_cuda",
            layer="user_profile",
            reason="Safe profile on CPU/MPS: enable inference_mode only",
            condition=lambda fp, p: p == "safe" and not _is_cuda(fp),
            updates={"inference.inference_mode": True},
        ),
        ConfigRule(
            name="profile_balanced_cuda",
            layer="user_profile",
            reason="Balanced profile on CUDA: enable TF32 (sm>=80), AMP, fused optimizer, compile(default), inference_mode (keep cudnn_benchmark off to avoid +125MB workspace bloat)",
            condition=lambda fp, p: p == "balanced" and _is_cuda(fp),
            updates=lambda fp, _p: {
                "runtime.tf32": fp.sm_int >= 80,
                "runtime.cudnn_benchmark": False,
                "amp.enabled": True,
                "optimizer.fused": True,
                "compile.enabled": True,
                "compile.mode": "default",
                "inference.inference_mode": True,
            },
        ),
        ConfigRule(
            name="profile_balanced_dataloader",
            layer="user_profile",
            reason="Safe/Balanced/Aggressive with healthy /dev/shm: enable DataLoader auto-tuning",
            condition=lambda fp, p: p in ("safe", "balanced", "aggressive") and _is_cuda(fp) and not fp.is_low_shm,
            updates=lambda fp, _p: {
                "dataloader.auto_tune": True,
                "dataloader.pin_memory": True,
                "dataloader.persistent_workers": True,
                "dataloader.num_workers": _recommended_workers(fp),
                "dataloader.prefetch_factor": 2,
            },
        ),
        ConfigRule(
            name="profile_balanced_non_cuda",
            layer="user_profile",
            reason="Balanced profile on CPU/MPS: enable inference_mode safely",
            condition=lambda fp, p: p == "balanced" and not _is_cuda(fp),
            updates={"inference.inference_mode": True},
        ),
        ConfigRule(
            name="profile_aggressive_cuda",
            layer="user_profile",
            reason="Aggressive profile on CUDA: enable TF32 (sm>=80), cudnn_benchmark, AMP, fused optimizer, compile, inference_mode",
            condition=lambda fp, p: p == "aggressive" and _is_cuda(fp),
            updates=lambda fp, _p: {
                "runtime.tf32": fp.sm_int >= 80,
                "runtime.cudnn_benchmark": True,
                "amp.enabled": True,
                "optimizer.fused": True,
                "compile.enabled": True,
                "inference.inference_mode": True,
            },
        ),
        ConfigRule(
            name="profile_aggressive_high_vram_cuda",
            layer="user_profile",
            reason="Aggressive profile on sm>=80 & VRAM>=16GB: enable max-autotune & CUDA graphs",
            condition=lambda fp, p: p == "aggressive" and _is_cuda(fp) and fp.sm_int >= 80 and fp.vram_mb >= 16384,
            updates={
                "compile.mode": "max-autotune",
                "inference.cuda_graphs": True,
                "compile.train_cuda_graphs": True,
            },
        ),
        ConfigRule(
            name="profile_aggressive_triton_kernels",
            layer="user_profile",
            reason="Aggressive profile with Triton & sm>=80: enable custom Triton kernels",
            condition=lambda fp, p: p == "aggressive" and _is_cuda(fp) and fp.sm_int >= 80 and bool(fp.host.triton_version),
            updates={"kernels.enabled": True},
        ),
        ConfigRule(
            name="profile_aggressive_non_cuda",
            layer="user_profile",
            reason="Aggressive profile on CPU/MPS: enable inference_mode",
            condition=lambda fp, p: p == "aggressive" and not _is_cuda(fp),
            updates={"inference.inference_mode": True},
        ),
    ]
