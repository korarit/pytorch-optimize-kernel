"""Unit tests for Phase 1.1: GPU/Host hardware probe and TextReportBuilder."""

from __future__ import annotations

import datetime
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.text_report import TextReportBuilder, format_report_filename
from src.detect import HardwareFingerprint, collect_fingerprint, probe_gpu, probe_host


def _make_fake_torch(
    *,
    cuda_available: bool = False,
    mps_available: bool = False,
    devices: list[tuple[str, int, int, int, int]] | None = None,
    bf16_supported: bool = False,
    cuda_version: str | None = None,
    hip_version: str | None = None,
    cuda_error: Exception | None = None,
    has_flash_sdp: bool = True,
) -> SimpleNamespace:
    """Create a lightweight mock torch module for hardware probe testing."""
    device_specs = devices or []

    def _is_cuda_avail() -> bool:
        if cuda_error is not None:
            raise cuda_error
        return cuda_available

    def _get_props(idx: int) -> SimpleNamespace:
        name, major, minor, vram_bytes, sm_count = device_specs[idx]
        return SimpleNamespace(
            name=name,
            major=major,
            minor=minor,
            total_memory=vram_bytes,
            multi_processor_count=sm_count,
        )

    cuda_ns = SimpleNamespace(
        is_available=_is_cuda_avail,
        device_count=lambda: len(device_specs),
        get_device_properties=_get_props,
        is_bf16_supported=lambda: bf16_supported,
    )
    cuda_backends = SimpleNamespace(flash_sdp_enabled=lambda: True) if has_flash_sdp else SimpleNamespace()
    mps_ns = SimpleNamespace(is_available=lambda: mps_available)
    backends_ns = SimpleNamespace(cuda=cuda_backends, mps=mps_ns)
    version_ns = SimpleNamespace(cuda=cuda_version, hip=hip_version)
    return SimpleNamespace(
        __version__="2.14.1+cu130" if cuda_version else "2.15.0+cpu",
        version=version_ns,
        cuda=cuda_ns,
        backends=backends_ns,
    )


def _fake_statvfs(total_mb: int, free_mb: int):
    block_size = 4096
    total_blocks = int((total_mb * 1024 * 1024) // block_size)
    free_blocks = int((free_mb * 1024 * 1024) // block_size)
    return lambda _path: SimpleNamespace(f_frsize=block_size, f_blocks=total_blocks, f_bavail=free_blocks)


def _write_fake_proc(proc_dir: Path, *, cpu_model: str, cores: int, flags: str, ram_mb: int) -> None:
    blocks = []
    for core_idx in range(cores):
        blocks.append(
            "\n".join([
                f"processor\t: {core_idx}",
                f"model name\t: {cpu_model}",
                "physical id\t: 0",
                f"core id\t\t: {core_idx}",
                f"flags\t\t: {flags}",
            ])
        )
    (proc_dir / "cpuinfo").write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    ram_kb = ram_mb * 1024
    (proc_dir / "meminfo").write_text(f"MemTotal: {ram_kb} kB\nMemAvailable: {ram_kb // 2} kB\n", encoding="utf-8")


class TestHardwareDetection(unittest.TestCase):
    """Tests covering the 7 mocked hardware/environment scenarios."""

    def test_colab_t4_sm75_low_shm(self) -> None:
        torch_mock = _make_fake_torch(
            cuda_available=True,
            devices=[("Tesla T4", 7, 5, 15 * 1024**3, 40)],
            bf16_supported=False,
            cuda_version="12.4",
        )
        with tempfile.TemporaryDirectory() as tmp:
            proc_dir = Path(tmp)
            _write_fake_proc(proc_dir, cpu_model="Intel Xeon", cores=2, flags="avx2 avx512f", ram_mb=12800)
            host = probe_host(
                torch_mod=torch_mock,
                env={"COLAB_RELEASE_TAG": "release-colab-t4", "COLAB_GPU": "1"},
                proc_root=proc_dir,
                statvfs_fn=_fake_statvfs(total_mb=1024, free_mb=1024),
                pkg_version_fn=lambda name: "3.1.0" if name == "triton" else None,
                find_lib_fn=lambda name: "libtcmalloc.so.4" if "tcmalloc" in name else None,
                os_name_override="Linux",
                cpu_arch_override="x86_64",
                logical_cores_override=2,
            )
            gpu = probe_gpu(torch_mod=torch_mock, env={}, query_driver=False)
            fp = HardwareFingerprint(host=host, gpu=gpu)

        self.assertTrue(fp.is_colab)
        self.assertTrue(fp.is_low_shm)
        self.assertEqual(fp.effective_backend, "cuda")
        self.assertEqual(fp.sm_int, 75)
        self.assertIsNotNone(fp.primary_gpu)
        assert fp.primary_gpu is not None
        self.assertFalse(fp.primary_gpu.tf32_supported)
        self.assertFalse(fp.primary_gpu.bf16_supported)
        self.assertFalse(fp.primary_gpu.fp8_supported)
        self.assertFalse(fp.primary_gpu.flash_sdp_supported)
        self.assertTrue(fp.primary_gpu.mem_efficient_sdp_supported)
        self.assertEqual(fp.host.tcmalloc_path, "libtcmalloc.so.4")

    def test_colab_l4_sm89(self) -> None:
        torch_mock = _make_fake_torch(
            cuda_available=True,
            devices=[("NVIDIA L4", 8, 9, int(22.5 * 1024**3), 58)],
            bf16_supported=True,
            cuda_version="13.0",
        )
        with tempfile.TemporaryDirectory() as tmp:
            proc_dir = Path(tmp)
            _write_fake_proc(proc_dir, cpu_model="Intel Xeon", cores=6, flags="avx2 avx512f", ram_mb=54229)
            host = probe_host(
                torch_mod=torch_mock,
                env={"COLAB_RELEASE_TAG": "release-colab-l4", "COLAB_GPU": "1"},
                proc_root=proc_dir,
                statvfs_fn=_fake_statvfs(total_mb=26155, free_mb=26155),
                pkg_version_fn=lambda name: "3.8.0" if name == "triton" else None,
                find_lib_fn=lambda _name: None,
                os_name_override="Linux",
                cpu_arch_override="x86_64",
                logical_cores_override=12,
            )
            gpu = probe_gpu(torch_mod=torch_mock, env={}, query_driver=False)
            fp = HardwareFingerprint(host=host, gpu=gpu)

        self.assertTrue(fp.is_colab)
        self.assertFalse(fp.is_low_shm)
        self.assertEqual(fp.sm_int, 89)
        assert fp.primary_gpu is not None
        self.assertTrue(fp.primary_gpu.tf32_supported)
        self.assertTrue(fp.primary_gpu.bf16_supported)
        self.assertTrue(fp.primary_gpu.fp8_supported)
        self.assertTrue(fp.primary_gpu.flash_sdp_supported)
        self.assertEqual(fp.host.triton_version, "3.8.0")

    def test_colab_a100_sm80(self) -> None:
        torch_mock = _make_fake_torch(
            cuda_available=True,
            devices=[("NVIDIA A100-SXM4-40GB", 8, 0, 40 * 1024**3, 108)],
            bf16_supported=True,
            cuda_version="12.4",
        )
        gpu = probe_gpu(torch_mod=torch_mock, env={}, query_driver=False)
        self.assertEqual(gpu.backend, "cuda")
        self.assertEqual(gpu.devices[0].sm_name, "sm_80")
        self.assertTrue(gpu.devices[0].tf32_supported)
        self.assertTrue(gpu.devices[0].bf16_supported)
        self.assertFalse(gpu.devices[0].fp8_supported)
        self.assertTrue(gpu.devices[0].flash_sdp_supported)

    def test_hopper_h100_sm90_multigpu(self) -> None:
        torch_mock = _make_fake_torch(
            cuda_available=True,
            devices=[
                ("NVIDIA H100 80GB HBM3", 9, 0, 80 * 1024**3, 132),
                ("NVIDIA H100 80GB HBM3", 9, 0, 80 * 1024**3, 132),
            ],
            bf16_supported=True,
            cuda_version="12.6",
        )
        gpu = probe_gpu(torch_mod=torch_mock, env={"LOCAL_RANK": "1", "CUDA_VISIBLE_DEVICES": "0,1"}, query_driver=False)
        self.assertEqual(gpu.device_count, 2)
        self.assertEqual(gpu.selected_index, 1)
        self.assertEqual(gpu.devices[1].sm_int, 90)
        self.assertTrue(gpu.devices[1].fp8_supported)

    def test_apple_silicon_mps_arm64(self) -> None:
        torch_mock = _make_fake_torch(cuda_available=False, mps_available=True)
        with tempfile.TemporaryDirectory() as tmp:
            host = probe_host(
                torch_mod=torch_mock,
                env={},
                proc_root=Path(tmp),
                pkg_version_fn=lambda _name: None,
                find_lib_fn=lambda _name: None,
                os_name_override="Darwin",
                cpu_arch_override="arm64",
                logical_cores_override=10,
            )
            gpu = probe_gpu(torch_mod=torch_mock, env={}, query_driver=False)
            fp = HardwareFingerprint(host=host, gpu=gpu)

        self.assertEqual(fp.effective_backend, "mps")
        self.assertEqual(fp.host.os_name, "Darwin")
        self.assertEqual(fp.host.cpu_arch, "arm64")
        self.assertIsNone(fp.primary_gpu)
        self.assertIsNone(fp.host.shm_total_mb)

    def test_cpu_only_x86_and_aarch64(self) -> None:
        torch_mock = _make_fake_torch(cuda_available=False, mps_available=False)
        for arch, flags in (("x86_64", "avx2 avx512_bf16"), ("aarch64", "asimd neon")):
            with tempfile.TemporaryDirectory() as tmp:
                proc_dir = Path(tmp)
                _write_fake_proc(proc_dir, cpu_model=f"CPU-{arch}", cores=4, flags=flags, ram_mb=16384)
                host = probe_host(
                    torch_mod=torch_mock,
                    env={},
                    proc_root=proc_dir,
                    statvfs_fn=_fake_statvfs(total_mb=4096, free_mb=4096),
                    pkg_version_fn=lambda _name: None,
                    find_lib_fn=lambda _name: None,
                    os_name_override="Linux",
                    cpu_arch_override=arch,
                    logical_cores_override=4,
                )
                gpu = probe_gpu(torch_mod=torch_mock, env={}, query_driver=False)
                self.assertEqual(gpu.backend, "cpu")
                self.assertEqual(host.cpu_arch, arch)
                self.assertEqual(host.physical_cores, 4)

    def test_low_shm_and_cuda_runtime_error_fallback(self) -> None:
        torch_mock = _make_fake_torch(cuda_error=RuntimeError("CUDA driver broken"))
        with tempfile.TemporaryDirectory() as tmp:
            proc_dir = Path(tmp)
            _write_fake_proc(proc_dir, cpu_model="Container CPU", cores=2, flags="avx2", ram_mb=4096)
            host = probe_host(
                torch_mod=torch_mock,
                env={},
                proc_root=proc_dir,
                statvfs_fn=_fake_statvfs(total_mb=512, free_mb=256),
                pkg_version_fn=lambda _name: None,
                find_lib_fn=lambda _name: None,
                os_name_override="Linux",
                cpu_arch_override="x86_64",
                logical_cores_override=2,
            )
            gpu = probe_gpu(torch_mod=torch_mock, env={}, query_driver=False)
            fp = HardwareFingerprint(host=host, gpu=gpu)

        self.assertEqual(fp.effective_backend, "cpu")
        self.assertTrue(fp.is_low_shm)
        self.assertEqual(fp.host.shm_total_mb, 512.0)
        self.assertTrue(any("CUDA driver broken" in err for err in fp.gpu.probe_errors))

    def test_fingerprint_json_roundtrip(self) -> None:
        torch_mock = _make_fake_torch(
            cuda_available=True,
            devices=[("NVIDIA L4", 8, 9, 23 * 1024**3, 58)],
            bf16_supported=True,
            cuda_version="13.0",
        )
        fp = collect_fingerprint(torch_mod=torch_mock, env={"COLAB_GPU": "1"}, query_driver=False)
        payload = fp.to_json()
        reloaded = HardwareFingerprint.from_dict(json.loads(payload))
        self.assertEqual(reloaded.to_dict(), fp.to_dict())


class TestTextReportBuilder(unittest.TestCase):
    """Tests for shared plain-text report formatting and file naming."""

    def test_report_filename_and_output(self) -> None:
        fixed_now = datetime.datetime(2026, 11, 17, 15, 5, 9)
        self.assertEqual(format_report_filename("phase-1.1", now=fixed_now), "150509_20261117-phase-1.1.txt")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            builder = TextReportBuilder(phase_title="PHASE 1.1 TEST", repo_root=root, now=fixed_now)
            builder.add_header({"Target": "Colab L4"})
            builder.add_kv_section("SUMMARY", {"Backend": "cuda", "SM": "sm_89"})
            builder.add_table("MATRIX", ["Arch", "Status"], [["sm_75", "PASS"], ["sm_89", "PASS"]])
            builder.add_verification_footer()
            out_path = builder.write("phase-1.1")

            self.assertEqual(out_path, root / "report" / "150509_20261117-phase-1.1.txt")
            content = out_path.read_text(encoding="utf-8")
            self.assertIn("PHASE 1.1 TEST", content)
            self.assertIn("sm_89", content)
            self.assertIn("PENDING_USER_REVIEW", content)


if __name__ == "__main__":
    unittest.main()
