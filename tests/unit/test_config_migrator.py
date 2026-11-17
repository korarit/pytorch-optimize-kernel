"""Unit tests for Phase 4.1: Config Schema v0 -> v1 auto-migrator and runtime cache-key binding."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import (
    CACHE_META_FILENAME,
    SCHEMA_VERSION,
    ConfigValidationError,
    OptimizeConfig,
    build_runtime_cache_key,
    detect_runtime_cache_key,
    load_and_migrate_config,
    migrate_config_dict,
    validate_or_scope_cache_dir,
)
from src.passes.auto_compile import _configure_cache_and_dynamo
from src.passes.memory_allocator import apply_prelaunch_env


class TestConfigMigrator(unittest.TestCase):
    """Tests for Schema v0 / legacy config auto-migration with clear warnings."""

    def test_v0_flat_legacy_config_migrates_to_schema_v1_with_warnings(self) -> None:
        legacy = {
            "version": 0,
            "profile": "fast",
            "tf32": True,
            "use_amp": True,
            "amp_dtype": "float16",
            "grad_scaler": True,
            "use_compile": True,
            "compile_mode": "max_autotune",
            "dynamic_shapes": True,
            "fused_optimizer": True,
            "flash_attention": True,
            "pin_memory": True,
            "cuda_graphs": True,
            "custom_kernels": True,
            "quantization": {"enabled": False},
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = migrate_config_dict(legacy, emit_warnings=True)

        self.assertEqual(res.config.schema_version, SCHEMA_VERSION)
        self.assertEqual(res.config.profile, "aggressive")
        self.assertTrue(res.migrated)
        self.assertTrue(res.config.runtime.tf32)
        self.assertTrue(res.config.amp.enabled)
        self.assertEqual(res.config.amp.dtype, "fp16")
        self.assertTrue(res.config.amp.grad_scaler)
        self.assertTrue(res.config.compile.enabled)
        self.assertEqual(res.config.compile.mode, "max-autotune")
        self.assertTrue(res.config.compile.dynamic)
        self.assertTrue(res.config.optimizer.fused)
        self.assertTrue(res.config.attention.flash_sdp)
        self.assertTrue(res.config.dataloader.pin_memory)
        self.assertTrue(res.config.inference.cuda_graphs)
        self.assertTrue(res.config.kernels.enabled)
        self.assertGreater(len(res.warnings), 0)
        self.assertGreater(len(caught), 0)
        self.assertTrue(any("quantization" in w for w in res.warnings))

    def test_legacy_scalar_sections_and_incompatibilities_auto_remediated(self) -> None:
        legacy = {
            "schema_version": 0,
            "profile": "default",
            "amp": True,
            "amp_dtype": "bfloat16",
            "grad_scaler": True,
            "compile": True,
            "optimizer": "fused",
            "num_workers": 0,
            "persistent_workers": True,
            "prefetch_factor": 4,
            "kernels": {"ops": ["rmsnorm"]},
        }
        res = migrate_config_dict(legacy, emit_warnings=False)
        self.assertEqual(res.config.profile, "balanced")
        self.assertTrue(res.config.amp.enabled)
        self.assertEqual(res.config.amp.dtype, "bf16")
        self.assertFalse(res.config.amp.grad_scaler)
        self.assertTrue(res.config.compile.enabled)
        self.assertTrue(res.config.optimizer.fused)
        self.assertEqual(res.config.dataloader.num_workers, 0)
        self.assertFalse(res.config.dataloader.persistent_workers)
        self.assertIsNone(res.config.dataloader.prefetch_factor)
        self.assertEqual(res.config.kernels.allow_list, ["rmsnorm"])
        self.assertTrue(any("grad_scaler" in w for w in res.warnings))
        self.assertTrue(any("persistent_workers" in w for w in res.warnings))

    def test_strict_v1_unknown_and_future_version_still_rejected(self) -> None:
        with self.assertRaises(ConfigValidationError):
            migrate_config_dict({"schema_version": 1, "unknown_v1_field": 123}, emit_warnings=False)
        with self.assertRaises(ConfigValidationError):
            migrate_config_dict({"schema_version": 99}, emit_warnings=False)

    def test_load_and_migrate_json_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy_ptopt.json"
            path.write_text(json.dumps({"profile": "strict", "amp": False, "grad_scaler": True}), encoding="utf-8")
            res = load_and_migrate_config(path, emit_warnings=False)
            self.assertEqual(res.config.profile, "supersafe")
            self.assertFalse(res.config.amp.enabled)
            self.assertFalse(res.config.amp.grad_scaler)


class TestRuntimeCacheKeyValidation(unittest.TestCase):
    """Tests ensuring Inductor/Triton cache keys bind to (gpu_arch, torch_version, triton_version)."""

    def test_cache_key_changes_when_any_dimension_changes(self) -> None:
        k_t4 = build_runtime_cache_key("sm_75", "2.5.1+cu121", "3.1.0")
        k_l4 = build_runtime_cache_key("sm_89", "2.5.1+cu121", "3.1.0")
        k_pt = build_runtime_cache_key("sm_89", "2.6.0+cu124", "3.1.0")
        k_tr = build_runtime_cache_key("sm_89", "2.5.1+cu121", "3.2.0")

        self.assertEqual(k_t4.cache_key, build_runtime_cache_key("sm_75", "2.5.1+cu121", "3.1.0").cache_key)
        self.assertEqual(len({k_t4.cache_key, k_l4.cache_key, k_pt.cache_key, k_tr.cache_key}), 4)

    def test_stale_cache_from_different_gpu_or_version_never_reused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "shared_cache"
            k_t4 = build_runtime_cache_key("sm_75", "2.5.1+cu121", "3.1.0")
            k_l4 = build_runtime_cache_key("sm_89", "2.5.1+cu121", "3.1.0")
            k_pt26 = build_runtime_cache_key("sm_89", "2.6.0+cu124", "3.2.0")

            # First run on T4 populates base cache_dir with T4 metadata + dummy artifact
            res1 = validate_or_scope_cache_dir(base, k_t4, write_meta=True, emit_warnings=False)
            self.assertEqual(res1.cache_dir, base.resolve())
            self.assertFalse(res1.rotated)
            self.assertFalse(res1.reused)
            (res1.cache_dir / "stale_t4_artifact.bin").write_bytes(b"T4")
            self.assertTrue((base / CACHE_META_FILENAME).is_file())

            # Re-run on matching T4 reuses same directory
            res1_again = validate_or_scope_cache_dir(base, k_t4, write_meta=True, emit_warnings=False)
            self.assertEqual(res1_again.cache_dir, base.resolve())
            self.assertTrue(res1_again.reused)
            self.assertFalse(res1_again.rotated)

            # Switch to L4 on same cache_dir -> refuses T4 cache, isolates into key subdir + warns
            res2 = validate_or_scope_cache_dir(base, k_l4, write_meta=True, emit_warnings=False)
            self.assertTrue(res2.rotated)
            self.assertFalse(res2.reused)
            self.assertNotEqual(res2.cache_dir, base.resolve())
            self.assertEqual(res2.cache_dir, (base / k_l4.cache_key).resolve())
            self.assertFalse((res2.cache_dir / "stale_t4_artifact.bin").exists())
            self.assertTrue(any("mismatch" in w.lower() for w in res2.warnings))

            # Switch PyTorch & Triton version on same base cache_dir -> gets distinct subdir too
            res3 = validate_or_scope_cache_dir(base, k_pt26, write_meta=True, emit_warnings=False)
            self.assertTrue(res3.rotated)
            self.assertNotEqual(res3.cache_dir, res2.cache_dir)

    def test_passes_integrate_cache_key_rotation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "pass_cache"
            k_old = build_runtime_cache_key("sm_75", "2.4.0+cu121", "3.0.0")
            k_new = build_runtime_cache_key("sm_89", "2.5.1+cu121", "3.1.0")
            validate_or_scope_cache_dir(base, k_old, write_meta=True, emit_warnings=False)

            cfg = OptimizeConfig(profile="balanced")
            cfg.compile.enabled = True
            cfg.compile.cache_dir = str(base)

            env: dict[str, str] = {"TORCHINDUCTOR_CACHE_DIR": str(base)}
            with patch("src.passes.memory_allocator.detect_runtime_cache_key", return_value=k_new):
                notes = apply_prelaunch_env(cfg, env)
            self.assertEqual(env["TORCHINDUCTOR_CACHE_DIR"], str((base / k_new.cache_key).resolve()))
            self.assertEqual(env["PTOPT_CACHE_KEY"], k_new.cache_key)
            self.assertTrue(any("cache_rotated=" in n for n in notes))

            with (
                patch("src.passes.auto_compile.detect_runtime_cache_key", return_value=k_new),
                patch.dict(os.environ, {"TORCHINDUCTOR_CACHE_DIR": str(base)}, clear=False),
            ):
                resolved = _configure_cache_and_dynamo(cfg, torch_mod=None, strict=False)
                self.assertEqual(resolved, str((base / k_new.cache_key).resolve()))
                self.assertEqual(os.environ["TORCHINDUCTOR_CACHE_DIR"], resolved)

    def test_lazy_torch_import_invariant_for_migrator(self) -> None:
        code = (
            "import sys; "
            "from src.config import migrate_config_dict, detect_runtime_cache_key; "
            "res = migrate_config_dict({'tf32': True}, emit_warnings=False); "
            "key = detect_runtime_cache_key(); "
            "assert 'torch' not in sys.modules, 'torch eagerly imported'; "
            "print(res.config.schema_version, key.cache_key)"
        )
        proc = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
        self.assertTrue(proc.stdout.strip().startswith("1 "))


if __name__ == "__main__":
    unittest.main()
