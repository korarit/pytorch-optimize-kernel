"""Unit tests for Phase 3.2: DataLoader auto-tuning, /dev/shm guard, and pickle guard."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import OptimizeConfig
from src.hooks import HookManager
from src.passes.dataloader_tune import (
    decide_dataloader_params,
    install_dataloader_hook,
    is_picklable_fast,
    uninstall_dataloader_hook,
)


class _PicklableDataset(Dataset):
    def __init__(self, n: int = 16) -> None:
        self.x = torch.randn(n, 8)
        self.y = torch.randint(0, 4, (n,))

    def __len__(self) -> int:
        return self.x.size(0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class _LambdaUnpicklableDataset(Dataset):
    def __init__(self, n: int = 16) -> None:
        self.x = torch.randn(n, 8)
        self.fn = lambda t: t + 1.0

    def __len__(self) -> int:
        return self.x.size(0)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.fn(self.x[idx])


class TestDataLoaderTuningPass(unittest.TestCase):
    """Tests for DataLoader tuning decisions and safety guards."""

    def _tuned_cfg(self, *, num_workers: int | None = 3) -> OptimizeConfig:
        cfg = OptimizeConfig(profile="balanced")
        cfg.dataloader.auto_tune = True
        cfg.dataloader.pin_memory = True
        cfg.dataloader.persistent_workers = True
        cfg.dataloader.num_workers = num_workers
        cfg.dataloader.prefetch_factor = 2
        return cfg

    def test_low_shm_caps_or_disables_workers(self) -> None:
        ds = _PicklableDataset()
        cfg = self._tuned_cfg(num_workers=4)

        # < 1024 MB -> keep 0
        tuned_512, notes_512 = decide_dataloader_params(
            cfg=cfg,
            dataset=ds,
            collate_fn=None,
            worker_init_fn=None,
            orig_num_workers=0,
            orig_pin_memory=False,
            orig_persistent_workers=False,
            orig_prefetch_factor=None,
            torch_mod=torch,
            shm_mb=512.0,
            cpu_cores=8,
        )
        self.assertEqual(tuned_512["num_workers"], 0)
        self.assertFalse(tuned_512["persistent_workers"])
        self.assertIsNone(tuned_512["prefetch_factor"])
        self.assertTrue(any(status == "SKIPPED" for status, _ in notes_512))

        # 1024 <= shm < 2048 -> cap to 2
        tuned_1536, notes_1536 = decide_dataloader_params(
            cfg=cfg,
            dataset=ds,
            collate_fn=None,
            worker_init_fn=None,
            orig_num_workers=0,
            orig_pin_memory=False,
            orig_persistent_workers=False,
            orig_prefetch_factor=None,
            torch_mod=torch,
            shm_mb=1536.0,
            cpu_cores=8,
        )
        self.assertEqual(tuned_1536["num_workers"], 2)
        self.assertTrue(any("capping" in msg for _, msg in notes_1536))

    def test_unpicklable_lambda_dataset_and_collate_guard(self) -> None:
        bad_ds = _LambdaUnpicklableDataset()
        ok_fast, err = is_picklable_fast(bad_ds, torch_mod=torch)
        self.assertFalse(ok_fast)
        self.assertIsNotNone(err)

        cfg = self._tuned_cfg(num_workers=3)
        tuned, notes = decide_dataloader_params(
            cfg=cfg,
            dataset=bad_ds,
            collate_fn=None,
            worker_init_fn=None,
            orig_num_workers=0,
            orig_pin_memory=False,
            orig_persistent_workers=False,
            orig_prefetch_factor=None,
            torch_mod=torch,
            shm_mb=16384.0,
            cpu_cores=8,
        )
        self.assertEqual(tuned["num_workers"], 0)
        self.assertFalse(tuned["bumped"])
        self.assertTrue(any(status == "FALLBACK" and "Unpicklable" in msg for status, msg in notes))

        # Also check unpicklable collate_fn with picklable dataset
        tuned_col, notes_col = decide_dataloader_params(
            cfg=cfg,
            dataset=_PicklableDataset(),
            collate_fn=lambda batch: batch,
            worker_init_fn=None,
            orig_num_workers=0,
            orig_pin_memory=False,
            orig_persistent_workers=False,
            orig_prefetch_factor=None,
            torch_mod=torch,
            shm_mb=16384.0,
            cpu_cores=8,
        )
        self.assertEqual(tuned_col["num_workers"], 0)
        self.assertTrue(any("collate_fn" in msg for _, msg in notes_col))

    def test_preserves_explicit_user_num_workers(self) -> None:
        cfg = self._tuned_cfg(num_workers=4)
        tuned, _ = decide_dataloader_params(
            cfg=cfg,
            dataset=_PicklableDataset(),
            collate_fn=None,
            worker_init_fn=None,
            orig_num_workers=1,
            orig_pin_memory=False,
            orig_persistent_workers=False,
            orig_prefetch_factor=2,
            torch_mod=torch,
            shm_mb=16384.0,
            cpu_cores=8,
        )
        self.assertEqual(tuned["num_workers"], 1)
        self.assertFalse(tuned["bumped"])

    def test_fast_tensor_dataset_picklability_and_hook_execution(self) -> None:
        ds = TensorDataset(torch.randn(16, 4), torch.randint(0, 2, (16,)))
        ok, _ = is_picklable_fast(ds, torch_mod=torch)
        self.assertTrue(ok)

        cfg = self._tuned_cfg(num_workers=2)
        mgr = HookManager()
        install_dataloader_hook(cfg, mgr, torch, shm_probe_fn=lambda: 16384.0)
        try:
            dl_good = DataLoader(_PicklableDataset(), batch_size=4, num_workers=0)
            self.assertEqual(dl_good.num_workers, 2)
            self.assertTrue(dl_good.persistent_workers)
            batches = list(dl_good)
            self.assertEqual(len(batches), 4)

            dl_bad = DataLoader(_LambdaUnpicklableDataset(), batch_size=4, num_workers=0)
            self.assertEqual(dl_bad.num_workers, 0)
            self.assertFalse(dl_bad.persistent_workers)
            self.assertEqual(len(list(dl_bad)), 4)
        finally:
            uninstall_dataloader_hook(torch)
            mgr.reset()


if __name__ == "__main__":
    unittest.main()
