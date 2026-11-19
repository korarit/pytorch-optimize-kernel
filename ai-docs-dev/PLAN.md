# Project Plan: Zero-Code PyTorch Optimization Wrapper (`kernel-pytouch-optimize`)

## เป้าหมายหลัก (Objective)

สร้างระบบ **Zero-Code PyTorch Optimization Wrapper** ที่นำไปครอบคำสั่งรันโค้ด PyTorch เดิม (ทั้งการ **Training** และ **Running / Inference**) เพื่อเพิ่มความเร็วและลดการใช้ VRAM **โดยไม่ต้องแก้ไขโค้ดเทรนของผู้ใช้** พร้อมระบบ:

1. **Hardware-Aware Config Resolution**: ปรับ config อัตโนมัติตาม GPU Architecture และ Machine Architecture (รองรับการรันบน **Google Colab** เช่น T4 `sm_75`, L4 `sm_89`, A100 `sm_80`, รวมถึง CPU/container `/dev/shm` ที่มีข้อจำกัด)
2. **Modular Optimization Passes**: แยกเทคนิคการ optimize เป็นโมดูลอิสระ พร้อม capability gate และ auto-fallback
3. **Dedicated Optimization Lab**: แยก phase ทดลองและวัดผล A/B ออกจากโครงสร้างหลัก
4. **Unit Tests & Migrate Tests**: คุ้มครองทั้ง logic ภายใน, การรันโค้ดเดิมโดยไม่แก้, และการโหลด checkpoint ข้ามโหมด
5. **Detailed Text Report & Human Verification Gate (Google Colab)**: ทุก Phase และ Sub-Phase ต้องมี script สร้าง **Detailed Text Report (`.txt` / stdout)** และ **ต้องรอให้ผู้ใช้เอาไปรันบน Google Colab และตรวจสอบยืนยันด้วยตนเองก่อน** จึงจะไปต่อได้

---

## ดัชนีแผนงานราย Phase และ Sub-Phase (1 File = 1 Phase / Sub-Phase)

### Phase 0: Environment Discovery & Design Spec
- [phase-0-discovery-and-spec.md](./phase-0-discovery-and-spec.md) — ตรวจสเปกเครื่อง Colab, กำหนด Config Schema v1, Text Report Format และเกณฑ์ Parity

### Phase 1: Hardware Fingerprinting & Layered Config ([Overview](./phase-1-hardware-config-and-unit-tests.md))
- [Phase 1.1](./phase-1/1.1-hardware-probe.md) — GPU & Machine Hardware Probe + Text Report Foundation + Unit Tests
- [Phase 1.2](./phase-1/1.2-config-rule-engine.md) — Config Schema v1, Profiles & Layered Rule Merge Engine + Unit Tests
- [Phase 1.3](./phase-1/1.3-explain-and-report.md) — `--explain` / `--show-config` CLI & Phase 1 Colab Verification Report

### Phase 2: Wrapper Core, Benchmark Harness & Baseline Migrate Tests ([Overview](./phase-2-wrapper-benchmark-and-migrate-tests.md))
- [Phase 2.1](./phase-2/2.1-cli-and-hook-injection.md) — CLI Launcher & `sitecustomize` Hook Injection + Idempotency Guards
- [Phase 2.2](./phase-2/2.2-benchmark-workloads.md) — Representative Pure PyTorch Workloads & Edge-Case Scripts (`pipline-for-test/workloads/`)
- [Phase 2.3](./phase-2/2.3-telemetry-and-bench-harness.md) — Non-Invasive Telemetry, A/B Runner (`bench.py`) & Detailed Text Report (`report.py`)
- [Phase 2.4](./phase-2/2.4-baseline-migrate-tests.md) — Baseline Migrate Tests (Unmodified Script + Checkpoint Cross-Load)

### Phase 3: [Dedicated] Optimization Lab ([Overview](./phase-3-optimization-lab.md))
- [Phase 3.1](./phase-3/3.1-runtime-and-memory.md) — Track 3A.1: CUDA Flags, Memory Allocator & `channels_last` / CPU Threads
- [Phase 3.2](./phase-3/3.2-dataloader-autotune.md) — Track 3A.2: DataLoader Auto-Tuning & Colab `/dev/shm` + Pickle Safety Guard
- [Phase 3.3](./phase-3/3.3-fused-optimizer-and-amp.md) — Track 3B.1: Fused Optimizers & Safe Auto-AMP (`bf16` / `fp16 + GradScaler`)
- [Phase 3.4](./phase-3/3.4-scale-bottleneck-profiling.md) — Track 3B.0: Scale-Up Bottleneck Profiling (Small vs Medium vs Large) & Deep Debug Profiler
- [Phase 3.5](./phase-3/3.5-torch-compile.md) — Track 3B.2: Auto `torch.compile`, Persistent Cache & `_orig_mod.` Checkpoint Transparency
- [Phase 3.6](./phase-3/3.6-inference-running.md) — Track 3C: Inference / Running Optimization (`inference_mode`, CUDA Graphs, Quant/FP8)
- [Phase 3.7](./phase-3/3.7-custom-kernels.md) — Track 3D: Custom Kernels (Triton / CUDA), Capability Gate & Final Profile Consolidation

### Phase 4: Hardening, Full Compatibility Matrix & Regression Gate ([Overview](./phase-4-hardening-and-regression-gate.md))
- [Phase 4.1](./phase-4/4.1-full-migrate-matrix.md) — Full 3-Axis Migrate Test Matrix (Code, Checkpoint, Config/Cache Migration)
- [Phase 4.2](./phase-4/4.2-diagnostics-and-docs.md) — `--doctor`, `--fallback-report` & Developer Docs
- [Phase 4.3](./phase-4/4.3-regression-gate.md) — Automated Regression Gate & Final Colab Sign-off Report

### Phase 5: Group 1 Kernel & Compile Refinements ([Overview](./phase-5-group1-kernel-and-compile-refinements.md))

### Phase 6: Runtime Fusion & LLM Scale ([Overview](./phase-6-runtime-fusion-and-llm-scale.md))

### Phase 7: Hardening, Calibration Integrity & Namespace Isolation ([Overview](./phase-7-hardening-calibration-and-namespace.md))
- **7.1** Telemetry & FP8 Static Calibration Integrity (`inference_opt.py`, `manager.py`, `telemetry_hook.py`)
- **7.2** Package Namespace Isolation (`src/` -> `ptopt/`)
- **7.3** Hybrid LCE Head Guard, `LazyLinearLogits` Dynamo Boundary, Compile/OOM & Activation Checkpointing
- **7.4** Core Hardening (`checkpoint_compat`, `channels_last`, `auto_amp`, `module_dispatch`, `rope`, `fingerprint`, `cli`, docs)
- **7.5** Mandatory 7-Part Deep Debug Verification Report (`scripts/verify_phase7.py`)

### Phase 8: VRAM-Headroom Bandwidth Booster & Detailed Telemetry ([Overview](./phase-8-vram-headroom-bandwidth-booster.md))
- **8.1** Adaptive Multi-Batch GPU Prefetch Queue + Per-Batch `cuda.Event` Sync & VRAM Back-Off (`dataloader_tune.py`)
- **8.2** Opt-In `--bandwidth-boost` Flag (`default=False`) & High-VRAM Rule (`vram_bandwidth_boost` in `resolver.py` / `cli/main.py` / `schema.py`)
- **8.3** High-VRAM Inductor Bandwidth Autotuning (`coordinate_descent_tuning` in `auto_compile.py`, `max-autotune` + `bandwidth_boost`)
- **8.4** VRAM-Headroom-Aware Inference FP8 Resident Cache & `M >= 128` Prewarm (`inference_opt.py`)
- **8.5** Detailed Bandwidth/Prefetch Telemetry & Verification (`telemetry_hook.py`, `real_bench.py`, `scripts/verify_phase8.py`)

---

## โครงสร้างโฟลเดอร์เป้าหมาย (Target Repository Layout)

```text
kernel-pytouch-optimize/
├── ai-docs-dev/                 # แผนงานราย Phase/Sub-Phase, Architecture Specs, Experiment Logs
│   ├── phase-1/
│   ├── phase-2/
│   ├── phase-3/
│   ├── phase-4/
│   ├── phase-5/
│   ├── phase-6/
│   └── phase-7-hardening-calibration-and-namespace.md
├── ptopt/                       # Wrapper Core, Detector, Config Engine, Passes, Kernels
├── pipline-for-test/            # Benchmark Harness, Workloads, Telemetry, Text Report Generator
├── report/                      # เก็บ Text Reports ทั้งหมด (ชื่อไฟล์ HHMMSS_YYYYMMDD-name.txt)
├── scripts/                     # Utility scripts สำหรับรันตรวจและสร้าง Text Report บน Google Colab
└── tests/
    ├── unit/                    # Unit tests
    └── migrate/                 # Migrate tests (Code, Checkpoint, Config migration)
```

---

## กฎกลางของทุก Phase และ Sub-Phase (Mandatory Rules)

1. **Google Colab First**: ทุกคำสั่งทดสอบและสคริปต์ต้องรันบน Google Colab ได้สะดวก (รองรับ notebook shell `!python ...`, ephemeral disk, GPU T4/L4/A100 และ `/dev/shm` ขนาดจำกัดบน Colab)
2. **Text Report First**: ทุก Phase/Sub-Phase ต้องมี script ที่ generate **Detailed Text Report (`.txt`)** เก็บในโฟลเดอร์ `report/` ทั้งหมด โดยตั้งชื่อไฟล์ตามรูปแบบ `HHMMSS_YYYYMMDD-name.txt` (เช่น `report/150810_20261117-phase-1.1.txt`) และพิมพ์ออกหน้าจอได้
3. **Human Verification Gate (รอตรวจสอบเอง)**: ห้ามสรุปว่าจบ Phase/Sub-Phase เอง — หลังสร้างโค้ด/เทสเสร็จในแต่ละขั้น ต้อง **รอให้ผู้ใช้ไปรันบน Google Colab, อ่าน Text Report, และยืนยันด้วยตนเอง** ก่อนเริ่มขั้นถัดไป
