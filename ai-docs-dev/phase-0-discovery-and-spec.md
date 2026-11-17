# Phase 0: Environment Discovery & Design Spec

## เป้าหมาย (Goal)
สำรวจสภาพแวดล้อมเครื่องพัฒนาจริง และตกลงมาตรฐานกลาง (Config Schema, Benchmark Rules, Numerical Parity Thresholds) ก่อนเริ่มเขียนโค้ดใน Phase 1

---

## งานที่ต้องทำ (Tasks)

### 0.1 Environment & Hardware Discovery (เน้น Google Colab)
- [ ] สร้าง script สำหรับรันบน **Google Colab** เพื่อตรวจข้อมูล GPU (`T4`, `L4`, `A100` ฯลฯ), Compute Capability (`sm_XX`), VRAM, driver, CUDA version, `bf16` support, FP8 support
- [ ] ตรวจสอบข้อมูลเครื่อง (Host): OS, CPU architecture (`x86_64` / `aarch64`), physical/logical cores, System RAM, `/dev/shm` size (Colab มักจำกัด `/dev/shm` ซึ่งสำคัญต่อ `DataLoader`)
- [ ] ตรวจสอบ Python Stack: Python version, PyTorch version, Triton availability, allocator libs (`jemalloc` / `tcmalloc`)
- [ ] ให้ script สร้าง **Text Report** เก็บลง `report/HHMMSS_YYYYMMDD-phase-0-env.txt` และเก็บ snapshot ลง `ai-docs-dev/`

### 0.2 Config Schema v1 Specification
- [ ] กำหนด schema version (`schema_version: 1`)
- [ ] กำหนดหมวดหมู่ค่า config:
  - `runtime`: `tf32`, `cudnn_benchmark`, `alloc_conf`, `omp_threads`, `jemalloc`
  - `amp`: `enabled`, `dtype` (`bf16`/`fp16`/`fp32`), `grad_scaler`
  - `compile`: `enabled`, `mode` (`default`/`reduce-overhead`/`max-autotune`), `dynamic`, `cache_dir`
  - `optimizer`: `fused`, `foreach`
  - `attention`: `flash_sdp`, `mem_efficient_sdp`, `math_sdp`
  - `dataloader`: `auto_tune`, `pin_memory`, `persistent_workers`, `num_workers`, `prefetch_factor`
  - `inference`: `inference_mode`, `cuda_graphs`
  - `kernels`: `enabled`, `allow_list`, `fallback_on_error`
- [ ] กำหนดลำดับความสำคัญของการ Merge Config (`base -> host_arch -> gpu_arch -> vram_tier -> profile -> CLI/ENV`)

### 0.3 Parity & Benchmark Standard
- [ ] กำหนดเกณฑ์ Numerical Parity:
  - `safe`: loss relative diff เล็กน้อยมาก (TF32 tolerance) และไม่มี `NaN`/`Inf`
  - `balanced` / `aggressive`: loss trajectory ใกล้เคียง baseline ภายในเกณฑ์ `bf16`/`fp16` tolerance และ final grad norm ไม่ explode
- [ ] กำหนดรูปแบบ JSON output ของ Telemetry และ Benchmark Report

### 0.4 Text Report Specification
- [ ] กำหนดมาตรฐานกลางของ **Text Report Generator** (`report/HHMMSS_YYYYMMDD-name.txt` + stdout) ที่ทุก Phase จะใช้ร่วมกัน:
  - ตำแหน่งจัดเก็บ: โฟลเดอร์ `report/` เท่านั้น และใช้รูปแบบชื่อไฟล์ `HHMMSS_YYYYMMDD-<name>.txt` (เวลา-วันที่ท้องถิ่นที่รัน)
  - Header (Phase, Timestamp, Git/Project Version, Colab/Host tag)
  - Hardware & Environment Summary
  - Resolved Config & Rule Trace
  - Test Summary (Unit + Migrate + Kernel Parity + Failure logs)
  - Benchmark Summary Table (Baseline vs Wrapped, Cold vs Warm)
  - Verification Status: `PENDING_USER_REVIEW`

---

## Deliverables
- Script ตรวจ environment + generate detailed text report (`report/HHMMSS_YYYYMMDD-phase-0-env.txt`) สำหรับรันบน Google Colab
- `ai-docs-dev/env-baseline.md` (หรือ snapshot ของเครื่อง Colab เป้าหมาย)
- Config Schema v1 & Text Report Format specification

---

## Exit Criteria & Human Verification Gate (เกณฑ์ผ่าน Phase 0)
- [ ] มี script ตรวจ environment ที่รันบน Google Colab แล้วได้ **Detailed Text Report (`report/HHMMSS_YYYYMMDD-phase-0-env.txt`)** ออกมาอ่านได้ครบถ้วน
- [ ] ตกลง Config Schema v1, Text Report Format และ Parity Thresholds เรียบร้อยแล้ว
- [ ] **[WAIT FOR USER]** รอผู้ใช้รันตรวจสอบบน Google Colab และยืนยันผลด้วยตนเองก่อนเริ่ม Phase 1
