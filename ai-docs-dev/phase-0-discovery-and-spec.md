# Phase 0: Environment Discovery & Design Spec

## เป้าหมาย (Goal)
สำรวจสภาพแวดล้อมเครื่องพัฒนาจริง และตกลงมาตรฐานกลาง (Config Schema, Benchmark Rules, Numerical Parity Thresholds) ก่อนเริ่มเขียนโค้ดใน Phase 1

---

## งานที่ต้องทำ (Tasks)

### 0.1 Environment & Hardware Discovery (เน้น Google Colab)
- [x] สร้าง script สำหรับรันบน **Google Colab** เพื่อตรวจข้อมูล GPU (`T4`, `L4`, `A100` ฯลฯ), Compute Capability (`sm_XX`), VRAM, driver, CUDA version, `bf16` support, FP8 support ([phase0_env_discovery.py](../scripts/phase0_env_discovery.py))
- [x] ตรวจสอบข้อมูลเครื่อง (Host): OS, CPU architecture (`x86_64` / `aarch64`), physical/logical cores, System RAM, `/dev/shm` size
- [x] ตรวจสอบ Python Stack: Python version, PyTorch version, Triton availability, allocator libs (`jemalloc` / `tcmalloc`)
- [x] ให้ script สร้าง **Text Report** เก็บลง `report/HHMMSS_YYYYMMDD-phase-0-env.txt` และเก็บ snapshot ลง `ai-docs-dev/env-baseline.md` (ทดสอบบนเครื่อง dev แล้ว; รอรันยืนยันบน Google Colab)

### 0.2 Config Schema v1 Specification
- [x] กำหนด schema version (`schema_version: 1`) ใน [config-schema-v1.md](./specs/config-schema-v1.md)
- [x] กำหนดหมวดหมู่ค่า config (`runtime`, `amp`, `compile`, `optimizer`, `attention`, `dataloader`, `inference`, `kernels`) และ cross-field validation
- [x] กำหนดลำดับความสำคัญของการ Merge Config (`base -> host_arch -> gpu_arch -> vram_tier -> profile -> CLI/ENV`)

### 0.3 Parity & Benchmark Standard
- [x] กำหนดเกณฑ์ Numerical Parity (`safe`, `balanced`, `aggressive`) ใน [parity-and-benchmark-spec.md](./specs/parity-and-benchmark-spec.md)
- [x] กำหนดรูปแบบ JSON output ของ Telemetry (`telemetry_schema_version: 1`) และกฎการวัด Benchmark

### 0.4 Text Report Specification
- [x] กำหนดมาตรฐานกลางของ **Text Report Generator** (`report/HHMMSS_YYYYMMDD-name.txt` + stdout) ที่ทุก Phase จะใช้ร่วมกันใน [parity-and-benchmark-spec.md](./specs/parity-and-benchmark-spec.md)

---

## Deliverables
- [phase0_env_discovery.py](../scripts/phase0_env_discovery.py) — Script ตรวจ environment + generate detailed text report (`report/HHMMSS_YYYYMMDD-phase-0-env.txt`) สำหรับรันบน Google Colab
- [env-baseline.md](./env-baseline.md) — Environment snapshot ของเครื่อง dev (และอัปเดตเมื่อรัน `--save-baseline` บน Colab)
- [config-schema-v1.md](./specs/config-schema-v1.md) — Config Schema v1 specification
- [parity-and-benchmark-spec.md](./specs/parity-and-benchmark-spec.md) — Parity, Benchmark & Text Report specification

---

## Exit Criteria & Human Verification Gate (เกณฑ์ผ่าน Phase 0)
- [x] มี script ตรวจ environment ที่พร้อมรันบน Google Colab แล้วได้ **Detailed Text Report (`report/HHMMSS_YYYYMMDD-phase-0-env.txt`)** ออกมาอ่านได้ครบถ้วน
- [x] ตกลง Config Schema v1, Text Report Format และ Parity Thresholds เรียบร้อยแล้ว
- [x] **[VERIFIED BY USER]** ผู้ใช้รันตรวจสอบบน Google Colab (NVIDIA L4 `sm_89`, PyTorch `2.14.1+cu130`, Triton `3.8.0`) แล้วใน [085558_20261117-phase-0-env.txt](../report/085558_20261117-phase-0-env.txt)
