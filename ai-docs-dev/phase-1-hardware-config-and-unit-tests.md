# Phase 1: Hardware Fingerprinting & Layered Config System

## เป้าหมาย (Goal)
สร้างโมดูลตรวจจับ **GPU Architecture** และ **Machine/Host Architecture** พร้อมระบบ **Layered Rule Engine** ที่เปลี่ยนค่า Config ตามสเปกเครื่องได้อัตโนมัติ โปร่งใส และทดสอบได้แบบ Deterministic

---

## ขอบเขตงาน (Scope & Modules)

### 1.1 Hardware & Runtime Fingerprinting (`src/detect/`)
- [ ] **GPU Probe**:
  - Detect backend: `cuda` / `rocm` / `mps` / `cpu`
  - Detect device properties: `name`, `compute_capability` (`major`, `minor`, `sm_XX`), `total_vram_mb`, `sm_count`
  - Detect feature flags: `bf16_supported`, `tf32_supported`, `fp8_supported`, `flash_sdp_supported`
  - รองรับเครื่อง Multi-GPU (อ่าน target device หรือ GPU rank ปัจจุบัน) และเครื่องไม่มี GPU (CPU fallback)
- [ ] **Machine / Host Probe**:
  - `os_name`, `cpu_arch` (`x86_64`, `aarch64`, `arm64`)
  - `physical_cores`, `logical_cores`, `total_ram_mb`, `shm_size_mb` (`/dev/shm`)
  - `torch_version`, `cuda_version`, `triton_available`, `allocator_libs_available`
- [ ] ออกแบบ `HardwareFingerprint` dataclass ที่ serialize เป็น JSON ได้

### 1.2 Layered Config Resolver (`src/config/`)
- [ ] สร้าง Rule Matching & Merge Engine ตามลำดับ:
  1. `base` (safe defaults)
  2. `host_arch` rules (เช่น `low_shm_guard`, `macos_mps`, `linux_x86_64`, `linux_aarch64`)
  3. `gpu_arch` rules (เช่น `cuda_turing_sm75`, `cuda_ampere_sm80_86`, `cuda_ada_sm89`, `cuda_hopper_sm90`, `cpu_only`)
  4. `vram_tier` rules (เช่น `vram_lt_12gb`, `vram_12_24gb`, `vram_ge_40gb`)
  5. `user_profile` (`safe` / `balanced` / `aggressive`)
  6. `CLI flags / ENV overrides` (ชนะทุกชั้นเสมอ)
- [ ] เก็บ **Resolution Trace** (บันทึกว่า rule ไหน match และเปลี่ยน field ไหนบ้าง)
- [ ] สร้างคำสั่ง/โหมด `--explain` / `--show-config` แสดงผลลัพธ์ให้อ่านง่าย + export JSON snapshot ได้
- [ ] เพิ่ม Schema Validation (ปฏิเสธ config key ที่ไม่รู้จัก หรือค่าที่ขัดกัน พร้อมข้อความ error ชัดเจน)

---

## แผนการทดสอบ (Testing Plan)

### Unit Tests (`tests/unit/`)
- [ ] **Mock Hardware Matrix**: จำลอง `HardwareFingerprint` โดยไม่ต้องมี GPU ครบทุกรุ่น:
  - Turing (`sm_75`, 16GB) → ต้องได้ `tf32=False`, `amp_dtype=fp16`, `grad_scaler=True`, `flash_sdp=False`
  - Ampere (`sm_80`, 80GB) → ต้องได้ `tf32=True`, `amp_dtype=bf16`, `grad_scaler=False`, `flash_sdp=True`
  - Consumer Ada (`sm_89`, 8GB) → เปิด `bf16`/`tf32` แต่ `vram_lt_12gb` rule ต้องปรับ allocator/compile ให้ประหยัด memory
  - Hopper (`sm_90`, 80GB) → เปิด capability สำหรับ `fp8`/`max-autotune` ใน profile ที่เหมาะสม
  - Apple Silicon (`mps`, `arm64`) → ปิด CUDA flags ทั้งหมด
  - CPU-only (`x86_64` vs `aarch64`) → ปรับ thread count และปิด GPU features
  - Container `/dev/shm` เล็ก (`< 2GB`) → ต้องปิด `dataloader.auto_tune` หรือจำกัด `num_workers`
- [ ] **Precedence & Override Tests**: ตรวจว่า CLI/ENV override ชนะทุก rule และ profile merging ไม่ทำ field อื่นหาย
- [ ] **Schema Validation Tests**: ทดสอบ invalid values, unknown keys, missing required fields

---

### 1.3 Detailed Text Report Script (`scripts/gen_phase1_report.py` / shared reporter)
- [ ] สร้าง script ที่รันตรวจจับฮาร์ดแวร์จริง (เช่น บน Google Colab T4/L4/A100) + รัน `pytest tests/unit` แล้ว generate **Detailed Text Report** ลงใน `report/HHMMSS_YYYYMMDD-phase-1.txt`
- [ ] เนื้อหาใน Text Report ต้องแสดงชัดเจน:
  - Hardware Fingerprint จริงของเครื่องที่รัน (GPU, `sm_XX`, VRAM, CPU, `/dev/shm`, PyTorch/CUDA/Triton)
  - Rule Trace ของทุก profile (`safe`, `balanced`, `aggressive`) ว่า rule ไหน match และตั้งค่าอะไรบ้าง
  - ผลรัน Unit Tests ทั้งหมด (จำนวน PASS/FAIL/SKIP + error traceback ถ้ามี)

---

## Deliverables
- `src/detect/` (GPU & Host probe)
- `src/config/` (Rule files, Merger, Validator, `--explain`)
- `tests/unit/test_detect.py`, `tests/unit/test_config_resolver.py`
- Script สร้าง Detailed Text Report ลง `report/HHMMSS_YYYYMMDD-phase-1.txt` (พร้อมรันบน Google Colab)

---

## Exit Criteria & Human Verification Gate (เกณฑ์ผ่าน Phase 1)
- [ ] `pytest tests/unit` ผ่าน 100% (ทั้งโหมด Mock และรันบนเครื่องจริง)
- [ ] `--explain` และ Text Report Generator (`report/HHMMSS_YYYYMMDD-phase-1.txt`) ทำงานสมบูรณ์บน Google Colab
- [ ] **[WAIT FOR USER]** รอผู้ใช้รันบน Google Colab, ตรวจ **Detailed Text Report (`report/HHMMSS_YYYYMMDD-phase-1.txt`)**, และยืนยันด้วยตนเองก่อนเริ่ม Phase 2
