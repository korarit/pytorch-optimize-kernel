# Phase 2: Zero-Code Wrapper Skeleton, Benchmark Harness & Baseline Migrate Tests

## เป้าหมาย (Goal)
สร้างโครงสร้าง **CLI Wrapper + Injection Hook** ที่รันโค้ด PyTorch เดิมได้โดยไม่ต้องแก้โค้ด พร้อม **Benchmark Harness** ใน `pipline-for-test/` และชุด **Migrate Tests** ตั้งต้น

> หมายเหตุ: ใน Phase นี้ยังไม่เปิด optimization ที่เสี่ยงสูง จะเน้นความเสถียรของท่อ (plumbing), telemetry, และความเข้ากันได้กับโค้ดเดิมเป็นหลัก

---

## ขอบเขตงาน (Scope & Modules)

### 2.1 Zero-Code CLI Launcher & Injection Engine (`src/cli`, `src/hooks`)
- [ ] สร้าง CLI entrypoint (เช่น `ptopt --profile safe -- python train.py ...`)
- [ ] สร้างกลไก inject hook อัตโนมัติ (ผ่าน `PYTHONPATH` + `sitecustomize.py` หรือ `runpy` wrapper) เพื่อให้ subprocess / worker processes ที่ถูก spawn (เช่น DataLoader workers หรือ DDP) รับรู้ config ได้ด้วย
- [ ] **Idempotency & Safety Guards**:
  - ป้องกันการ patch ซ้ำซ้อนถ้า hook ถูกโหลดมากกว่า 1 ครั้ง
  - ตรวจจับหาก user script เปิด `autocast`, `GradScaler`, หรือ `torch.compile` ไว้เองแล้ว เพื่อไม่ให้ซ้อนทับจนพัง
  - รองรับ `try/except` fallback + warning log เมื่อ hook บางตัวใช้ไม่ได้กับ script เป้าหมาย

### 2.2 Benchmark Harness (`pipline-for-test/`)
- [ ] สร้าง **Workloads (Pure PyTorch Scripts — ห้ามใส่โค้ดเฉพาะของ wrapper)**:
  - `cnn_vision.py`: ResNet-style CNN + CPU transform (จำลอง DataLoader/CPU bottleneck)
  - `transformer_lm.py`: Tiny-GPT / Transformer blocks (จำลอง Compute/VRAM/Attention bottleneck)
  - `deep_mlp.py`: MLP ที่มี layer เล็กจำนวนมาก (จำลอง Kernel launch / Optimizer overhead)
  - `infer_runner.py`: Inference benchmark (Batch=1 latency & Batch=N throughput)
  - `edge_cases.py`: สคริปต์ที่มี `autocast` อยู่แล้ว, dynamic input shapes, และ unpicklable Dataset
- [ ] สร้าง **Telemetry Collector**:
  - วัด Step time (median, p95, min, std) ด้วย `torch.cuda.synchronize()` / `torch.cuda.Event`
  - แยกวัด Cold-start / Warmup steps vs Steady-state steps
  - เก็บ `Peak Allocated VRAM`, `Peak Reserved VRAM`, `End-to-End Wall Time`, `Loss Trajectory`
  - บันทึก `hardware_fingerprint` + `resolved_config` แนบไปในผล JSON ทุกครั้ง
- [ ] สร้าง **A/B Runner (`bench.py`) & Detailed Text Report Generator (`report.py`)**:
  - รัน Baseline vs Wrapped แยก subprocess เสมอ (รองรับการสั่งรันจาก Google Colab cell ได้ในคำสั่งเดียว)
  - สร้าง **Detailed Text Report (`.txt` เป็นหลัก + stdout summary)** เปรียบเทียบทุก workload: Cold vs Warm, Step time (median/p95/min/std), End-to-End wall time, Speedup, Peak VRAM (allocated/reserved), Max Loss Diff, Fallback/Skip reasons และ Status (`PASS` / `FAIL`)
  - *(optional)* รองรับ export JSON/CSV คู่กันเพื่อเก็บประวัติย้อนหลัง

---

## แผนการทดสอบ (Testing Plan)

### Unit Tests (`tests/unit/`)
- [ ] ทดสอบ CLI argument parsing และการส่งต่อ command args ของ user script (เช่น flags, stdin, exit code)
- [ ] ทดสอบ Hook idempotency (เรียก init ซ้ำไม่ wrap ซ้ำ)
- [ ] ทดสอบ Telemetry collector JSON schema, Text Report formatter และการคำนวณ metrics (median, p95, speedup)

### Migrate Tests (`tests/migrate/`) — ชุดตั้งต้น
- [ ] **Test Group 1: Unmodified Script Migration**:
  - รัน workload scripts ทั้งหมด (รวม `edge_cases.py`) ผ่าน wrapper โดยไม่แก้โค้ดต้นฉบับ → ต้อง exit code `0`, ไม่มี crash, loss อยู่ใน parity threshold
- [ ] **Test Group 2: Checkpoint Cross-Compatibility (Baseline ↔ Wrapped)**:
  - เทรนด้วย Baseline → `torch.save(model.state_dict())` → โหลดมารันต่อใน Wrapped → ต้องโหลดได้โดยไม่ error และ loss ต่อเนื่อง
  - เทรนด้วย Wrapped → `torch.save(model.state_dict())` → โหลดกลับมาใช้ใน PyTorch ปกติ (Unwrapped) → ต้องโหลดได้ไม่มี prefix แปลกปลอม (เช่น `_orig_mod.`) ตกค้างจนโหลดพัง

---

## Deliverables
- `src/cli/`, `src/hooks/`
- `pipline-for-test/` (`workloads/`, `bench.py`, `telemetry.py`, `report.py`)
- `tests/unit/test_cli_hooks.py`, `tests/migrate/test_script_migration.py`, `tests/migrate/test_checkpoint_compat.py`
- Script รัน Unit + Migrate + Baseline Benchmark และออก **Detailed Text Report** ลง `report/HHMMSS_YYYYMMDD-phase-2.txt` บน Google Colab

---

## Exit Criteria & Human Verification Gate (เกณฑ์ผ่าน Phase 2)
- [ ] สามารถสั่งครอบรันทุก workload ใน `pipline-for-test/` บน Google Colab ได้โดยไม่แก้โค้ด workload
- [ ] `bench.py` + `report.py` รัน Baseline vs Wrapped และออก **Detailed Text Report** ใน `report/HHMMSS_YYYYMMDD-phase-2.txt` ได้ครบถ้วน
- [ ] Unit tests และ Migrate tests (Script + Checkpoint cross-load) ผ่าน 100%
- [ ] **[WAIT FOR USER]** รอผู้ใช้รันบน Google Colab, ตรวจ **Detailed Text Report (`report/HHMMSS_YYYYMMDD-phase-2.txt`)**, และยืนยันด้วยตนเองก่อนเริ่ม Phase 3
