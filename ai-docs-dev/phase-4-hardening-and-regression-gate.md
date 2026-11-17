# Phase 4: Hardening, Full Compatibility Matrix & Regression Gate

## เป้าหมาย (Goal)
รวมทุกส่วนให้พร้อมใช้งานจริง เชื่อถือได้ระยะยาว และมีระบบทดสอบอัตโนมัติ (Regression Gate) ทั้ง **Unit Tests** และ **Migrate Tests ครบ 3 มิติ** เพื่อป้องกัน bug ย้อนกลับเมื่อเพิ่ม kernel หรือ rule ใหม่ในอนาคต

---

## ขอบเขตงาน (Scope)

### 4.1 Full Migrate Test Suite (`tests/migrate/`) — ครบ 3 มิติ
1. **Code Migration Tests (Unmodified & Legacy Scripts)**:
   - สคริปต์สไตล์ PyTorch 1.x / 2.x (เช่น `torch.cuda.amp.autocast` แบบเก่า vs `torch.amp.autocast`)
   - สคริปต์ที่มี custom `autograd.Function`, dynamic control flow, gradient accumulation, และ multi-optimizer
   - สคริปต์ DataLoader edge cases (unpicklable transform, `num_workers=0`, custom `collate_fn`)
   - สคริปต์ Inference ที่ไม่ใช้ `model.eval()` หรือเรียกซ้ำหลาย batch sizes
2. **Checkpoint & Artifact Migration Tests**:
   - Save `model.state_dict()` + `optimizer.state_dict()` จาก Baseline → Resume บน Wrapped (`safe`, `balanced`, `aggressive`)
   - Save จาก Wrapped (รวมกรณีเปิด `torch.compile` และ `channels_last`) → Load กลับมาใช้ใน PyTorch ปกติ
   - ตรวจว่า weight tensors ตรงกัน (dtype/shape/keys ไม่เพี้ยน)
3. **Config & Cache Schema Migration Tests**:
   - ทดสอบโหลด config schema เวอร์ชันเก่าบนระบบเวอร์ชันใหม่ (มี auto-migrator หรือ clear migration error message)
   - ทดสอบ invalidation / compatibility ของ compile cache เมื่อเปลี่ยน GPU architecture หรือ PyTorch version

### 4.2 Full Unit Test Suite (`tests/unit/`)
- [ ] ครอบคลุม Hardware Detector, Layered Config Resolver, Hook Guards, Fallback Logic, Telemetry, และ Custom Kernels
- [ ] แยกรันได้ทั้งโหมด **CPU-only / Mock** (รันบน CI ทั่วไปได้เร็ว) และ **GPU Suite** (รันบนเครื่องที่มี CUDA จริง)

### 4.3 Diagnostics, UX & Observability
- [ ] เพิ่มคำสั่ง `ptopt --doctor`:
  - ตรวจความพร้อมเครื่อง (CUDA driver/runtime, Triton, `jemalloc`, `/dev/shm`, disk cache permissions)
  - แนะนำ profile ที่เหมาะสมกับ hardware ปัจจุบัน
- [ ] เพิ่ม `--fallback-report`: สรุปตอนจบการทำงานว่ามี pass ไหนถูก skip หรือ fallback พร้อมเหตุผล (เช่น "skipped Triton RMSNorm: GPU is sm_75 < min_sm 80")
- [ ] สร้างคู่มือการใช้งานและวิธีเขียน/เพิ่ม Optimization Pass หรือ Custom Kernel ใหม่ลงใน `ai-docs-dev/`

### 4.4 Automated Regression Gate & Final Text Report
- [ ] สร้างคำสั่งเดียวสำหรับรัน test gate (รองรับการรันใน Google Colab cell):
  - Fast Gate (`unit + cpu-safe migrate tests`)
  - Full GPU Gate (`unit + full migrate + smoke benchmark parity check`)
- [ ] สร้าง **Full Verification Text Report Generator (`report/HHMMSS_YYYYMMDD-phase-4.txt`)**:
  - สรุปครบทุกส่วนในไฟล์ text เดียว: Environment/Colab GPU Fingerprint, `--doctor` diagnostics, Unit Test Matrix, Migrate Test Matrix (3 มิติ), Benchmark Summary, และ Fallback Report

---

## Exit Criteria & Human Verification Gate (เกณฑ์ผ่าน Phase 4)
- [ ] Unit tests และ Migrate tests ทั้ง 3 มิติผ่าน 100%
- [ ] `--doctor`, `--explain`, และ `--fallback-report` ทำงานสมบูรณ์บน Google Colab
- [ ] มี script สร้าง **Detailed Text Report (`report/HHMMSS_YYYYMMDD-phase-4.txt`)** และเอกสารคู่มือครบถ้วน
- [ ] **[WAIT FOR USER]** รอผู้ใช้รัน Full Gate บน Google Colab, ตรวจ **Detailed Text Report (`report/HHMMSS_YYYYMMDD-phase-4.txt`)**, และยืนยันปิดโปรเจกต์ด้วยตนเอง
