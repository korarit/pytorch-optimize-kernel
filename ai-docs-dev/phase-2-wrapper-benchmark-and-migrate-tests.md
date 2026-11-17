# Phase 2: Zero-Code Wrapper Core, Benchmark Harness & Baseline Migrate Tests (Overview)

## เป้าหมาย (Goal)
สร้างท่อ **CLI Wrapper + Injection Hook** ที่รันโค้ด PyTorch เดิมได้โดยไม่แก้โค้ด พร้อมสร้าง **Benchmark Harness (`pipline-for-test/`)** และ **Baseline Migrate Test Suite** เพื่อยืนยันความเสถียรก่อนเริ่มทดลอง optimize ใน Phase 3

---

## Sub-Phases

| Sub-Phase | ชื่อ | ไฟล์รายละเอียด | Report เป้าหมาย |
| :--- | :--- | :--- | :--- |
| **2.1** | CLI Launcher & Hook Injection | [2.1-cli-and-hook-injection.md](./phase-2/2.1-cli-and-hook-injection.md) | `report/HHMMSS_YYYYMMDD-phase-2.1.txt` |
| **2.2** | Pure PyTorch Workloads & Edge Cases | [2.2-benchmark-workloads.md](./phase-2/2.2-benchmark-workloads.md) | `report/HHMMSS_YYYYMMDD-phase-2.2.txt` |
| **2.3** | Telemetry & A/B Benchmark Harness | [2.3-telemetry-and-bench-harness.md](./phase-2/2.3-telemetry-and-bench-harness.md) | `report/HHMMSS_YYYYMMDD-phase-2.3.txt` |
| **2.4** | Baseline Migrate Tests (Script & Checkpoint) | [2.4-baseline-migrate-tests.md](./phase-2/2.4-baseline-migrate-tests.md) | `report/HHMMSS_YYYYMMDD-phase-2.txt` |

---

## Exit Criteria รวมของ Phase 2
- [x] Sub-Phase 2.1 – 2.4 ผ่านครบทุกข้อ (ส่วน implementation & dev tests)
- [x] รันทุก workload ใน `pipline-for-test/` ผ่าน wrapper โดยไม่แก้โค้ด workload
- [x] Unit tests + Migrate tests (Script + Checkpoint cross-load) ผ่าน 100% บนเครื่อง dev
- [ ] **[WAIT FOR USER]** ผู้ใช้ตรวจสอบ **Detailed Text Report (`report/HHMMSS_YYYYMMDD-phase-2.txt`)** บน Google Colab และยืนยันผลด้วยตนเองก่อนเริ่ม Phase 3
