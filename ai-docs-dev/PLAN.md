# Project Plan: Zero-Code PyTorch Optimization Wrapper (`kernel-pytouch-optimize`)

## เป้าหมายหลัก (Objective)

สร้างระบบ **Zero-Code PyTorch Optimization Wrapper** ที่นำไปครอบคำสั่งรันโค้ด PyTorch เดิม (ทั้งการ **Training** และ **Running / Inference**) เพื่อเพิ่มความเร็วและลดการใช้ VRAM **โดยไม่ต้องแก้ไขโค้ดเทรนของผู้ใช้** พร้อมระบบ:

1. **Hardware-Aware Config Resolution**: ปรับ config อัตโนมัติตาม GPU Architecture และ Machine Architecture (รองรับการรันบน **Google Colab** เช่น T4 `sm_75`, L4 `sm_89`, A100 `sm_80`, รวมถึง CPU/container `/dev/shm` ที่มีข้อจำกัด)
2. **Modular Optimization Passes**: แยกเทคนิคการ optimize เป็นโมดูลอิสระ พร้อม capability gate และ auto-fallback
3. **Dedicated Optimization Lab**: แยก phase ทดลองและวัดผล A/B ออกจากโครงสร้างหลัก
4. **Unit Tests & Migrate Tests**: คุ้มครองทั้ง logic ภายใน, การรันโค้ดเดิมโดยไม่แก้, และการโหลด checkpoint ข้ามโหมด
5. **Detailed Text Report & Human Verification Gate (Google Colab)**: ทุก Phase ต้องมี script สร้าง **Detailed Text Report (`.txt` / stdout)** และ **ต้องรอให้ผู้ใช้เอาไปรันบน Google Colab และตรวจสอบยืนยันด้วยตนเองก่อน** จึงจะถือว่าผ่าน Phase และไปต่อได้

---

## ดัชนีแผนงานราย Phase (1 File = 1 Phase)

| Phase | ชื่อ Phase | ไฟล์รายละเอียด | จุดเน้นหลัก |
| :--- | :--- | :--- | :--- |
| **Phase 0** | Environment Discovery & Design Spec | [phase-0-discovery-and-spec.md](./phase-0-discovery-and-spec.md) | ตรวจสเปกเครื่อง, กำหนด Config Schema v1, เกณฑ์ Parity & Benchmark |
| **Phase 1** | Hardware Fingerprint & Layered Config | [phase-1-hardware-config-and-unit-tests.md](./phase-1-hardware-config-and-unit-tests.md) | ตรวจจับ GPU/Host Arch, Rule Engine, `--explain`, **Unit Tests (Mock Matrix)** |
| **Phase 2** | Wrapper Core, Benchmark & Migrate Tests | [phase-2-wrapper-benchmark-and-migrate-tests.md](./phase-2-wrapper-benchmark-and-migrate-tests.md) | CLI + Hook Injection, Benchmark Harness (`pipline-for-test/`), **Migrate Tests ตั้งต้น** |
| **Phase 3** | **[Dedicated]** Optimization Lab | [phase-3-optimization-lab.md](./phase-3-optimization-lab.md) | ทดลอง Track 3A–3D (Runtime, Compile/AMP, Inference, Triton Kernels) + Promotion Gate |
| **Phase 4** | Hardening & Full Regression Gate | [phase-4-hardening-and-regression-gate.md](./phase-4-hardening-and-regression-gate.md) | Full Unit + Migrate Matrix (3 มิติ), Config Migration, `--doctor`, CI Gate |

---

## โครงสร้างโฟลเดอร์เป้าหมาย (Target Repository Layout)

```text
kernel-pytouch-optimize/
├── ai-docs-dev/                 # แผนงานราย Phase, Architecture Specs, Benchmark Logs
├── src/                         # Wrapper Core, Detector, Config Engine, Passes, Kernels
├── pipline-for-test/            # Benchmark Harness, Workloads, Telemetry, Text Report Generator
├── report/                      # เก็บ Text Reports ทั้งหมด (ชื่อไฟล์ HHMMSS_YYYYMMDD-name.txt)
├── scripts/                     # Utility scripts (เช่น gen_report.py, colab_verify.py)
└── tests/
    ├── unit/                    # Unit tests
    └── migrate/                 # Migrate tests (Code, Checkpoint, Config migration)
```

---

## กฎกลางของทุก Phase (Mandatory Rules)

1. **Google Colab First**: ทุกคำสั่งทดสอบและสคริปต์ต้องรันบน Google Colab ได้สะดวก (รองรับ notebook shell `!python ...`, ephemeral disk, GPU T4/L4/A100 และ `/dev/shm` ขนาดจำกัดบน Colab)
2. **Text Report First**: ทุก Phase ต้องมี script ที่ generate **Detailed Text Report (`.txt`)** เก็บในโฟลเดอร์ `report/` ทั้งหมด โดยตั้งชื่อไฟล์ตามรูปแบบ `HHMMSS_YYYYMMDD-name.txt` (เช่น `report/150810_20261117-phase-0-env.txt`) และพิมพ์ออกหน้าจอได้ แสดงรายละเอียดครบทั้ง Hardware, Config Trace, Test Results, Benchmark Metrics และ Fallback Reasons
3. **Human Verification Gate (รอตรวจสอบเอง)**: ห้ามสรุปว่าจบ Phase เอง — หลังสร้างโค้ด/เทสเสร็จในแต่ละ Phase ต้อง **รอให้ผู้ใช้ไปรันบน Google Colab, อ่าน Text Report, และยืนยันด้วยตนเอง** ก่อนเริ่ม Phase ถัดไป
