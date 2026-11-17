# Phase 1: Hardware Fingerprinting & Layered Config System (Overview)

## เป้าหมาย (Goal)
สร้างระบบตรวจจับ **GPU Architecture** และ **Machine/Host Architecture** พร้อม **Layered Rule Engine** ที่เปลี่ยน Config ตามสเปกเครื่องได้อัตโนมัติ โปร่งใส และทดสอบได้แบบ Deterministic

---

## Sub-Phases

| Sub-Phase | ชื่อ | ไฟล์รายละเอียด | Report เป้าหมาย |
| :--- | :--- | :--- | :--- |
| **1.1** | Hardware & Runtime Probe | [1.1-hardware-probe.md](./phase-1/1.1-hardware-probe.md) | `report/HHMMSS_YYYYMMDD-phase-1.1.txt` |
| **1.2** | Config Schema v1 & Rule Engine | [1.2-config-rule-engine.md](./phase-1/1.2-config-rule-engine.md) | `report/HHMMSS_YYYYMMDD-phase-1.2.txt` |
| **1.3** | `--explain` CLI & Phase 1 Gate | [1.3-explain-and-report.md](./phase-1/1.3-explain-and-report.md) | `report/HHMMSS_YYYYMMDD-phase-1.txt` |

---

## Exit Criteria รวมของ Phase 1
- [ ] Sub-Phase 1.1, 1.2, 1.3 ผ่านครบทุกข้อ
- [ ] `pytest tests/unit` ฝั่ง detector + config resolver ผ่าน 100%
- [ ] **[WAIT FOR USER]** ผู้ใช้ตรวจสอบ **Detailed Text Report (`report/HHMMSS_YYYYMMDD-phase-1.txt`)** บน Google Colab และยืนยันผลด้วยตนเองก่อนเริ่ม Phase 2
