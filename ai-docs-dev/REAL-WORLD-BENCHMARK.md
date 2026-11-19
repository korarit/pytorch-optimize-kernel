# Real-World GitHub Pipeline Benchmark Guide (`ptopt`)

คู่มือนี้อธิบายวิธีนำ **`kernel-pytorch-optimize` (`ptopt`)** ไปทดสอบ Benchmark กับ **Real Repositories และ Real Models จาก GitHub** แบบ **Zero-Code Modification** โดย **ห้ามนำซอร์สโค้ดของโปรเจกต์อื่นเข้ามาเก็บใน repo นี้** (clone แยกไว้นอกโปรเจกต์ เช่น `/content/external_repos` บน Google Colab หรือ `../external_repos` บนเครื่องตัวเอง)

---

## 1. สิ่งที่เตรียมไว้แล้วใน `kernel-pytorch-optimize`

1. **`ptopt --telemetry-only --telemetry-out <path>`** ([`ptopt/cli/main.py`](../ptopt/cli/main.py), [`ptopt/hooks/manager.py`](../ptopt/hooks/manager.py))
   - โหมด **Untouched Baseline Probe**: ปิดการรัน optimization passes ทั้งหมด, ไม่ตั้งค่า `PYTORCH_CUDA_ALLOC_CONF` / `TORCHINDUCTOR_CACHE_DIR`, และไม่ติดตั้ง `checkpoint_compat` hooks
   - ติดตั้งเฉพาะ [`TelemetryCollector`](../ptopt/hooks/telemetry_hook.py) เพื่อเก็บ `step_times_ms`, `train_step_times_ms`, `eval_step_times_ms`, `losses`, และ `peak_allocated_mb` ของ Baseline โดยไม่แก้โค้ด repo เป้าหมาย
2. **Real-Loop Safe `TelemetryCollector`** ([`ptopt/hooks/telemetry_hook.py`](../ptopt/hooks/telemetry_hook.py))
   - แยก `train_step_times_ms` กับ `eval_step_times_ms` ออกจากกัน (ลูป validation ระหว่าง epoch จะไม่ถูกนับรวมเข้าไปบิดเบือนเวลาของ train step ถัดไป)
   - กัน `DataLoader` worker subprocesses ไม่ให้เขียนทับไฟล์ telemetry ของ process หลัก และแยกไฟล์ตาม `.rank{RANK}` อัตโนมัติเมื่อรันผ่าน `torchrun`
3. **Generic External Benchmark Runner & Artifact Bundle** ([`pipline-for-test/real_bench.py`](../pipline-for-test/real_bench.py))
   - รันคำสั่งของ repo ภายนอกแบบ **4-Arm Comparison** ใน subprocess แยกอิสระ (รองรับ Cold/Warm Inductor cache, `--measure-pure-wall` *(เปิดเป็นค่าเริ่มต้น, ปิดด้วย `--no-pure-wall`)* + `nvidia-smi` timeline, `--diag-mode deep` *(เปิดเป็นค่าเริ่มต้น)*, `--metric-regex` ดึง `val_loss`/`Acc@1`/`tok/s` จาก stdout, `--parity-mode`, `--manifest`, `--timeout` ค่าเริ่มต้น `3600` วินาที / 1 ชั่วโมงต่อ run)
   - **Live Per-Profile Output & Incremental Save**: แสดงสรุปผลเทียบ Baseline ทันทีที่จบ **แต่ละ profile/arm** (ไม่ต้องรอจบทั้งหมด), บันทึก `summary.txt`/`summary.json`/`comparisons.csv` อัปเดตทีละขั้น (กดหยุดกลางทางก็ยังมีผลที่จบแล้วครบ), และเพิ่ม `--stream` หากต้องการดู stdout/stderr ของโปรเจกต์ที่กำลังรันแบบ real-time
   - เก็บผลลัพธ์และ log เชิงลึกทั้งหมดลงในโฟลเดอร์ **`benchmark_results/HHMMSS_YYYYMMDD-<name>/`** (ปรับได้ด้วย `--results-dir`) พร้อมเขียนรายงานสรุปลง [`report/`](../report/)

---

## 1.5 โครงสร้างโฟลเดอร์เก็บผลและ Deep Optimization Logs (`benchmark_results/`)

ทุกครั้งที่รัน `real_bench.py` ระบบจะสร้างโฟลเดอร์ผลลัพธ์แยกรายรอบอัตโนมัติ (หรือปิดด้วย `--no-save-bundle`) เพื่อให้ใช้วิเคราะห์คอขวดและปรับจูน `ptopt` ต่อได้ทันที:

```text
benchmark_results/HHMMSS_YYYYMMDD-<name>/
├── env_hardware.json          # Hardware fingerprint, GPU/SM/VRAM, PyTorch/Triton, target specs
├── summary.txt                # รายงานเต็ม (ตาราง 4-Arm, Phase Breakdown %, Dynamo Stats, Allocator, Top Ops, Module Shapes, Hints)
├── summary.json               # ข้อมูลดิบทั้งหมดแบบ machine-readable
├── comparisons.csv            # ตารางเปรียบเทียบสำหรับทำกราฟ / Excel
└── runs/
    └── <idx>_<target>__<arm>__<cache>/
        ├── run_meta.json          # คำสั่งเต็ม, cwd, env, exit code, wall/pure-wall, nvidia-smi peak & avg GPU util%
        ├── stdout_stderr.log      # stdout/stderr ฉบับเต็มของ repo
        ├── telemetry.json         # step_times_ms, train/eval แยกกัน, losses, memory_mb, fallback_events
        ├── resolved_config.json   # Schema v1 config และ rule trace ที่ ptopt resolve ได้จริง (เฉพาะ ptopt arms)
        ├── ptopt_events.json      # รายการ APPLIED / SKIPPED / FALLBACK ทั้งหมดแบบละเอียด
        ├── ptopt_fallback.txt     # สรุป pass audit แบบอ่านง่าย
        ├── debug_profile.json     # Phase breakdown (Data/Fwd/Loss/Bwd/Opt ms & %), Top-20 CUDA & CPU ops, CUDA allocator stats, Dynamo counters & compile_times, Module shape trace
        ├── diag_stderr.log        # TORCH_LOGS (graph_breaks, recompiles, perf_hints) + ptopt --verbose warnings
        ├── pure_wall.log          # log รอบ unhooked pure wall-clock (ไม่มี telemetry sync overhead)
        └── nvsmi_timeline.csv     # Time-series GPU memory, gpu_util_pct, mem_util_pct, power_w, temp_c ทุก 0.25s
```

> **อ่านอะไรเพื่อ Optimize `ptopt` ต่อได้ทันที?**
> - **Phase Breakdown (`debug_profile.json` / `summary.txt`)**: ดูว่าหลัง `ptopt` เร่ง `Forward+Backward` แล้วสัดส่วนไปกองที่ `Data/Other` (`cuda_prefetch` / workers), `Loss/Clip` (`Hybrid FusedLCE`), หรือ `Optim` (`optimizer.fusion`)
> - **Dynamo & `TORCH_LOGS` (`diag_stderr.log`)**: ดูจำนวน `graph_break_count`, `recompiles_count`, และ `inductor_cache_hit/miss` เพื่อแก้จุดที่ Dynamo แตกกราฟหรือ cache miss
> - **Top CUDA & CPU Ops**: จับ kernel หรือ CPU overhead ตัวที่กินเวลามากสุด (เช่น `aten::copy_`, `aten::_to_copy`, `cudaLaunchKernel`)
> - **Module & Dynamic Shape Trace**: เช็กว่าโมเดลมี dynamic input shapes กี่แบบ และ trigger compile gate ถูกหรือไม่
> - **`nvsmi_timeline.csv`**: ถ้า `gpu_util_pct` ต่ำทั้งที่ forward ไม่ช้า แสดงว่า launch-bound (ควรใช้ CUDA Graphs) หรือ CPU/DataLoader-bound

---

## 2. หลักการเปรียบเทียบ 4-Arm Matrix

ทุก GitHub Repo ควรวัดเทียบครบทั้ง 4 ขา (ตามความเหมาะสมของ repo):

| Arm | คำสั่งที่รัน | สิ่งที่พิสูจน์ |
| :--- | :--- | :--- |
| **Arm A: Naive Baseline** | `ptopt --telemetry-only -- <cmd default>` *(หรือรันเปล่าในโหมด `--measure-pure-wall`)* | ความเร็ว/VRAM เดิมของโค้ดผู้ใช้ที่ยังไม่จูน |
| **Arm B: `ptopt` Zero-Code** | `ptopt --profile <safe\|balanced\|aggressive> -- <cmd default>` | **Zero-Code Gain**: วัดแยก **Cold Cache** และ **Warm Cache** |
| **Arm C: Repo Native Opt** | `ptopt --telemetry-only -- <cmd เปิด flag opt ของ repo เอง>` | **Golden Reference**: เทียบ `ptopt` กับ flag ที่เจ้าของ repo จูนไว้เอง (เช่น `--compile=True`, `--amp`) |
| **Arm D: Conflict / Co-existence** | `ptopt --profile balanced -- <Arm C cmd>` | **Safety Check**: ถ้าโค้ดเดิมมี `autocast` / `torch.compile` อยู่แล้ว ครอบ `ptopt` ซ้ำ **ต้องไม่พัง ไม่ double-compile และไม่ช้าลง** |

> **ข้อสำคัญเรื่องความยุติธรรมในการวัดผล:**
> 1. **Pre-download ล่วงหน้า**: ดาวน์โหลด dataset, weights, และ tokenize ให้เสร็จก่อนรัน benchmark เสมอ เพื่อไม่ให้เวลาโหลดเน็ตมาปนใน End-to-End time
> 2. **เปิด `--measure-pure-wall` สำหรับตัวเลข End-to-End**: `TelemetryCollector` มี `cuda.synchronize()` ท้าย step เพื่อวัด step latency แม่นยำ แต่ `--measure-pure-wall` จะรันรอบเพียวๆ ที่ปิด telemetry hook เพื่อวัด **True Wall-Clock Speedup** และ peak `nvidia-smi` VRAM คู่กัน

---

## 3. เตรียม Environment (Google Colab / Local Linux GPU)

```bash
# 1) ติดตั้ง ptopt แบบไม่ทับ PyTorch/CUDA ของ Colab
cd /content/kernel-pytouch-optimize/kernel-pytorch-optimize
pip install --no-deps -e .

# 2) เช็กความพร้อมเครื่อง (GPU SM, VRAM, Triton, /dev/shm)
ptopt --doctor

# 3) สร้างโฟลเดอร์เก็บ external repos นอก repo นี้
mkdir -p /content/external_repos
export PTOPT_REPO=/content/kernel-pytouch-optimize/kernel-pytorch-optimize
```

---

## 4. ชุด GitHub Repos แนะนำสำหรับทดสอบ (แบ่ง 3 Tiers)

### Tier 1: Pure PyTorch Clean Repos (รันไว เหมาะทำ Regular Regression Check)

#### 4.1 `kuangliu/pytorch-cifar` — Vision CNN Training + Checkpoint `--resume`
ทดสอบ `DataLoader`, `SGD` fusion, `TF32`/`BF16`, `torch.compile`, small-input (`32x32`) `channels_last` gate, และการโหลด checkpoint ข้ามโหมด

```bash
cd /content/external_repos
git clone --depth 1 https://github.com/kuangliu/pytorch-cifar.git

# Pre-download CIFAR-10 ล่วงหน้า 1 รอบ (หรือรัน 1 epoch สั้นๆ)
cd /content/external_repos/pytorch-cifar
python3 -c "import torchvision; torchvision.datasets.CIFAR10(root='./data', train=True, download=True); torchvision.datasets.CIFAR10(root='./data', train=False, download=True)"
```

รัน Benchmark โดยจำกัดเหลือ **10 epochs** และล็อก seed ใน memory (เพราะ `main.py` ของ repo นี้ฮาร์ดโค้ด `start_epoch+200`, ไม่มี flag `--epochs`, และไม่ได้ fix random seed ทำให้ค่า `Acc:` ช่วง 10 epoch แรกแกว่ง ~70–80% ได้ระหว่างรอบ):
```bash
python3 $PTOPT_REPO/pipline-for-test/real_bench.py \
  --name pytorch-cifar-resnet18 \
  --cwd /content/external_repos/pytorch-cifar \
  --cmd "python3 -c \"import pathlib, random, re, torch; random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0); src = re.sub(r'start_epoch\s*\+\s*200', 'start_epoch+10', pathlib.Path('main.py').read_text()); exec(compile(src, 'main.py', 'exec'), {'__name__': '__main__', '__file__': 'main.py'})\"" \
  --profiles safe,balanced,aggressive \
  --cache-modes auto \
  --metric-regex "test_acc=Acc[:@\s]*([0-9.]+)%" \
  --parity-mode metric \
  --timeout 3600 \
  --measure-pure-wall
```
**ตรวจ Checkpoint Round-Trip เพิ่มเติม:**
1. รัน `ptopt --profile balanced -- python3 -c "import pathlib, re; src = re.sub(r'start_epoch\s*\+\s*200', 'start_epoch+10', pathlib.Path('main.py').read_text()); exec(compile(src, 'main.py', 'exec'), {'__name__': '__main__', '__file__': 'main.py'})"` (ให้เซฟ `./checkpoint/ckpt.pth`)
2. ปิด `ptopt` แล้วรัน Pure PyTorch: `python3 main.py --resume` -> ต้องโหลด `ckpt.pth` สำเร็จ 100% (ไม่มี `_orig_mod.` key mismatch)

---

#### 4.2 `karpathy/nanoGPT` — LLM Pretraining (ทดสอบกับ Vocab จำนวนมาก `vocab_size=50304`) + Autoregressive `sample.py` + Existing `autocast`/`compile`
ทดสอบ **Hybrid Fused Linear+CE** กับ **vocab จำนวนมาก (`vocab_size=50304` บน GPT-2 BPE)**, **Arm D Conflict** (nanoGPT มี `autocast` + `GradScaler` + `--compile` ในตัว), interleaved `val_loss` eval, `ckpt.pt` cross-load, และ dynamic KV/sequence length ใน `sample.py`

> **หมายเหตุจากผล L4 Benchmark (`192626_20261119`):**
> - เดิมใช้ `shakespeare_char` (`vocab_size=65`) ซึ่งเล็กกว่าเกณฑ์ Hybrid `FusedLinearCrossEntropy` (`>= 4096/8192`) และเปิด `bf16` + `FlashAttention` + `fused AdamW` อยู่แล้ว จึงเปลี่ยนมาใช้ชุด **`data/shakespeare/prepare.py` (GPT-2 BPE, `vocab_size=50304`)** ร่วมกับ `--dropout=0.0` และ `--max_iters=500` เพื่อวัดผล **Large-Vocab `FusedLCE` + `torch.compile`** ได้เต็มรูปแบบ

```bash
cd /content/external_repos
git clone --depth 1 https://github.com/karpathy/nanoGPT.git
cd /content/external_repos/nanoGPT
pip install -q tiktoken datasets transformers tqdm

# เตรียม dataset แบบ GPT-2 BPE (vocab_size=50257 -> pad เป็น 50304, ไม่นับเวลาช่วง prepare เข้า benchmark)
python3 data/shakespeare/prepare.py
```

**A) ทดสอบ Training ครบทั้ง 4 Arms กับ Vocab จำนวนมาก (`vocab_size=50304`, `12L/12H/768D`, Arm A=`--compile=False`, Arm B=`ptopt`, Arm C=`--compile=True`, Arm D=`ptopt` ซ้อน `--compile=True`):**
```bash
python3 $PTOPT_REPO/pipline-for-test/real_bench.py \
  --name nanogpt-gpt2-bpe \
  --cwd /content/external_repos/nanoGPT \
  --cmd "python3 train.py config/train_shakespeare_char.py --dataset=shakespeare --n_layer=12 --n_head=12 --n_embd=768 --batch_size=16 --block_size=256 --dropout=0.0 --compile=False --max_iters=500 --eval_interval=250 --eval_iters=10 --always_save_checkpoint=True" \
  --native-cmd "python3 train.py config/train_shakespeare_char.py --dataset=shakespeare --n_layer=12 --n_head=12 --n_embd=768 --batch_size=16 --block_size=256 --dropout=0.0 --compile=True --max_iters=500 --eval_interval=250 --eval_iters=10 --always_save_checkpoint=True" \
  --test-conflict \
  --profiles safe,balanced,aggressive \
  --cache-modes auto \
  --metric-regex "val_loss=val loss ([0-9.]+)" \
  --parity-mode auto \
  --measure-pure-wall
```

**B) ทดสอบ Autoregressive Inference (`sample.py`) และ Checkpoint Cross-Load (รันต่อจาก A ที่เซฟ `out-shakespeare-char/ckpt.pt` ไว้):**
```bash
python3 $PTOPT_REPO/pipline-for-test/real_bench.py \
  --name nanogpt-sample \
  --cwd /content/external_repos/nanoGPT \
  --cmd "python3 sample.py --out_dir=out-shakespeare-char --num_samples=2 --max_new_tokens=200 --seed=1337" \
  --profiles safe,balanced,aggressive \
  --cache-modes auto \
  --parity-mode exit_only \
  --measure-pure-wall
```

---

#### 4.3 `pytorch/examples` (`imagenet/main.py`) — Standard Vision 224x224
ทดสอบ `ResNet50` ที่ขนาดภาพจริง `224x224` (`channels_last` 4D, `TF32`, `BF16`, `compile`, `DataLoader` + `cuda_prefetch`) โดยใช้ `--dummy` พร้อม cap ขนาด `FakeData` ใน memory (เพราะ `--dummy` ปกติของ `imagenet/main.py` สร้าง `FakeData(1281167, ...)` + `FakeData(50000, ...)` = 20,800 steps/epoch บน CPU PIL ซึ่งจะ timeout บน Colab จึงจำกัดเหลือ 6,400 train / 1,280 val samples หรือ ~100+20 batches):

```bash
cd /content/external_repos
git clone --depth 1 https://github.com/pytorch/examples.git

python3 $PTOPT_REPO/pipline-for-test/real_bench.py \
  --name pytorch-examples-resnet50 \
  --cwd /content/external_repos/examples/imagenet \
  --cmd "python3 -c \"import sys, runpy, torchvision.datasets as d; _F=d.FakeData; d.FakeData=lambda sz,*a,**kw: _F(6400 if sz>100000 else (1280 if sz>10000 else sz),*a,**kw); sys.argv=['main.py','-a','resnet50','--dummy','--epochs','1','--batch-size','64','--workers','2']; runpy.run_path('main.py', run_name='__main__')\"" \
  --profiles safe,balanced,aggressive \
  --ptopt-arg="--override dataloader.cuda_prefetch=true" \
  --cache-modes auto \
  --parity-mode auto \
  --timeout 3600 \
  --measure-pure-wall
```

---

### Tier 2: Modern Architectures (จุดโชว์ของ Triton Kernels, Hybrid LCE, Auto Act-Ckpt & Static FP8)

#### 4.4 `huggingface/pytorch-image-models` (`timm`) — ViT & ConvNeXt
ทดสอบว่าเทียบกับ `timm` default (Arm A) แล้ว `ptopt` (Arm B) เร่งขึ้นกี่เท่า และเทียบชั้นกับการเปิด `--amp --channels-last --torchcompile inductor` ของ `timm` เอง (Arm C) ได้แค่ไหน

```bash
cd /content/external_repos
git clone --depth 1 https://github.com/huggingface/pytorch-image-models.git
pip install -q timm

python3 $PTOPT_REPO/pipline-for-test/real_bench.py \
  --name timm-vit-base \
  --cwd /content/external_repos/pytorch-image-models \
  --cmd "python3 train.py --dataset fake --num-classes 100 --model vit_base_patch16_224 --epochs 1 --batch-size 64 --workers 2" \
  --native-cmd "python3 train.py --dataset fake --num-classes 100 --model vit_base_patch16_224 --epochs 1 --batch-size 64 --workers 2 --amp --channels-last --torchcompile inductor" \
  --test-conflict \
  --profiles safe,balanced,aggressive \
  --metric-regex "loss=Loss:\s*([0-9.]+)" \
  --timeout 3600 \
  --measure-pure-wall
```
> **หมายเหตุระยะเวลารัน (`timm`)**: คำสั่งนี้มี 8 arms (`--test-conflict` + 3 profiles) × 3 passes และบางเวอร์ชันของ `timm` อาจต้องใช้ `--dataset torch/cifar100 --dataset-download --img-size 224` แทน `--dataset fake` หากต้องการรันไวขึ้นบน Colab ให้ลดเหลือ `--profiles safe,balanced` ก่อน


---

#### 4.5 `huggingface/trl` หรือ `Lightning-AI/litgpt` — Modern LLM (`RMSNorm + RoPE + SwiGLU + Vocab 151k`)
เหมาะทดสอบ **Triton Kernels**, **Hybrid `FusedLinearCrossEntropy`**, และ **Auto Activation Checkpointing (OOM Rescue)** บน `Qwen/Qwen2-0.5B` (pre-download model & dataset ก่อนเริ่มรัน):

```bash
cd /content/external_repos
git clone --depth 1 https://github.com/huggingface/trl.git
pip install -q trl peft accelerate datasets transformers

# Pre-download model & dataset
python3 -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen2-0.5B'); AutoModelForCausalLM.from_pretrained('Qwen/Qwen2-0.5B')"
python3 -c "from datasets import load_dataset; load_dataset('trl-lib/Capybara', split='train[:200]')"
```

รันเทียบ Arm A vs `ptopt` (`balanced` ที่มี Hybrid LCE + Auto Act-Ckpt > 82% VRAM):
```bash
python3 $PTOPT_REPO/pipline-for-test/real_bench.py \
  --name trl-sft-qwen2-0.5b \
  --cwd /content/external_repos/trl \
  --cmd "python3 trl/scripts/sft.py --model_name_or_path Qwen/Qwen2-0.5B --dataset_name trl-lib/Capybara --max_steps 30 --per_device_train_batch_size 4 --logging_steps 5 --save_strategy no --output_dir /tmp/qwen_sft_out" \
  --profiles safe,balanced \
  --env HF_HUB_OFFLINE=1 \
  --env HF_DATASETS_OFFLINE=1 \
  --metric-regex "train_loss='loss':\s*([0-9.]+)" \
  --parity-mode auto \
  --measure-pure-wall
```
> **OOM Rescue Stress Test**: ลองเพิ่ม `--per_device_train_batch_size` จน Arm A ชน `CUDA out of memory` แล้วทดสอบ `ptopt --profile balanced --fallback-report` เพื่อยืนยันว่า Auto Activation Checkpointing / Step-0 OOM rescue + Hybrid LCE ช่วยให้รันจบได้โดยไม่แก้โค้ด!

---

#### 4.6 `facebookresearch/DiT` (หรือ `huggingface/diffusers`) — Static Diffusion Inference & Static FP8
เหมาะทดสอบ **Static-shape Iterative Denoising** ด้วย `Inference CUDA Graphs` + `PTOPT_INFERENCE_FP8=1` + `PTOPT_FP8_STATIC_SCALE=1` (บน L4 `sm_89` / H100 `sm_90`):

```bash
cd /content/external_repos
git clone --depth 1 https://github.com/facebookresearch/DiT.git
pip install -q diffusers timm

# Pre-download DiT-XL/2 + VAE weights ก่อนจับเวลา
python3 -c "import sys; sys.path.insert(0, '/content/external_repos/DiT'); from download import find_model; find_model('DiT-XL-2-256x256.pt')"

# เทียบ Baseline vs balanced vs Static FP8 (L4 sm_89+)
python3 $PTOPT_REPO/pipline-for-test/real_bench.py \
  --name dit-xl2-fp8-static \
  --cwd /content/external_repos/DiT \
  --cmd "python3 sample.py --image-size 256 --num-sampling-steps 25 --seed 42" \
  --profiles balanced,aggressive \
  --env PTOPT_INFERENCE_FP8=1 \
  --env PTOPT_FP8_STATIC_SCALE=1 \
  --parity-mode exit_only \
  --measure-pure-wall
```

---

### Tier 3: Official Suite & Framework Stress Tests

#### 4.7 `pytorch/benchmark` (TorchBench) — เปรียบเทียบมาตรฐาน `eager` vs `ptopt` vs `inductor`
รันเฉพาะ subset ที่เหมาะกับ GPU 16–24GB (`resnet50`, `convnext_base`, `timm_vision_transformer`, `BERT_pytorch`, `hf_GPT2`, `dcgan`, `yolov3`):

```bash
cd /content/external_repos
git clone --depth 1 https://github.com/pytorch/benchmark.git
cd benchmark
python3 install.py resnet50 BERT_pytorch hf_GPT2 timm_vision_transformer

# ตัวอย่างเทียบ BERT_pytorch training: Eager (Arm A) vs ptopt (Arm B) vs Inductor (Arm C)
python3 $PTOPT_REPO/pipline-for-test/real_bench.py \
  --name torchbench-bert-train \
  --cwd /content/external_repos/benchmark \
  --cmd "python3 run.py BERT_pytorch -d cuda -t train -m eager" \
  --native-cmd "python3 run.py BERT_pytorch -d cuda -t train -m inductor" \
  --test-conflict \
  --profiles safe,balanced,aggressive \
  --parity-mode exit_only \
  --measure-pure-wall
```

#### 4.8 `huggingface/transformers` (`run_clm.py`) — `Trainer` + `Accelerate` Compatibility
ทดสอบว่าเมื่อโค้ดอยู่ใต้ HuggingFace `Trainer` เต็มรูปแบบ `ptopt` ไม่ทำให้ `Trainer` พัง, ไม่ double-AMP, และรายงาน pass audit ออกมาสะอาดผ่าน `--fallback-report --save-report`

---

## 5. รันหลาย Repo ต่อเนื่องด้วย `--manifest` (เก็บไฟล์ JSON ไว้บนเครื่องเอง)

สามารถสร้างไฟล์ `/content/my_real_manifest.json` (อยู่นอก git repo ของ `ptopt`) แล้วสั่งรันทีเดียวเพื่อสรุปรวมลงรายงานเดียว:

```json
{
  "targets": [
    {
      "name": "pytorch-cifar",
      "cwd": "/content/external_repos/pytorch-cifar",
      "cmd": "python3 -c \"import pathlib, re; src = re.sub(r'start_epoch\\s*\\+\\s*200', 'start_epoch+10', pathlib.Path('main.py').read_text()); exec(compile(src, 'main.py', 'exec'), {'__name__': '__main__', '__file__': 'main.py'})\"",
      "profiles": ["safe", "balanced"],
      "cache_modes": ["auto"],
      "metric_regex": { "acc": "Acc[:@\\s]*([0-9.]+)%" },
      "parity_mode": "metric",
      "timeout_sec": 3600,
      "measure_pure_wall": true
    },
    {
      "name": "nanogpt-shakespeare",
      "cwd": "/content/external_repos/nanoGPT",
      "cmd": "python3 train.py config/train_shakespeare_char.py --compile=False --max_iters=150 --eval_interval=50 --eval_iters=10",
      "native_cmd": "python3 train.py config/train_shakespeare_char.py --compile=True --max_iters=150 --eval_interval=50 --eval_iters=10",
      "test_conflict": true,
      "profiles": ["safe", "balanced", "aggressive"],
      "cache_modes": ["auto"],
      "metric_regex": { "val_loss": "val loss ([0-9.]+)" },
      "measure_pure_wall": true
    }
  ]
}
```

คำสั่งรันและ export ผล:
```bash
python3 $PTOPT_REPO/pipline-for-test/real_bench.py \
  --manifest /content/my_real_manifest.json \
  --report-name real-suite-l4 \
  --json-out $PTOPT_REPO/report/real-suite-l4.json \
  --csv-out $PTOPT_REPO/report/real-suite-l4.csv
```

---

## 6. Checklist ตรวจผลหลังรันแต่ละ Repo

- [ ] **Zero Code Diff**: `git status` ใน repo เป้าหมายต้อง clean (`nothing to commit`, ไม่มีการแก้โค้ด)
- [ ] **Pass Audit Clean**: เปิดดู section `[PASS AUDIT & DIAGNOSTICS]` ในรายงาน (หรือรัน `ptopt --fallback-report --save-report report/ -- ...`) ว่าไม่มี `FALLBACK (runtime error)` ที่ไม่คาดคิด
- [ ] **Pure Wall vs Step Speedup**: เช็กทั้ง `Step Spd` (steady-state) และ `Pure Wall` (รวม compile warmup หรือ cold E2E)
- [ ] **VRAM & OOM Margin**: เทียบทั้ง `VRAM Alloc/Res` และ `NVSMI MB` ในกรณี `DataLoader` / `Hybrid LCE` / `Auto Act-Ckpt`
- [ ] **Quality Parity**: ตรวจ `val_loss` / `Acc@1` / คุณภาพ sample (โดยเฉพาะเมื่อเปิด `FP8`) ว่าไม่ diverge
- [ ] **Checkpoint Cross-Load**: นำไฟล์ `.pt` / `.pth` ที่เทรนด้วย `ptopt` ไปโหลดใน Pure PyTorch (ไม่ครอบ `ptopt`) แล้วยืนยันว่า `load_state_dict` ผ่านโดยไม่มีคีย์ `_orig_mod.` หรือปัญหา non-contiguous `channels_last`
