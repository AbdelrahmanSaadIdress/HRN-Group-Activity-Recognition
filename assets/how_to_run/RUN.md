## How to Run

### 1. Install dependencies
```bash
pip install wandb
wandb login
```

---

### 2. Stage 1 — Train
```bash
python main.py \
  --mode train \
  --model B1_NoRelations \
  --stage 1 \
  --config B1_NoRelations_stage1.yaml
```

**Resume if interrupted:**
```bash
python main.py \
  --mode train \
  --model B1_NoRelations \
  --stage 1 \
  --config B1_NoRelations_stage1.yaml \
    
```
> Change `preload` in the YAML to `cont` and omit `--checkpoint` if you just want it to auto-pick the latest epoch checkpoint.

---

### 3. Stage 1 — Test
```bash
python main.py \
  --mode test \
  --model B1_NoRelations \
  --stage 1 \
  --config B1_NoRelations_stage1.yaml \
  --checkpoint Outputs/checkpoints/stage1/best.pt
```

---

### 4. Stage 2 — Train
```bash
python main.py \
  --mode train \
  --model B1_NoRelations \
  --stage 2 \
  --config B1_NoRelations_stage2.yaml \
  --stage1_checkpoint Outputs/checkpoints/stage1/best.pt
```

**Resume if interrupted:**
```bash
python main.py \
  --mode train \
  --model B1_NoRelations \
  --stage 2 \
  --config B1_NoRelations_stage2.yaml \
  --stage1_checkpoint Outputs/checkpoints/stage1/best.pt \
  --checkpoint Outputs/checkpoints/stage2/epoch_005.pt
```

---

### 5. Stage 2 — Test
```bash
python main.py \
  --mode test \
  --model B1_NoRelations \
  --stage 2 \
  --config B1_NoRelations_stage2.yaml \
  --checkpoint Outputs/checkpoints/stage2/best.pt
```

---