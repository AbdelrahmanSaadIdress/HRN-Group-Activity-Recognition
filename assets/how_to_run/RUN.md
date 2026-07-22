## How to Run

The project supports training and testing multiple models across two stages:

* **Stage 1:** Individual/person-level activity recognition.
* **Stage 2:** Group activity recognition using the Stage 1 model.

> **Important:** Stage 1 is trained first and then **frozen** during Stage 2 training. Therefore, the main focus of the experiments and model development is on **Stage 2**, where the group activity recognition component is trained using the fixed Stage 1 feature extractor.

Replace the placeholders in the commands with the model and configuration you want to use.

---

### 1. Install Dependencies

```bash
pip install wandb
wandb login
```

---

## 2. Stage 1 — Train

Stage 1 is trained to recognize individual/person-level activities.

```bash
python main.py \
  --mode train \
  --model <MODEL_NAME> \
  --stage 1 \
  --config <STAGE1_CONFIG_FILE>
```

After Stage 1 training is complete, the best checkpoint is used as the feature extractor for Stage 2.

---

## 3. Stage 1 — Test

To evaluate the trained Stage 1 model:

```bash
python main.py \
  --mode test \
  --model <MODEL_NAME> \
  --stage 1 \
  --config <STAGE1_CONFIG_FILE> \
  --checkpoint <STAGE1_CHECKPOINT_PATH>
```

---

## 4. Stage 2 — Train

Stage 2 requires a trained Stage 1 checkpoint.

During Stage 2 training:

* The Stage 1 model is loaded from the provided checkpoint.
* The Stage 1 model is **frozen**.
* Its parameters are not updated during backpropagation.
* Training focuses primarily on the Stage 2 group activity recognition architecture.

```bash
python main.py \
  --mode train \
  --model <MODEL_NAME> \
  --stage 2 \
  --config <STAGE2_CONFIG_FILE> \
  --stage1_checkpoint <STAGE1_CHECKPOINT_PATH>
```

### Example

```bash
python main.py \
  --mode train \
  --model B1_NoRelations \
  --stage 2 \
  --config B1_NoRelations_stage2.yaml \
  --stage1_checkpoint Outputs/checkpoints/stage1/best.pt
```

### Resume Stage 2 Training

```bash
python main.py \
  --mode train \
  --model <MODEL_NAME> \
  --stage 2 \
  --config <STAGE2_CONFIG_FILE> \
  --stage1_checkpoint <STAGE1_CHECKPOINT_PATH> \
  --checkpoint <STAGE2_CHECKPOINT_PATH>
```

---

## 5. Stage 2 — Test

To evaluate the trained Stage 2 model:

```bash
python main.py \
  --mode test \
  --model <MODEL_NAME> \
  --stage 2 \
  --config <STAGE2_CONFIG_FILE> \
  --checkpoint <STAGE2_CHECKPOINT_PATH>
```

---

## Command Parameters

| Parameter             | Description                                                |
| --------------------- | ---------------------------------------------------------- |
| `--mode`              | Operation mode: `train` or `test`                          |
| `--model`             | Name of the model to train or evaluate                     |
| `--stage`             | Pipeline stage: `1` or `2`                                 |
| `--config`            | Path to the YAML configuration file                        |
| `--checkpoint`        | Path to a checkpoint for testing or resuming training      |
| `--stage1_checkpoint` | Path to the trained Stage 1 checkpoint required by Stage 2 |

---

## General Workflow

```text
1. Train Stage 1
      ↓
2. Save the best Stage 1 checkpoint
      ↓
3. Freeze Stage 1
      ↓
4. Train Stage 2 using the frozen Stage 1 model
      ↓
5. Test Stage 2
```

> **In practice, most model development and experimentation focuses on Stage 2.** Stage 1 serves primarily as a pretrained and frozen feature extractor, while the different Stage 2 architectures are trained and evaluated for group activity recognition.
