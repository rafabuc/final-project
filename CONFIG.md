# Hydra Configuration Guide - Multimodal Fake News Detection

This document describes all configuration variables used in the Multimodal Fake News Detection system, with **special emphasis on model loading, versioning, and transfer learning parameters**.

## Configuration Structure

```
configs/
├── config.yaml                # Main training configuration
├── config_inference.yaml      # Main inference configuration
├── data/
│   └── fakeddit.yaml         # Dataset configuration
├── model/
│   └── multimodal.yaml       # Model architecture configuration
├── training/
│   └── default.yaml          # Training hyperparameters
└── inference/
    └── default.yaml          # Inference settings
```

---

## Critical Training Parameters Overview

### Model Loading & Versioning Workflow

```
┌─────────────────────────────────────────────────────────────┐
│  1. load_model: true                                         │
│     ├─> best_model_name: "best_model_1.6.pth" (SOURCE)     │
│     └─> Loads pre-trained weights from checkpoint           │
│                                                              │
│  2. version: "1.8"                                          │
│     └─> Saves new model as "best_model_1.8.pth" (TARGET)   │
│                                                              │
│  3. freeze_vision: false  &  freeze_text: false             │
│     └─> Fine-tunes ALL layers (slow but better accuracy)   │
│                                                              │
│  4. learning_rate: 0.00002                                  │
│     └─> Small LR for fine-tuning without catastrophic      │
│         forgetting of pre-trained knowledge                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔥 Key Training Variables (Most Important)

### 1. Model Loading Configuration (`model/multimodal.yaml`)

#### **`load_model`** ⚠️ CRITICAL
- **Type:** Boolean
- **Default:** `true`
- **Location:** `configs/model/multimodal.yaml:11`
- **Code Reference:** `train.py:105-130`

**Purpose:** Determines whether to start training from scratch or continue from a previously trained checkpoint.

**Impact on Training:**
- `true`: Loads pre-trained weights → **Faster convergence**, leverages previous learning
- `false`: Random initialization → **Slower training**, starts from zero

**Usage in code:**
```python
# train.py:105-130
load_model = cfg.model.get('load_model', False)
if load_model:
    checkpoint_path = os.path.join(
        cfg.training.checkpoint_dir,
        best_model_name
    )
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
```

---

#### **`best_model_name`** ⚠️ CRITICAL
- **Type:** String
- **Default:** `"best_model_1.6.pth"`
- **Location:** `configs/model/multimodal.yaml:12`
- **Code Reference:** `train.py:109`

**Purpose:** Specifies which checkpoint file to load as the starting point.

**Impact on Training:**
- Defines the **source model** for transfer learning
- Must exist in `checkpoint_dir` or training will fail with error

**Common Values:**
- `best_model_1.6.pth` - Load from version 1.6
- `best_model_1.7.pth` - Load from version 1.7
- `best_model_previous.pth` - Generic previous version

**Error Handling:**
```python
# train.py:127-130
if not os.path.exists(checkpoint_path):
    logger.warning("¡Not found checkpoint!")
    raise ValueError(f"Not found checkpoint {checkpoint_path}")
```

---

#### **`version`** ⚠️ CRITICAL
- **Type:** String/Float
- **Default:** `1.8`
- **Location:** `configs/training/default.yaml:5`
- **Code Reference:** `trainer.py:347, 445, 496`

**Purpose:** Identifies the **new model version** being trained. Used for:
1. Checkpoint naming: `best_model_{version}.pth`
2. MLflow run tracking
3. Artifact naming (plots, confusion matrices)

**Impact on Training:**
- Prevents overwriting previous checkpoints
- Enables version tracking and comparison
- Used in MLflow for experiment organization

**Checkpoint Saving:**
```python
# trainer.py:345-348
checkpoint_path = os.path.join(
    self.config.get('checkpoint_dir', 'checkpoints'),
    f'best_model_{self.config.get("version", "1.0")}.pth'
)
```

**Example Evolution:**
```
v1.6 (dropout=0.5)
  ↓ (load_model=true, best_model_name="best_model_1.6.pth")
v1.7 (dropout=0.3, new transforms)
  ↓ (load_model=true, best_model_name="best_model_1.7.pth")
v1.8 (dropout=0.5, alternative transforms)
```

---

### 2. Transfer Learning Controls (`model/multimodal.yaml`)

#### **`freeze_vision`** 🔥 HIGH IMPACT
- **Type:** Boolean
- **Default:** `false`
- **Location:** `configs/model/multimodal.yaml:15`
- **Code Reference:** `train.py:100`

**Purpose:** Controls whether EfficientNet-B0 backbone parameters are trainable.

**Impact on Training:**

| Value | Effect | Speed | Accuracy | Use Case |
|-------|--------|-------|----------|----------|
| `true` | **Frozen** - No gradient updates to vision backbone | ⚡ Fast | 🔵 Lower | Small dataset, quick experiments |
| `false` | **Fine-tuning** - Updates all vision layers | 🐌 Slow | 🟢 Higher | Large dataset, final training |

**Memory & Compute:**
- `freeze_vision=true`: ~40% less GPU memory, ~2x faster training
- `freeze_vision=false`: Full backprop through EfficientNet (1280 features)

**Gradient Flow:**
```
freeze_vision=true:  Input → [EfficientNet FROZEN] → Embedding → Fusion ✓
freeze_vision=false: Input → [EfficientNet ✓✓✓✓✓] → Embedding → Fusion ✓
```

---

#### **`freeze_text`** 🔥 HIGH IMPACT
- **Type:** Boolean
- **Default:** `false`
- **Location:** `configs/model/multimodal.yaml:16`
- **Code Reference:** `train.py:101`

**Purpose:** Controls whether DistilBERT backbone parameters are trainable.

**Impact on Training:**

| Value | Effect | Speed | Accuracy | Use Case |
|-------|--------|-------|----------|----------|
| `true` | **Frozen** - No gradient updates to text backbone | ⚡ Fast | 🔵 Lower | Generic text, limited compute |
| `false` | **Fine-tuning** - Updates all text layers | 🐌 Slow | 🟢 Higher | Domain-specific language |

**Memory & Compute:**
- `freeze_text=true`: ~50% less GPU memory, ~2.5x faster training
- `freeze_text=false`: Full backprop through DistilBERT (66M parameters)

**Common Configurations:**

| Scenario | freeze_vision | freeze_text | Training Time | Accuracy |
|----------|---------------|-------------|---------------|----------|
| **Quick Experiment** | `true` | `true` | 1x (baseline) | ⭐⭐ |
| **Vision Focus** | `false` | `true` | 2x | ⭐⭐⭐ |
| **Text Focus** | `true` | `false` | 2.5x | ⭐⭐⭐ |
| **Full Fine-tuning** | `false` | `false` | 4x | ⭐⭐⭐⭐ |

---

### 3. Learning Rate Configuration (`training/default.yaml`)

#### **`learning_rate`** 🔥 HIGH IMPACT
- **Type:** Float
- **Default:** `0.00002` (2e-5)
- **Location:** `configs/training/default.yaml:16`
- **Code Reference:** `train.py:138-139, trainer.py:154`

**Purpose:** Controls the step size for gradient descent updates.

**Impact on Training:**

| Value | Effect | Risk | Use Case |
|-------|--------|------|----------|
| **High** (>1e-4) | Fast convergence | Catastrophic forgetting, unstable | Training from scratch |
| **Medium** (1e-4 to 5e-5) | Balanced | Moderate | Light fine-tuning |
| **Low** (<2e-5) | Slow, stable | Underfitting if too low | **Transfer learning** ✓ |

**Why 2e-5 for Fine-tuning?**
- Pre-trained models already have good weights
- Small updates preserve learned features
- Prevents destroying pre-trained knowledge
- Standard practice for BERT-family models

**Learning Rate Schedule:**
```python
# train.py:152-159 - OneCycleLR
scheduler = OneCycleLR(
    optimizer,
    max_lr=cfg.training.learning_rate,  # Peak LR
    epochs=cfg.training.num_epochs,
    steps_per_epoch=len(train_loader),
    pct_start=0.1,                       # Warmup 10%
    anneal_strategy='cos'                # Cosine decay
)
```

**Typical Evolution:**
```
Epoch 1-2: 0 → 2e-5  (warmup)
Epoch 3:    2e-5      (peak)
Epoch 4-10: 2e-5 → 0  (cosine annealing)
```

---

#### **`use_scheduler`**
- **Type:** Boolean
- **Default:** `true`
- **Location:** `configs/training/default.yaml:23`
- **Code Reference:** `train.py:151-159, trainer.py:150-153`

**Purpose:** Enable/disable learning rate scheduling.

**Impact on Training:**
- `true`: Dynamic LR (warmup + cosine decay) → Better convergence
- `false`: Fixed LR → Simpler, may plateau early

**Scheduler Update Strategy:**
```python
# trainer.py:150-153 - Updated PER BATCH
if isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
    self.scheduler.step()  # Called after each batch
```

---

#### **`min_lr`**
- **Type:** Float
- **Default:** `0.000001` (1e-6)
- **Location:** `configs/training/default.yaml:24`

**Purpose:** Minimum learning rate floor for cosine annealing.

**Impact:** Prevents LR from going to zero, maintains minimal updates.

---

### 4. Regularization Parameters

#### **`dropout_rate`** 🔥 HIGH IMPACT
- **Type:** Float
- **Default:** `0.3`
- **Location:** `configs/model/multimodal.yaml:8`
- **Code Reference:** `train.py:98`

**Purpose:** Dropout probability in fusion layers for regularization.

**Impact on Training:**

| Value | Effect | Generalization | Training Accuracy |
|-------|--------|----------------|-------------------|
| **0.1-0.2** | Low dropout | Overfitting risk | High |
| **0.3-0.4** | **Balanced** ✓ | Good | Moderate |
| **0.5-0.6** | High dropout | Better generalization | Lower |

**Version History:**
- v1.6: `dropout_rate=0.5` (aggressive regularization)
- v1.7: `dropout_rate=0.3` (reduced for better train accuracy)
- v1.8: `dropout_rate=0.5` (reverted to prevent overfitting)

**Where Applied:**
```python
# In MultimodalNet fusion layers
nn.Dropout(dropout_rate)  # Applied between dense layers
```

---

#### **`weight_decay`**
- **Type:** Float
- **Default:** `0.01`
- **Location:** `configs/training/default.yaml:17`
- **Code Reference:** `train.py:139`

**Purpose:** L2 regularization penalty on weights (prevents large weights).

**Impact:** Complements dropout, standard value for AdamW optimizer.

---

### 5. Optimizer Configuration

#### **`optimizer`**
- **Type:** String
- **Default:** `"adamw"`
- **Location:** `configs/training/default.yaml:20`
- **Code Reference:** `train.py:135-146`

**Options:**
- `"adam"`: Classic Adam optimizer
- `"adamw"`: **Adam with decoupled weight decay** (recommended)

**Why AdamW for Fine-tuning?**
- Better generalization than Adam
- Decoupled weight decay (more effective regularization)
- Standard for transformer models

---

## Complete Configuration Files

### 1. Main Configuration (`config.yaml`)

```yaml
defaults:
  - model: multimodal      # → configs/model/multimodal.yaml
  - data: fakeddit        # → configs/data/fakeddit.yaml
  - training: default     # → configs/training/default.yaml
  - _self_

mlflow:
  tracking_uri: ./mlruns                          # MLflow storage location
  experiment_name: multimodal_fake_news_detection # Experiment identifier

hydra:
  run:
    dir: outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}  # Output directory per run
  sweep:
    dir: multirun/${now:%Y-%m-%d}/${now:%H-%M-%S} # Hyperparameter sweep directory
```

---

### 2. Data Configuration (`data/fakeddit.yaml`)

#### Dataset Paths
| Variable | Default | Description |
|----------|---------|-------------|
| `csv_file` | `/` | Path to main CSV (update per user) |
| `root_dir` | `../../data/` | Root data directory |
| `images_dir_train` | `../../data/images/train/` | Training images directory |
| `images_dir_val` | `../../data/images/val/` | Validation images directory |

**Code Usage:** `train.py:48-72`

#### Data Loading
| Variable | Default | Description | Code Reference |
|----------|---------|-------------|----------------|
| `n_rows` | `1000` | Number of CSV rows (-1=all) | `train.py:57, 71` |
| `text_column` | `text_content` | CSV column for text | `train.py:54` |
| `image_column` | `image_path` | CSV column for image paths | `train.py:55` |
| `label_column` | `label` | CSV column for labels (0/1) | `train.py:56` |
| `train_split` | `0.8` | Train/val split ratio | Not used (separate CSVs) |

#### Preprocessing
| Variable | Default | Description | Impact |
|----------|---------|-------------|--------|
| `image_size` | `224` | Image dimensions (224×224) | Must match EfficientNet input |
| `max_length` | `128` | Max token sequence length | Longer = more context, slower |

**Code Usage:** `train.py:51, 53, 65`

#### Data Augmentation (Training Only)
| Variable | Default | Description | Applied When |
|----------|---------|-------------|--------------|
| `augmentation.horizontal_flip` | `true` | Random horizontal flip | `mode='train'` only |
| `augmentation.rotation_degrees` | `15` | Random rotation ±15° | `mode='train'` only |
| `augmentation.color_jitter.brightness` | `0.2` | Brightness variation | `mode='train'` only |
| `augmentation.color_jitter.contrast` | `0.2` | Contrast variation | `mode='train'` only |
| `augmentation.color_jitter.saturation` | `0.2` | Saturation variation | `mode='train'` only |

**Code Usage:** `train.py:53` via `get_image_transforms(mode='train')`

**Validation:** Uses `mode='val'` → No augmentation, only resize + normalize

---

### 3. Model Configuration (`model/multimodal.yaml`)

#### Architecture Dimensions
| Variable | Default | Description | Impact |
|----------|---------|-------------|--------|
| `embedding_dim` | `512` | Common embedding space size | Higher = more capacity, slower |
| `fusion_hidden_dim` | `256` | Fusion layer hidden size | Hidden layer between embeddings and output |

**Code Usage:** `train.py:96-97`

#### Model Checkpointing (see detailed section above)
| Variable | Default | Description |
|----------|---------|-------------|
| `load_model` | `true` | Load checkpoint before training |
| `best_model_name` | `best_model_1.6.pth` | Source checkpoint filename |

#### Transfer Learning (see detailed section above)
| Variable | Default | Description |
|----------|---------|-------------|
| `freeze_vision` | `false` | Freeze EfficientNet backbone |
| `freeze_text` | `false` | Freeze DistilBERT backbone |

#### Backbone Configuration
| Variable | Default | Description |
|----------|---------|-------------|
| `vision.pretrained` | `true` | Use ImageNet pre-trained EfficientNet |
| `vision.backbone` | `efficientnet_b0` | Vision model architecture |
| `text.pretrained_model` | `distilbert-base-uncased` | HuggingFace model ID |
| `text.hidden_size` | `768` | DistilBERT output dimension |

---

### 4. Training Configuration (`training/default.yaml`)

#### Experiment Tracking
| Variable | Default | Description | Code Reference |
|----------|---------|-------------|----------------|
| `experiment_name` | `multimodal_fake_news_detection` | MLflow experiment | `trainer.py:244` |
| `version` | `1.8` | Model version (for checkpoint naming) | `trainer.py:347` |
| `tracking_uri` | `sqlite:///mlflow.db` | MLflow backend URI | `trainer.py:75` |

#### Training Hyperparameters (see detailed sections above)
| Variable | Default | Description |
|----------|---------|-------------|
| `num_epochs` | `5` | Training epochs |
| `batch_size` | `32` | Samples per batch |
| `learning_rate` | `0.00002` | Initial LR (2e-5) |
| `weight_decay` | `0.01` | L2 regularization |
| `optimizer` | `adamw` | Optimizer type |
| `use_scheduler` | `true` | Enable LR scheduling |
| `min_lr` | `0.000001` | Minimum LR (1e-6) |

#### Data Loading
| Variable | Default | Description | Code Reference |
|----------|---------|-------------|----------------|
| `num_workers` | `1` | DataLoader workers | `train.py:81, 89` |

#### Checkpointing
| Variable | Default | Description | Code Reference |
|----------|---------|-------------|----------------|
| `checkpoint_dir` | `./checkpoints` | Checkpoint save directory | `trainer.py:345` |
| `save_frequency` | `5` | Save every N epochs | Not implemented |

#### Reproducibility
| Variable | Default | Description | Code Reference |
|----------|---------|-------------|----------------|
| `seed` | `42` | Random seed | `train.py:34-36` |

#### Early Stopping (Not Implemented)
| Variable | Default | Description |
|----------|---------|-------------|
| `early_stopping.enabled` | `true` | Enable early stopping |
| `early_stopping.patience` | `5` | Epochs without improvement |
| `early_stopping.min_delta` | `0.001` | Minimum improvement threshold |

#### Performance (Not Implemented)
| Variable | Default | Description |
|----------|---------|-------------|
| `mixed_precision` | `true` | FP16 training (not active) |

---

### 5. Inference Configuration (`inference/default.yaml`)

#### Inference Mode
| Variable | Default | Description |
|----------|---------|-------------|
| `mode` | `batch` | `single` or `batch` inference |

#### Model Checkpoint
| Variable | Default | Description |
|----------|---------|-------------|
| `checkpoint.path` | `./src/training/checkpoints/best_model_1.7.pth` | Trained model path |
| `checkpoint.device` | `null` | Device (`cuda`/`cpu`/`null` auto) |

#### Single Prediction
| Variable | Default | Description |
|----------|---------|-------------|
| `single.image_path` | `./data/images/test/c7jxj5.jpg` | Image path |
| `single.text` | `"This frog sitting on a light"` | Text content |

#### Batch Prediction
| Variable | Default | Description |
|----------|---------|-------------|
| `batch.csv_file` | `./data/test.csv` | Input CSV |
| `batch.output_file` | `./data/test-predictions.csv` | Output CSV |
| `batch.root_dir` | `./data/images/test/` | Image root directory |
| `batch.n_rows` | `200` | Rows to process |

#### Preprocessing
| Variable | Default | Description |
|----------|---------|-------------|
| `text.max_length` | `128` | Max token length |
| `text.tokenizer` | `distilbert-base-uncased` | Tokenizer ID |
| `image.size` | `224` | Image size |
| `image.mode` | `val` | No augmentation |

#### Output Options
| Variable | Default | Description |
|----------|---------|-------------|
| `output.verbose` | `true` | Print detailed results |
| `output.save_probabilities` | `true` | Include raw probabilities |
| `output.save_confidence` | `true` | Include confidence scores |
| `output.threshold` | `0.5` | Classification threshold |

#### Batch Processing
| Variable | Default | Description |
|----------|---------|-------------|
| `batch_processing.batch_size` | `32` | Batch size |
| `batch_processing.num_workers` | `4` | DataLoader workers |

#### Model Architecture (Must Match Training)
| Variable | Default | Description |
|----------|---------|-------------|
| `freeze_vision` | `false` | Must match training config |
| `freeze_text` | `false` | Must match training config |
| `dropout_rate` | `0.3` | Must match training config |

---

## Training Workflow Examples

### Example 1: Full Fine-tuning from Checkpoint
```yaml
# configs/model/multimodal.yaml
load_model: true                    # ← Load previous model
best_model_name: best_model_1.6.pth # ← Source checkpoint
freeze_vision: false                # ← Fine-tune vision
freeze_text: false                  # ← Fine-tune text
dropout_rate: 0.5                   # ← High regularization

# configs/training/default.yaml
version: 1.8                        # ← New version identifier
learning_rate: 0.00002              # ← Low LR for stability
num_epochs: 5                       # ← Short fine-tuning
optimizer: adamw                    # ← Recommended
use_scheduler: true                 # ← Dynamic LR
```

**Result:** Loads v1.6 → Fine-tunes all layers → Saves as v1.8

---

### Example 2: Fast Experiment (Frozen Backbones)
```yaml
# configs/model/multimodal.yaml
load_model: true
best_model_name: best_model_1.6.pth
freeze_vision: true                 # ← Freeze EfficientNet
freeze_text: true                   # ← Freeze DistilBERT
dropout_rate: 0.3

# configs/training/default.yaml
version: 1.9_experiment
learning_rate: 0.0001               # ← Can use higher LR
num_epochs: 3                       # ← Quick test
```

**Result:** Only trains fusion layers → 4x faster → Quick prototyping

---

### Example 3: Training from Scratch
```yaml
# configs/model/multimodal.yaml
load_model: false                   # ← No checkpoint loading
freeze_vision: false
freeze_text: false
dropout_rate: 0.4

# configs/training/default.yaml
version: 2.0_scratch
learning_rate: 0.0001               # ← Higher LR OK
num_epochs: 20                      # ← More epochs needed
```

**Result:** Random initialization → Long training → Fresh start

---

## Command Line Overrides

Override any configuration parameter:

```bash
# Change learning rate and batch size
python src/training/train.py \
  training.learning_rate=0.0001 \
  training.batch_size=64

# Load different checkpoint
python src/training/train.py \
  model.load_model=true \
  model.best_model_name=best_model_1.5.pth \
  training.version=1.9

# Freeze backbones for fast training
python src/training/train.py \
  model.freeze_vision=true \
  model.freeze_text=true \
  training.num_epochs=3

# Full fine-tuning with low LR
python src/training/train.py \
  model.freeze_vision=false \
  model.freeze_text=false \
  training.learning_rate=0.00001 \
  model.dropout_rate=0.5

# Multiple parameters
python src/training/train.py \
  model.load_model=true \
  model.best_model_name=best_model_1.7.pth \
  training.version=1.8_custom \
  training.learning_rate=0.00002 \
  training.num_epochs=10 \
  model.dropout_rate=0.4
```

---

## Version History & Evolution

| Version | Source | dropout | freeze_vision | freeze_text | LR | Notes |
|---------|--------|---------|---------------|-------------|-----|-------|
| **1.6** | Scratch | 0.5 | true | true | 0.0001 | Baseline with high dropout |
| **1.7** | v1.6 | 0.3 | false | false |  0.00005 | Reduced dropout, new transforms |
| **1.8** | v1.6 | 0.5 | false | false |  0.00002 | Reverted to v1.6, alternative transforms |

**Typical Evolution Pattern:**
```
v1.6 (baseline) → saved as best_model_1.6.pth
  ↓ (load v1.6, modify dropout, train)
v1.7 (iteration) → saved as best_model_1.7.pth
  ↓ (load v1.6 again, different approach)
v1.8 (alternative) → saved as best_model_1.8.pth
```

---

## Quick Reference: Where Variables Are Used

### Critical Code Locations

**Model Loading:**
- `train.py:105-130` - Checkpoint loading logic
- `train.py:109` - `best_model_name` usage
- `train.py:110` - Checkpoint path construction

**Model Initialization:**
- `train.py:95-102` - MultimodalNet creation
- `train.py:100-101` - `freeze_vision`, `freeze_text` passed

**Optimizer & Scheduler:**
- `train.py:135-146` - Optimizer selection
- `train.py:151-159` - OneCycleLR scheduler creation
- `trainer.py:150-153` - Scheduler step per batch

**Checkpoint Saving:**
- `trainer.py:336-358` - Best model saving
- `trainer.py:347` - `version` used in filename
- `trainer.py:351-357` - Checkpoint contents

**MLflow Logging:**
- `trainer.py:244` - Experiment name
- `trainer.py:248-259` - Log all hyperparameters
- `trainer.py:370-378` - Log model artifact

---

## Best Practices

### 1. **Incremental Training**
```yaml
load_model: true
best_model_name: best_model_X.Y.pth  # Previous version
version: X.Z                          # New version
learning_rate: 0.00002                # Low LR
freeze_vision: false                  # Fine-tune if data sufficient
freeze_text: false
```

### 2. **Quick Experiments**
```yaml
freeze_vision: true
freeze_text: true
learning_rate: 0.0001     # Can be higher
num_epochs: 3             # Short runs
n_rows: 1000              # Subset data
```

### 3. **Final Production Model**
```yaml
freeze_vision: false
freeze_text: false
learning_rate: 0.00001    # Very low LR
num_epochs: 20            # More epochs
dropout_rate: 0.5         # High regularization
n_rows: -1                # Full dataset
```

### 4. **Preventing Catastrophic Forgetting**
- Always use `learning_rate ≤ 2e-5` when fine-tuning
- Enable `use_scheduler: true` for gradual LR decay
- Consider `weight_decay: 0.01` for regularization
- Monitor validation metrics to detect overfitting

---

## Troubleshooting

### Error: "Not found checkpoint"
**Cause:** `best_model_name` file doesn't exist in `checkpoint_dir`

**Solution:**
```yaml
# Option 1: Set load_model to false
load_model: false

# Option 2: Use correct checkpoint name
best_model_name: best_model_1.7.pth  # Verify file exists
```

### Poor Transfer Learning Results
**Cause:** Learning rate too high or backbones frozen

**Solution:**
```yaml
learning_rate: 0.00001   # Lower LR
freeze_vision: false      # Enable fine-tuning
freeze_text: false
```

### GPU Out of Memory
**Cause:** Large batch size or unfrozen backbones

**Solution:**
```yaml
batch_size: 16            # Reduce batch size
freeze_vision: true       # Freeze backbones
freeze_text: true
num_workers: 1            # Reduce workers
```

---

## Summary

**Most Important Variables for Transfer Learning:**

1. **`load_model`** - Enable checkpoint loading
2. **`best_model_name`** - Source model file
3. **`version`** - Target model version
4. **`freeze_vision` / `freeze_text`** - Control fine-tuning
5. **`learning_rate`** - Must be low (≤2e-5) for fine-tuning
6. **`dropout_rate`** - Regularization strength
7. **`optimizer`** - Use `adamw` for best results

These parameters work together to control the balance between:
- **Speed** (frozen backbones, high LR)
- **Accuracy** (fine-tuning, low LR)
- **Generalization** (dropout, weight decay)
