# Architecture Documentation

## System Overview

This is a  **Late-Fusion Multimodal Deep Learning Pipeline** for fake news detection using paired image-text data.

## Model Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────┐
│                    Multimodal Input                          │
│                  (Image + Text Pair)                         │
└──────────────────┬────────────────────┬─────────────────────┘
                   │                    │
         ┌─────────▼─────────┐  ┌──────▼──────────┐
         │  Vision Branch    │  │  Text Branch    │
         │  EfficientNet-B0  │  │  DistilBERT     │
         │  (pretrained)     │  │  (pretrained)   │
         └─────────┬─────────┘  └──────┬──────────┘
                   │                    │
         ┌─────────▼─────────┐  ┌──────▼──────────┐
         │  Projection Head  │  │  Projection Head│
         │  1280 → 512       │  │  768 → 512      │
         └─────────┬─────────┘  └──────┬──────────┘
                   │                    │
                   │   Image Emb.       │   Text Emb.
                   │   [B, 512]         │   [B, 512]
                   └────────┬───────────┘
                            │
                   ┌────────▼─────────┐
                   │  Concatenation   │
                   │  [B, 1024]       │
                   └────────┬─────────┘
                            │
                   ┌────────▼─────────┐
                   │  Fusion Layer    │
                   │  Dense + Dropout │
                   │  1024 → 256 → 1  │
                   └────────┬─────────┘
                            │
                   ┌────────▼─────────┐
                   │   Sigmoid        │
                   │   [B, 1]         │
                   └──────────────────┘
                            │
                            ▼
                    Prediction (0-1)
```

### Component Details

#### 1. Vision Branch (`src/models/vision_branch.py`)

**Architecture:**
```python
EfficientNet-B0 (pretrained)
├── Input: [B, 3, 224, 224]
├── Features: [B, 1280]
└── Projection:
    ├── Linear(1280, 512)
    ├── ReLU
    ├── Dropout(0.3)
    └── Linear(512, 512)
```

**Key Features:**
- Pretrained on ImageNet (transfer learning)
- Efficient architecture (low parameter count)
- Optional backbone freezing for faster training
- Custom projection head for embedding normalization

#### 2. Text Branch (`src/models/text_branch.py`)

**Architecture:**
```python
DistilBERT (pretrained)
├── Input: [B, seq_len] (token IDs + attention mask)
├── CLS Token: [B, 768]
└── Projection:
    ├── Linear(768, 512)
    ├── ReLU
    ├── Dropout(0.3)
    └── Linear(512, 512)
```

**Key Features:**
- Distilled version of BERT (40% smaller, 60% faster)
- Uses [CLS] token as sentence representation
- Optional backbone freezing
- Custom projection head for embedding normalization

#### 3. Fusion Layer (`src/models/multimodal_net.py`)

**Architecture:**
```python
Late Fusion
├── Concatenate: [B, 1024] (512 + 512)
├── Linear(1024, 256)
├── ReLU
├── Dropout(0.4)
├── Linear(256, 128)
├── ReLU
├── Dropout(0.2)
├── Linear(128, 1)
└── Sigmoid
```

**Design Choice: Late Fusion**
- **Why?** Allows each modality to learn rich representations independently
- **Alternative:** Early fusion (concatenate raw features) - less expressive
- **Alternative:** Cross-attention - more complex, slower

## Data Pipeline

### Dataset Class (`src/data/dataset.py`)

**Input Format:**
```csv
image_path,text_content,label
path/to/img1.jpg,"News text here...",0
path/to/img2.jpg,"Fake news text...",1
```

**Processing Pipeline:**

1. **Image Processing:**
   ```python
   Image.open() → RGB conversion
   → Resize(224x224)
   → Data Augmentation (train only)
     - Random horizontal flip
     - Random rotation (±15°)
     - Color jitter
   → ToTensor
   → Normalize(ImageNet stats)
   ```

2. **Text Processing:**
   ```python
   Raw text
   → DistilBertTokenizer
   → Add [CLS], [SEP] tokens
   → Truncate/Pad to max_length
   → Return input_ids + attention_mask
   ```

3. **Output Batch:**
   ```python
   {
       'image': Tensor[B, 3, 224, 224],
       'input_ids': Tensor[B, max_length],
       'attention_mask': Tensor[B, max_length],
       'label': Tensor[B]
   }
   ```

## Training Pipeline

### Training Loop (`src/training/trainer.py`)

**Components:**

1. **Loss Function:** Binary Cross-Entropy (BCE)
   ```python
   loss = BCE(predictions, labels)
   ```

2. **Optimizer:** AdamW (default)
   ```python
   AdamW(lr=1e-4, weight_decay=0.01)
   ```

3. **Learning Rate Scheduler:** 
   ```python
   OneCycleLR(T_max=num_epochs, anneal_strategy='cos')
   ```

### MLflow Integration

**Tracked Metrics:**
- Training/Validation Loss
- Accuracy, Precision, Recall, F1
- AUC-ROC
- Learning Rate

**Logged Artifacts:**
- Best model checkpoint (`.pth`)
- Model architecture (PyTorch format)
- Configuration files (YAML)

## Configuration Management

### Hydra Structure

```
configs/
├── config.yaml              # Main config (orchestrates others)
├── model/
│   └── multimodal.yaml     # Model hyperparameters
├── data/
│   └── fakeddit.yaml       # Data paths and preprocessing
└── training/
    └── default.yaml        # Training hyperparameters
```

**Composable Configuration:**
```bash
# Override any parameter from CLI
python train.py \
    model.embedding_dim=1024 \
    training.batch_size=64 \
    data.max_length=256
```

## Performance Characteristics

### Model Size

| Component | Parameters |
|-----------|------------|
| EfficientNet-B0 | ~4M |
| DistilBERT | ~66M |
| Projection Heads | ~1M |
| Fusion Layer | ~0.5M |
| **Total** | **~71M** |

### Memory Requirements

| Batch Size | GPU Memory (Mixed Precision) |
|------------|------------------------------|
| 8 | ~4 GB |
| 16 | ~6 GB |
| 32 | ~10 GB |
| 64 | ~18 GB |

### Training Speed (V100 GPU)

| Configuration | Samples/sec |
|---------------|-------------|
| Frozen backbones | ~80 |
| Fine-tuning | ~40 |

## Design Decisions

### Why Late Fusion?

**Pros:**
- Each modality learns independently
- Easier to debug and interpret
- Better for imbalanced modalities
- Simpler architecture

**Cons:**
- No cross-modal interactions during feature extraction
- May miss early fusion patterns

**Alternative Considered:** Cross-modal attention
- More complex
- Harder to train
- Potentially better performance (future work)

### Why EfficientNet-B0?

**Pros:**
- Efficient (good accuracy/parameter ratio)
- Fast inference
- Pretrained on ImageNet
- Standard input size (224x224)

**Alternatives:**
- ResNet50: Larger, slower
- ViT: Requires more data
- MobileNet: Less accurate

### Why DistilBERT?

**Pros:**
- 40% smaller than BERT
- 60% faster
- 97% of BERT's performance
- Good for production

**Alternatives:**
- BERT: Larger, slower
- RoBERTa: Better but larger
- TinyBERT: Faster but less accurate

## Extension Points

### Adding New Modalities

```python
# Add audio branch
class AudioBranch(nn.Module):
    def __init__(self):
        self.wav2vec = Wav2Vec2Model.from_pretrained(...)
        self.projection = nn.Linear(768, 512)

# Update fusion
fused = torch.cat([image_emb, text_emb, audio_emb], dim=1)
```

### Custom Fusion Strategies

```python
# Cross-attention fusion
class CrossAttentionFusion(nn.Module):
    def __init__(self):
        self.attention = nn.MultiheadAttention(512, num_heads=8)

    def forward(self, image_emb, text_emb):
        # Cross-modal attention
        attended_features, _ = self.attention(
            query=image_emb,
            key=text_emb,
            value=text_emb
        )
        return attended_features
```


---

## Source Files Reference

### Data Module (`src/data/`)

#### `dataset.py`
Custom PyTorch Dataset implementation for multimodal fake news detection.
- **Class:** `MultimodalDataset`
- **Features:**
  - Loads image-text pairs from CSV files
  - Handles image preprocessing and text tokenization
  - Supports flexible column naming
  - Returns batches with images, tokenized text, and labels
- **Usage:** Training and inference data loading

#### `load_data.py`
Data loading utilities and helper functions.
- CSV file processing
- Data validation and error handling
- Path resolution for image files

#### `transforms.py`
Image transformation pipelines using torchvision.
- **Training transforms:** RandomHorizontalFlip, RandomRotation, ColorJitter, normalization
- **Validation/Test transforms:** Resize, normalization
- ImageNet statistics normalization
- Returns transforms for different modes (train/val/test)

---

### Models Module (`src/models/`)

#### `multimodal_net.py`
Main multimodal architecture with late fusion strategy.
- **Class:** `MultimodalNet`
- **Architecture:**
  - Combines vision and text branches
  - Late fusion: concatenates embeddings (1024-dim)
  - Fusion network: 1024 → 256 → 128 → 1
  - Dropout layers for regularization
- **Parameters:**
  - `embedding_dim`: 512 (default)
  - `fusion_hidden_dim`: 256 (default)
  - `dropout_rate`: 0.4 (default)
  - `freeze_vision`, `freeze_text`: Transfer learning control

#### `vision_branch.py`
Vision encoder using EfficientNet-B0.
- **Class:** `VisionBranch`
- **Architecture:**
  - Pretrained EfficientNet-B0 backbone (ImageNet)
  - Global Average Pooling: 1280-dim features
  - Projection head: 1280 → 512 → 512
  - ReLU activation and dropout
- **Features:**
  - Optional backbone freezing for transfer learning
  - ~4M parameters

#### `text_branch.py`
Text encoder using DistilBERT.
- **Class:** `TextBranch`
- **Architecture:**
  - Pretrained DistilBERT-base-uncased
  - [CLS] token embedding: 768-dim
  - Projection head: 768 → 512 → 512
  - ReLU activation and dropout
- **Features:**
  - Optional backbone freezing
  - ~66M parameters
  - Handles input_ids and attention_mask

---

### Training Module (`src/training/`)

#### `train.py`
Main training script with Hydra configuration.
- **Functions:**
  - Loads configuration via Hydra decorators
  - Initializes dataloaders, model, optimizer
  - Sets up MLflow experiment tracking
  - Executes training loop via MultimodalTrainer
  - Saves best model checkpoints
- **Features:**
  - Two-phase training (frozen → fine-tuned)
  - OneCycleLR learning rate scheduling
  - Automatic device detection (CPU/GPU)

#### `trainer.py`
Training loop implementation with MLflow integration.
- **Class:** `MultimodalTrainer`
- **Features:**
  - Training and validation epochs
  - Gradient clipping and mixed precision support
  - MLflow logging (metrics, parameters, artifacts)
  - Model checkpointing (best model based on validation loss)
  - Progress tracking with tqdm
  - Comprehensive metric calculation

#### `visualization.py`
Training visualization utilities.
- Plotting functions for loss curves
- Metric visualization
- Confusion matrix plotting
- ROC curve generation

---

### Utils Module (`src/utils/`)

#### `logging.py`
Logging configuration and setup.
- **Function:** `setup_logger()`
- **Features:**
  - Configurable log levels
  - File and console handlers
  - Formatted output with timestamps
  - Logger name customization

#### `metrics.py`
Performance metrics calculation.
- **Metrics implemented:**
  - Accuracy
  - Precision, Recall, F1-Score
  - ROC-AUC
  - Confusion Matrix
- **Functions:**
  - `calculate_metrics()`: Computes all metrics from predictions
  - Binary classification metric utilities

---

## References

- EfficientNet: [Tan & Le, 2019](https://arxiv.org/abs/1905.11946)
- DistilBERT: [Sanh et al., 2019](https://arxiv.org/abs/1910.01108)
- Fakeddit Dataset: [Nakamura et al., 2020](https://arxiv.org/abs/1911.03854)
