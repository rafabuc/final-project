# Multimodal Fake News Detection System

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-green.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Version](https://img.shields.io/badge/Version-1.7-red.svg)

A deep learning system that combines visual and textual information to detect fake news using a multimodal neural network architecture with MLflow tracking, Docker deployment, and Kubernetes orchestration.

---

## Table of Contents

1. [The Problem](#the-problem)
2. [The Solution](#the-solution)
3. [Model Architecture](#model-architecture)
4. [Results](#-results)
5. [Performance](#performance)
6. [Project Structure](#project-structure)
7. [Source Code Overview](#source-code-overview)
8. [Key Components](#key-components)
9. [Installation](#installation)
10. [Configuration](#configuration)
11. [Training](#training)
12. [Inference](#inference)
13. [Deployment](#deployment)
14. [Data and Model Versioning](#data-and-model-versioning)
15. [Architecture Details](#architecture-details)
16. [Troubleshooting](#troubleshooting)
17. [Advanced Usage](#advanced-usage)
18. [Best Practices](#best-practices)
19. [References](#references)
20. [Contact & Support](#contact--support)
21. [Acknowledgments](#acknowledgments)

---

## The Problem

In the modern digital landscape, misinformation often manifests not just as fake text or manipulated images, but as a **semantic conflict** between the two. A real image can be captioned with a lie, or a real news story can be illustrated with a misleading image.

Traditional unimodal models (Text-only or Image-only) fail to catch these subtleties:
* **Text Models** miss visual context.
* **Image Models** cannot verify the factual claims in the caption.

---

## The Solution

This system treats the detection task as a **Multimodal Classification Problem**. It processes the visual and textual streams independently to extract high-level feature representations, which are then fused to learn the cross-modal relationship. If the image and text features do not align semantically, the model classifies the content as **FAKE**.

### Key Features

- **Multimodal Architecture**: Combines EfficientNet-B0 (vision) + DistilBERT (text)
- **Late Fusion Strategy**: Independent feature learning with cross-modal fusion
- **MLflow Integration**: Complete experiment tracking and model versioning
- **Production Ready**: Docker containers, Kubernetes deployment, ONNX export
- **Data Versioning**: DVC for datasets, Git LFS for model checkpoints
- **Hydra Configuration**: Flexible, composable configuration management

---

## Model Architecture

### High-Level Design

```
                            INPUT DATA
                                |
                    +-----------+-----------+
                    |                       |
                  IMAGE                   TEXT
                    |                       |
            +-------v-------+       +-------v--------+
            | EfficientNet  |       | DistilBERT     |
            | (Pre-trained) |       | (Pre-trained)  |
            +-------+-------+       +-------+--------+
                    |                       |
            [1280 features]         [768 features]
                    |                       |
            +-------v-------+       +-------v--------+
            | Vision FC     |       | Text FC        |
            | 1280 -> 512   |       | 768 -> 512     |
            +-------+-------+       +-------+--------+
                    |                       |
                    +----------+------------+
                               |
                        FUSION LAYER
                               |
                    +----------v-----------+
                    |  Concatenate (1024)  |
                    +----------+-----------+
                               |
                    +----------v-----------+
                    |   Fusion Network     |
                    |   1024 -> 256 -> 1   |
                    +----------+-----------+
                               |
                          Sigmoid
                               |
                        OUTPUT (0-1)
                    [0: Real, 1: Fake]
```

---

> **📖 For detailed architecture documentation, see [ARCHITECTURE.md](ARCHITECTURE.md)**
>
> This document covers:
> - Component specifications (Vision Branch, Text Branch, Fusion Layer)
> - Design decisions and architectural choices
> - Fusion strategies (why late fusion vs early fusion)
> - Implementation details and parameter counts

---

### Architecture Components

#### 1. Vision Branch (The "Eye")
* **Backbone:** EfficientNet-B0 (Pretrained on ImageNet)
* **Input:** RGB Images resized to 224x224 and normalized
* **Processing:** Features extracted from Global Average Pooling (1280-dim) → 512-dim embedding space

#### 2. Text Branch (The "Brain")
* **Backbone:** DistilBERT-base-uncased (Pretrained on Wikipedia)
* **Input:** Tokenized text (padded to 128 tokens)
* **Processing:** [CLS] token embedding (768-dim) → 512-dim embedding space

#### 3. Fusion & Classification
* **Fusion:** Visual and text embeddings concatenated (512+512=1024)
* **Classifier:** Fully Connected Network (Dense Layers + ReLU + Dropout) → single logit for binary classification

### Training Strategy

To prevent "Catastrophic Forgetting" of the pretrained weights, we employed a **Two-Stage Transfer Learning** strategy:

**Phase 1 (Head Warming):**
The backbones (EfficientNet/DistilBERT) were frozen. Only the fusion head was trained to align the random weights with the pretrained features.

```yaml
freeze_vision: true
freeze_text: true
learning_rate: 0.0001
```
**Best Model:** `src/training/checkpoints/best_model_1.6.pth`

**Phase 2 (Fine-Tuning):**
The entire model was unfrozen and trained with a very low learning rate to fine-tune the feature extractors for fake news patterns.

```yaml
freeze_vision: false
freeze_text: false
learning_rate: 0.00005
```
**Best Model:** `src/training/checkpoints/best_model_1.7.pth`

## 📊 Results

| Metric | Phase 1 (Frozen) | Phase 2 (Fine-Tuned) |
| :--- | :---: | :---: |
| **Validation Accuracy** | ~74.1% | **84.2%** |
| **Training Accuracy** | ~72.0% | **92.6%** |


---

## Project Structure

```
final-project/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py           # MultimodalDataset class
│   │   ├── load_data.py         # Data loading utilities
│   │   └── transforms.py        # Image transformations
│   ├── models/
│   │   ├── __init__.py
│   │   ├── multimodal_net.py    # MultimodalNet architecture
│   │   ├── vision_branch.py     # EfficientNet branch
│   │   └── text_branch.py       # DistilBERT branch
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py             # Main training script
│   │   ├── trainer.py           # MultimodalTrainer with MLflow
│   │   ├── visualization.py     # Training visualization
│   │   ├── mlflow.db            # MLflow database
│   │   ├── checkpoints/         # Model checkpoints (Git LFS)
│   │   └── mlruns/              # MLflow experiment tracking
│   └── utils/
│       ├── __init__.py
│       ├── logging.py           # Logger setup
│       └── metrics.py           # Performance metrics
│
├── configs/                     # Hydra configuration files
│   ├── config.yaml             # Main config
│   ├── config_inference.yaml   # Inference config
│   ├── model/                  # Model configs
│   ├── data/                   # Data configs
│   ├── training/               # Training configs
│   └── inference/              # Inference configs
│
├── data/                        # Dataset directory (DVC tracked)
│   ├── images/                 # Image files
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── train.csv               # Training data
│   ├── val.csv                 # Validation data
│   ├── test.csv                # Test data
│   └── test-predictions.csv    # Test predictions (generated by inference.py in batch mode)
│
├── notebooks/                   # Jupyter notebooks
│   └── inference.ipynb         # Inference examples and API usage
│
│
├── mlruns/                      # MLflow experiment tracking (root level)
│
├── outputs/                     # Hydra output directory

├── inference.py                 # Inference script (Hydra and PyTorch)
├── api.py                       # FastAPI REST API (PyTorch)
├── inference-onnx.py            # Inference script (Hydra and ONNX Runtime)
├── api-onnx.py                  # FastAPI REST API (ONNX Runtime)
├── export_onnx.py               # PyTorch → ONNX converter
├── multimodal_model.onnx        # ONNX model Tracked with DVC
├── multimodal_model.onnx.data   # ONNX model Tracked with DVC
│
├── Dockerfile-torch             # Docker (PyTorch)
├── Dockerfile-onnx              # Docker (ONNX Runtime)
├── k8s-onnx-deployment.yaml     # Kubernetes deployment (ONNX Runtime)
│
├── requirements-dev.txt         # Development dependencies
├── requirements-deploy-torch.txt # PyTorch deployment (to build docker pytorch image)
├── requirements-deploy-onnx.txt  # ONNX deployment (to build docker onnx image)
│
├── .gitignore                   # Git ignore file
├── .gitattributes               # Git attributes file
├── .dvc                         # DVC configuration
├── .dvc-ignore                  # DVC ignore file
│
├── architecture_diagram.png     # Architecture diagram
├── ARCHITECTURE.md              # Detailed architecture documentation
├── INFERENCE_INSTRUCTIONS.md    # Inference instructions

└── README.md                    # This file
```

---

## Source Code Overview

> **📖 For detailed source code documentation, see [ARCHITECTURE.md - Source Files Reference](ARCHITECTURE.md#source-files-reference)**

**Quick Overview:**
- **Data Module** (`src/data/`): Dataset class, data loaders, image transforms
- **Models Module** (`src/models/`): MultimodalNet, VisionBranch, TextBranch
- **Training Module** (`src/training/`): Training script, trainer class, MLflow integration
- **Utils Module** (`src/utils/`): Logging, metrics calculation

See ARCHITECTURE.md for complete details on each module and class.

---

## Key Components

### 1. Data Pipeline
- **MultimodalDataset**: Custom PyTorch Dataset that loads images and text
- **Image Transforms**: Data augmentation for training (RandomHorizontalFlip, RandomRotation, ColorJitter) and validation
- **Text Tokenization**: DistilBERT tokenizer with max_length=128

### 2. Model Architecture
- **Vision Encoder**: EfficientNet-B0 (pre-trained on ImageNet)
  - Output: 1280-dim features -> 512-dim embeddings
- **Text Encoder**: DistilBERT (pre-trained)
  - Output: 768-dim features -> 512-dim embeddings
- **Fusion Network**:
  - Concatenates vision and text embeddings (1024-dim)
  - Two fully connected layers (1024 -> 256 -> 1)
  - Dropout for regularization
  - Sigmoid activation for binary classification

### 3. Training Pipeline
- **Loss Function**: Binary Cross-Entropy Loss (BCELoss)
- **Optimizer**: Adam or AdamW with weight decay
- **Learning Rate Scheduler**: Cosine Annealing
- **Gradient Clipping**: max_norm=1.0 for stability
- **MLflow Tracking**: Logs hyperparameters, metrics, and model artifacts

### 4. Metrics
- Accuracy
- Precision, Recall, F1-Score
- ROC-AUC
- Confusion Matrix


---

## Installation

### Prerequisites

- Python 3.11
- CUDA 11.0+ (for GPU support)
- 8GB+ RAM
- 4GB+ GPU memory (recommended)
- Git, 
- DVC (for data versioning)

### Step 1: Clone Repository

```bash

# Clone repository
git clone <repository-url>
cd final-project
```

### Step 2: Download Data and Models


This project uses DVC (Data Version Control) to manage large datasets and artifacts that cannot be stored directly on GitHub.

### Prerequisites
```bash
pip install dvc[s3]
```

### Setup and Download

1. **Configure DVC remote:**
```bash
dvc remote add -d myremote s3://q-reg/artifact-final-project
```

2. **Configure anonymous access (public bucket):**
```bash
dvc remote modify myremote anon true
```

3. **Download the data:**
```bash
# Download all files
dvc pull

# Or download specific files
dvc pull multimodal_model.onnx.dvc
dvc pull multimodal_model.onnx.data.dvc
dvc pull data.dvc
dvc pull src/training/checkpoints.dvc
dvc pull src/training/outputs.dvc
```

### Data Structure

After running `dvc pull`, you will have:
```
.
├── data/
    └── images/train/     # Image dataset
    └── images/test/      # Image dataset
    └── images/val/       # Image dataset
    └── train.csv         # Training dataset
    └── test.csv          # Test dataset
    └── val.csv           # Validation dataset
    └── test-predictions.csv  # Test predictions (generated by inference.py in batch mode)


├── multimodal_model.onnx          # ONNX model file
├── multimodal_model.onnx.data     # ONNX model data
└── src/
    └── training/
        ├── checkpoints/           # Training checkpoints
            ├── best_model_1.6.pth      # Best model checkpoint in phase 1 (freezing vision encoder)

            ├── best_model_1.7.pth      # Best model checkpoint in phase 2 (fine-tuning vision   encoder, takes best_model_1.6.pth as initial checkpoint) FINAL MODEL
            
            ├── best_model_1.8.pth      # Improved model checkpoint in phase 2 (fine-tuning vision encoder takes best_model_1.7.pth as initial checkpoint)
            
        └── outputs/              # Training outputs
```

### Troubleshooting

If you encounter issues downloading:
```bash
# Verify remote configuration
dvc remote list

# Check connection to remote
dvc remote list --verbose

# Force re-download
dvc pull --force

### Step 3: Create Virtual Environment

```bash
# Using conda (recommended)
conda create -n fakenews python=3.11
conda activate fakenews

# OR using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 4: Install Dependencies

```bash
# Development (includes all dependencies)
pip install -r requirements-dev.txt


```

---

## Configuration

The project uses **Hydra** for hierarchical configuration management.

### 📚 Complete Configuration Documentation

**⚠️ IMPORTANT:** Before training or modifying the model, please review the comprehensive configuration guide:

**➡️ [📖 CONFIG.md - Complete Configuration Reference](CONFIG.md)**

This essential guide covers:
- ✅ **Model Loading & Versioning** - How to load checkpoints and version models (`load_model`, `best_model_name`, `version`)
- ✅ **Transfer Learning Controls** - Freeze/unfreeze layers for speed vs accuracy (`freeze_vision`, `freeze_text`)
- ✅ **Training Parameters** - Learning rate, dropout, optimizer settings and their impact
- ✅ **Performance Impact Analysis** - How each parameter affects training speed, GPU memory, and accuracy
- ✅ **Workflow Examples** - Full fine-tuning, frozen backbones, incremental training scenarios
- ✅ **Best Practices** - Preventing catastrophic forgetting, hyperparameter tuning strategies
- ✅ **Command Line Overrides** - How to override any configuration from the command line
- ✅ **Troubleshooting** - Common configuration issues and solutions

### ⚡ Quick Configuration Reference

**Critical Variables for Incremental Training:**

```yaml
# configs/model/multimodal.yaml
load_model: true                    # ← Load previous checkpoint before training
best_model_name: best_model_1.7.pth # ← Source checkpoint file to start from
version: 1.8                        # ← New version identifier for this training run

freeze_vision: false                # ← Fine-tune vision backbone (EfficientNet)
freeze_text: false                  # ← Fine-tune text backbone (DistilBERT)
dropout_rate: 0.3                   # ← Regularization strength (0.3-0.5)

# configs/training/default.yaml
learning_rate: 0.00002              # ← Low LR for fine-tuning (2e-5)
optimizer: adamw                    # ← Recommended for transfer learning
use_scheduler: true                 # ← Dynamic LR with OneCycleLR
num_epochs: 5                       # ← Training epochs
batch_size: 32                      # ← Samples per batch
```

**Performance Impact Quick Guide:**
- `freeze_vision=true` → ⚡ **2x faster training**, 💾 **40% less GPU memory**, 📊 Lower accuracy
- `freeze_text=true` → ⚡ **2.5x faster training**, 💾 **50% less GPU memory**, 📊 Lower accuracy
- `learning_rate ≤ 2e-5` → 🔒 **Required for stable fine-tuning** (prevents catastrophic forgetting)
- `dropout_rate: 0.5` → 📈 Better generalization, lower training accuracy
- `dropout_rate: 0.3` → 📊 Higher training accuracy, possible overfitting

**👉 See [CONFIG.md](CONFIG.md) for complete parameter documentation with code references and examples.**

### Configuration Files Structure

```
configs/
├── config.yaml              # Main training configuration
├── config_inference.yaml    # Inference configuration
├── model/
│   └── multimodal.yaml     # Model architecture & checkpointing ⭐
├── data/
│   └── fakeddit.yaml       # Dataset paths & preprocessing
├── training/
│   └── default.yaml        # Training hyperparameters ⭐
└── inference/
    └── default.yaml        # Inference settings
```

---

## Training

### Start Training

```bash
# Basic training
python src/training/train.py

```

### Monitor with MLflow

```bash
# In a separate terminal
cd src/training
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

# Open browser: http://localhost:5000
```

### Training Pipeline

**Components:**

1. **Loss Function:** Binary Cross-Entropy (BCE)
2. **Optimizer:** AdamW with weight decay
3. **Learning Rate Scheduler:** OneCycleLR with cosine annealing
4. **Gradient Clipping:** max_norm=1.0 for stability

**MLflow Tracking:**
- Training/Validation Loss, Accuracy, Precision, Recall, F1, AUC-ROC
- Model checkpoints and configuration files
- Learning rate progression

---

## Inference

---

> **📖 Complete Inference Guide: [INFERENCE_INSTRUCTIONS.md](INFERENCE_INSTRUCTIONS.md)**
>
> Comprehensive documentation covering:
> - 🔍 **Single Sample Prediction** - How to predict on individual image-text pairs
> - 📊 **Batch Prediction** - Processing multiple samples from CSV files
> - 🚀 **API Usage** - FastAPI endpoints (PyTorch and ONNX versions)
> - ⚡ **ONNX Inference** - High-performance production deployment
> - 🐛 **Troubleshooting** - Common issues and solutions
> - 💡 **Best Practices** - Tips for optimal inference performance

---

The inference system supports both single sample and batch predictions using Hydra configuration.

### Single Sample Prediction

```bash
python inference.py \
  inference.mode=single \
  inference.single.image_path=data/images/test.jpg \
  inference.single.text="Breaking news article text here"
```

**Example Output:**
```
Prediction: REAL
Probability (Fake): 0.1527
Confidence: 0.8473
```

### Batch Prediction

```bash
python inference.py \
  inference.mode=batch \
  inference.batch.csv_file=data/test.csv \
  inference.batch.output_file=predictions.csv
```

**CSV Format:**
```csv
text_content,image_path,label
"News text...",path/to/img1.jpg,0
"Fake news...",path/to/img2.jpg,1
```

**Output:** `predictions.csv` with columns:
- `prediction`: "FAKE" or "REAL"
- `label`: 1 (FAKE) or 0 (REAL)
- `probability`: Raw probability of being fake (0-1)
- `confidence`: Confidence in prediction (0-1)



---

## Deployment

### 1. REST API (Development)

#### Install Dependencies

```bash
pip install fastapi uvicorn python-multipart
```

#### Run API Server

```bash
# Start server with auto-reload
uvicorn api:app --reload

# Server runs on http://localhost:8000
```

#### Test via Swagger UI

1. Open: `http://localhost:8000/docs`
2. Click POST `/predict` endpoint
3. Click "Try it out"
4. Upload image and enter text
5. Click "Execute"
6. See test.docx for example output
#### Test via cURL

```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'text=Breaking news test' \
  -F 'image=@/path/to/image.jpg'
```

**Response:**
```json
{
  "filename": "test_image.jpg",
  "prediction": "FAKE",
  "probability_fake": 0.6505,
  "confidence": 0.6505,
  "label_id": 1
}
```

---

### 2. Docker Deployment

#### Option A: PyTorch Version (Full Features)

```bash
# Build image (~12.1GB)
docker build -t fake-news-api:v1_torch -f Dockerfile-torch .

# Run container
docker run -d -p 8000:8000 --name fake-news-api fake-news-api:v1_torch

# Monitor logs
docker logs -f fake-news-api

# Test API
curl -X POST http://localhost:8000/predict \
  -F 'text=Test news' \
  -F 'image=@test.jpg'
```

#### Option B: ONNX Version (Production, Lightweight)

**Benefits:**
- 94% smaller (754MB vs 12.1GB)
- Faster startup and inference
- Lower memory footprint

**Export to ONNX:**

```bash
# Convert PyTorch model to ONNX
python export_onnx.py

# Output: multimodal_model.onnx
```

**Build and Run:**

```bash
# Build ONNX image (~754MB)
docker build -t fake-news-api:v1_onnx -f Dockerfile-onnx .

# Run container
docker run -d -p 8000:8000 --name fake-news-api fake-news-api:v1_onnx

# Same API interface as PyTorch version
```

---

### 3. Kubernetes Deployment

#### Prerequisites

```bash
# Start Minikube
minikube start
minikube status
```

#### Load Image into Minikube

```bash
# Transfer Docker image to Minikube
minikube image load fake-news-api:v1_onnx

# Verify
minikube ssh -- docker images
```

#### Deploy to Kubernetes

```bash
# Apply deployment
kubectl apply -f k8s-onnx-deployment.yaml

# Check status
kubectl get all

# Expected output:
# pod/fake-news-deployment-xxx   1/1   Running
# service/fake-news-service      NodePort   10.x.x.x   <none>   80:31018/TCP
```

#### Access the Service

**Option 1: Port Forwarding**
```bash
kubectl port-forward service/fake-news-service 8000:80

# Access: http://localhost:8000
```

**Option 2: Minikube Service URL**
```bash
minikube service fake-news-service --url

# Returns: http://192.168.49.2:31018
```

**Option 3: Minikube Tunnel**
```bash
minikube tunnel

# Provides external IP access (requires sudo)
```

#### Test Deployment

```bash
curl -X POST http://localhost:8000/predict \
  -F 'text=This frog sitting on a light' \
  -F 'image=@test.jpg'
```

---

### Deployment Comparison

| Method | Use Case | Size | Speed | Complexity |
|--------|----------|------|-------|------------|
| **Local Script** | Development, testing | N/A | Fast | Low |
| **API (Local)** | Development, testing | N/A | Fast | Low |
| **Docker (PyTorch)** | Full features, debugging | 12.1GB | Medium | Medium |
| **Docker (ONNX)** | Production, efficiency | 754MB | Fast | Medium |
| **Kubernetes** | Scalability, production | 754MB | Fast | High |

---

## Data and Model Versioning

### Strategy Overview

| Artifact Type | Location | Size | Strategy | Tool |
|---------------|----------|------|----------|------|
| **Training Images** | `data/images/` | Large (GB) | Remote Storage | DVC |
| **Model Checkpoints** | `src/training/checkpoints/` | Medium (100MB-2GB) | Version Control | Git LFS |
| **MLflow Runs** | `src/training/mlruns/` | Growing | Ignored + Backup | .gitignore |
| **Final Models** | Releases | Medium-Large | GitHub Releases | Git Tags |

---

### DVC for Training Images

**Why DVC?**
- Designed for large datasets
- Works with cloud storage (S3, Google Drive, Azure)
- Lightweight metadata in Git
- Easy sharing and reproduction

#### Initial Setup

```bash
# Install DVC
pip install dvc dvc-gdrive

# Initialize DVC
dvc init

# Configure remote storage (Google Drive)
dvc remote add -d gdrive gdrive://YOUR_GOOGLE_DRIVE_FOLDER_ID

# Track images
dvc add data/images

# Commit metadata
git add data/images.dvc .gitignore .dvc/config
git commit -m "Add training images to DVC tracking"

# Push data to remote
dvc push

# Push metadata to Git
git push
```

#### Daily Workflow

```bash
# 1. Add/modify images in data/images/
# 2. Update DVC tracking
dvc add data/images

# 3. Commit changes
git add data/images.dvc
git commit -m "Update training dataset - added 500 new images"

# 4. Push to both remotes
dvc push  # Data to DVC remote
git push  # Metadata to GitHub
```

#### Download Data (New User)

```bash

# Pull data from DVC
dvc pull

# Verify
ls data/images/
```

---



#### Daily Workflow



---

### Quick Reference Commands

#### DVC Commands

```bash
dvc add data/images          # Track data with DVC
dvc push                     # Upload to remote
dvc pull                     # Download from remote
dvc status                   # Check DVC status
dvc checkout                 # Restore data versions
```

#### Git LFS Commands

```bash
git lfs track "*.pth"        # Track file pattern
git lfs ls-files             # List LFS files
git lfs pull                 # Download LFS files
git lfs prune                # Clean LFS cache
```

#### Version Control

```bash
# Checkout specific data version
git checkout v1.0
dvc checkout
```

---

### File Size & Storage Guidelines

**File Size Recommendations:**

| Size | Recommendation | Tool |
|------|----------------|------|
| < 10MB | Regular Git | Git |
| 10MB - 100MB | Git (with caution) or Git LFS | Git / Git LFS |
| 100MB - 2GB | Git LFS | Git LFS |
| > 2GB | DVC | DVC |
| Datasets | Always DVC | DVC |

**Storage Options:**

| Option | Free Tier | Best For | Use Case |
|--------|-----------|----------|----------|
| **Google Drive** | 15GB | Personal projects | DVC remote for small teams |
| **AWS S3** | 5GB (12mo) | Production | Scalable, reliable storage |
| **GitHub LFS** | 1GB | Small checkpoints | Model files < 2GB |
| **Local Storage** | Unlimited | Testing | Development and debugging |

**Recommendation:** Use DVC with S3 for production, Google Drive for personal projects.

---

## Architecture Details

---

> **📖 For comprehensive architecture documentation, see [ARCHITECTURE.md](ARCHITECTURE.md)**

---

### Component Specifications

#### 1. Vision Branch (`src/models/vision_branch.py`)

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

**Features:**
- Pretrained on ImageNet
- Efficient architecture (~4M parameters)
- Optional backbone freezing
- Custom projection head for embedding normalization

#### 2. Text Branch (`src/models/text_branch.py`)

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

**Features:**
- Distilled BERT (40% smaller, 60% faster)
- Uses [CLS] token representation
- ~66M parameters
- Optional backbone freezing

#### 3. Fusion Layer (`src/models/multimodal_net.py`)

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
- Allows each modality to learn independently
- Easier to debug and interpret
- Better for imbalanced modalities
- Simpler than cross-attention alternatives

---

### Data Pipeline

**Image Processing:**
```python
Image.open() → RGB conversion
→ Resize(224x224)
→ Data Augmentation (train only):
  - Random horizontal flip
  - Random rotation (±15°)
  - Color jitter
→ ToTensor
→ Normalize(ImageNet stats)
```

**Text Processing:**
```python
Raw text
→ DistilBertTokenizer
→ Add [CLS], [SEP] tokens
→ Truncate/Pad to max_length
→ Return input_ids + attention_mask
```

**Output Batch:**
```python
{
    'image': Tensor[B, 3, 224, 224],
    'input_ids': Tensor[B, max_length],
    'attention_mask': Tensor[B, max_length],
    'label': Tensor[B]
}
```

---

### Design Decisions

#### Why Late Fusion?

**Pros:**
- Each modality learns rich representations independently
- Easier to debug and interpret
- Better for imbalanced modalities
- Simpler architecture

**Cons:**
- No cross-modal interactions during feature extraction
- May miss early fusion patterns

**Alternative Considered:** Cross-modal attention (more complex, potentially better performance)

#### Why EfficientNet-B0?

**Pros:**
- Efficient (good accuracy/parameter ratio)
- Fast inference
- Pretrained on ImageNet
- Standard input size (224x224)

**Alternatives:** ResNet50 (larger), ViT (requires more data), MobileNet (less accurate)

#### Why DistilBERT?

**Pros:**
- 40% smaller than BERT
- 60% faster inference
- 97% of BERT's performance
- Good for production

**Alternatives:** BERT (larger, slower), RoBERTa (better but larger), TinyBERT (faster but less accurate)

---

## Performance

### Model Size

| Component | Parameters |
|-----------|------------|
| EfficientNet-B0 | ~4M |
| DistilBERT | ~66M |
| Projection Heads | ~1M |
| Fusion Layer | ~0.5M |
| **Total** | **~71M** |

### Expected Results

| Metric | Phase 1 (Frozen) | Phase 2 (Fine-Tuned) |
|--------|------------------|----------------------|
| **Validation Accuracy** | ~74.1% | **84.2%** |
| **Training Accuracy** | ~72.0% | **92.6%** |
| **F1-Score** | ~0.70 | **0.83-0.90** |
| **ROC-AUC** | ~0.80 | **0.90-0.95** |

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

---

## Troubleshooting

### Training Issues

#### CUDA Out of Memory

```bash
# Reduce batch size
python src/training/train.py training.batch_size=16

# Freeze backbones
python src/training/train.py model.freeze_vision=true model.freeze_text=true
```

#### Slow Training

```bash
# Increase workers
python src/training/train.py training.num_workers=8

# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

#### Poor Performance

- Check data quality and balance
- Increase num_epochs
- Adjust learning_rate and weight_decay
- Try different fusion_hidden_dim values

---

### Inference Issues

#### Checkpoint Not Found

```bash
# Check checkpoint path
ls -lh src/training/checkpoints/

# Use correct path
python inference.py inference.checkpoint.path=checkpoints/your_model.pth
```

#### Column Not Found in CSV

```bash
# Check CSV columns
head -n 1 data/test.csv

# Update column names
python inference.py \
  data.text_column=your_text_column \
  data.image_column=your_image_column
```

#### CUDA Out of Memory

```bash
# Use CPU instead
python inference.py inference.checkpoint.device=cpu
```

---

### Deployment Issues

#### Docker Container Not Starting

```bash
# Check logs
docker logs fake-news-api

# Verify port availability
lsof -i :8000

# Use different port
docker run -d -p 8080:8000 fake-news-api:v1_onnx
```

#### Kubernetes Pod Not Running

```bash
# Check pod status
kubectl describe pod fake-news-deployment-xxx

# Check logs
kubectl logs fake-news-deployment-xxx

# Delete and recreate
kubectl delete -f k8s-onnx-deployment.yaml
kubectl apply -f k8s-onnx-deployment.yaml
```

---




## Advanced Usage

### Transfer Learning

```bash
# Freeze backbones for faster training
python src/training/train.py \
  model.freeze_vision=true \
  model.freeze_text=true
```

### Custom Model Architecture

Modify `src/models/multimodal_net.py` to experiment with:
- Different vision encoders (EfficientNet, ViT, ResNet)
- Different text encoders (BERT, RoBERTa, ALBERT)
- Alternative fusion strategies (attention, bilinear)

### Hyperparameter Tuning

Use MLflow with Optuna or Ray Tune for automated hyperparameter search.

### Model Export

**ONNX:**
```bash
python export_onnx.py
```

---

## Best Practices

### Development

- Use MLflow to track all experiments
- Save configurations with Hydra
- Test on small sample first before full dataset
- Use batch mode inference for large datasets
- Set appropriate threshold based on use case

### Deployment

- Use ONNX for production (smaller, faster)
- Deploy to Kubernetes for scalability
- Monitor API logs and performance
- Version all models with Git tags
- Use health checks in production

### Versioning

- Use DVC for datasets (data/images/)
- Use Git LFS for model checkpoints (*.pth)
- Tag important versions (git tag)
- Keep .dvc files in Git
- Document data changes in commit messages
- Never commit large files directly to Git

---

## References

### Papers

- **EfficientNet:** Tan & Le, 2019 - [Efficientnet: Rethinking model scaling for convolutional neural networks](https://arxiv.org/abs/1905.11946)
- **DistilBERT:** Sanh et al., 2019 - [DistilBERT, a distilled version of BERT](https://arxiv.org/abs/1910.01108)
- **Fakeddit Dataset:** Nakamura et al., 2020 - [Fakeddit: A New Multimodal Benchmark Dataset for Fine-grained Fake News Detection](https://arxiv.org/abs/1911.03854)

### Tools & Frameworks

- **PyTorch:** https://pytorch.org/
- **Transformers (Hugging Face):** https://huggingface.co/transformers/
- **MLflow:** https://mlflow.org/
- **Hydra:** https://hydra.cc/
- **FastAPI:** https://fastapi.tiangolo.com/
- **DVC:** https://dvc.org/- **ONNX Runtime:** https://onnxruntime.ai/

---

## Contact & Support

For questions or issues:
- Open a GitHub issue
- Check existing documentation in the repository
- Review MLflow experiment logs
**Author:** Rafael Bucio
**Project:** MLOps Zoomcamp - Capstone Project  
**Date:** December 2025


---

## Acknowledgments

This project was created for the MLOps Zoomcamp course by **DataTalks.Club**

---

**Last Updated:** 2025-12
**Version:** 1.7

---

**End of Documentation**
