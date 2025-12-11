# Inference Instructions - Complete Guide

This document provides a comprehensive guide for running inference, deploying the API, and containerizing the Multimodal Fake News Detection model.

---

## Table of Contents
1. [Running Inference Script](#1-running-inference-script)
2. [API Deployment and Testing](#2-api-deployment-and-testing)
3. [Docker Deployment](#3-docker-deployment)
4. [ONNX Model Export](#4-onnx-model-export)
5. [Kubernetes Deployment](#5-kubernetes-deployment)

---

## 1. Running Inference Script


The inference script (`inference.py`) uses Hydra for configuration management, matching the training pipeline. This provides:
- **Hierarchical configuration**: Organized config files for different components
- **Command-line overrides**: Easy parameter changes without editing files
- **Config composition**: Mix and match different configurations
- **Reproducibility**: All configurations are logged automatically

### Configuration Structure

```
configs/
├── config_inference.yaml      # Main inference config
├── inference/
│   └── default.yaml          # Inference-specific settings
├── model/
│   └── multimodal.yaml       # Model architecture
└── data/
    └── fakeddit.yaml         # Data settings
```

### 1.1 Single Mode Inference


**Command:**
```bash
python inference.py
```

**What it does:**
- Runs inference on a single image-text pair
- Loads the trained multimodal model from checkpoint
- Processes one sample at a time as configured in the config file
- Returns prediction (FAKE/REAL), probability, and confidence score

**Configuration:**
- Mode: `single`
- Requires: checkpoint path, single image path, and text content
- Model: Uses EfficientNet-B0 (vision) + DistilBERT (text)
- Device: Auto-detects CUDA/CPU

**Example Output:**
```
Prediction: REAL
Probability (Fake): 0.1527
Confidence: 0.8473
```

**Example Use Cases:**
- Testing individual news samples
- Quick validation of model performance
- Interactive testing with specific image-text combinations

---

### 1.2 Batch Mode Inference

**Command:**
```bash
python inference.py
```
(with batch mode configured in config file)

**What it does:**
- Processes multiple samples from a CSV file
- Generates predictions for entire datasets
- Saves results to output CSV file
- More efficient for large-scale testing

**Configuration:**
- Mode: `batch`
- Requires: CSV file with image paths and text
- Outputs: predictions.csv with all results
- Batch size: 32 (configurable)

---

## 2. API Deployment and Testing

### 2.1 Install API Dependencies

**Command:**
```bash
pip install fastapi uvicorn python-multipart
```

**What it does:**
- **fastapi**: Installs the FastAPI framework for building the REST API
- **uvicorn**: Installs the ASGI server to run the API
- **python-multipart**: Enables handling of multipart form data (required for file uploads)

---

### 2.2 Run the API Server

**Command:**
```bash
uvicorn api:app --reload
```

**What it does:**
- Starts the FastAPI application server
- `api:app` - references the app object in api.py
- `--reload` - enables auto-restart when code changes (development mode)
- Loads the trained model on startup
- Listens on http://localhost:8000 by default

**Startup Process:**
1. Loads model from checkpoint
2. Initializes tokenizer (DistilBERT)
3. Sets up prediction endpoint
4. Displays "Model loaded successfully" when ready

---

### 2.3 Test via Swagger UI (Interactive Documentation)

**Steps:**
1. Open browser and navigate to: `http://localhost:8000/docs`
2. Click on POST `/predict` endpoint
3. Click "Try it out" button
4. Fill in:
   - **Text field**: Enter news headline/content
   - **Image field**: Upload test image (.jpg or .png)
5. Click "Execute"
6. View JSON response with prediction results

**What it does:**
- Provides interactive API documentation
- Allows testing without writing code
- Displays request/response schemas
- Shows example values and formats

---

### 2.4 Test via cURL (Command Line)

**Command:**
```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'text=This is a breaking news test' \
  -F 'image=@/path/to/your/image.jpg'
```

**What it does:**
- Sends POST request to /predict endpoint
- `-F 'text=...'` - sends text content as form field
- `-F 'image=@...'` - uploads image file
- Returns JSON with prediction, probability, confidence, and label

**Example Response:**
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

### 2.5 Run API Directly (Development)

**Command:**
```bash
python api.py
```

**What it does:**
- Starts the API server directly (alternative to uvicorn command)
- Loads model and tokenizer on startup
- Useful for debugging
- May show more detailed startup logs

**Note:** Port 8000 must be available (error if already in use)

---

## 3. Docker Deployment

### 3.1 Build Docker Image (PyTorch Version)

**Command:**
```bash
docker build -t fake-news-api:v1_torch -f Dockerfile-torch .
```

**What it does:**
- Builds Docker container with full PyTorch dependencies
- `-t fake-news-api:v1_torch` - tags image with name and version
- `-f Dockerfile-torch` - specifies Dockerfile to use
- `.` - uses current directory as build context
- Image size: ~12.1GB (includes full PyTorch libraries)

**Contains:**
- Python environment
- PyTorch and torchvision
- All model dependencies
- Trained model checkpoint
- API code

---

### 3.2 Run Docker Container

**Command:**
```bash
docker run -d -p 8000:8000 --name fake-news-api fake-news-api:v1_torch
```

**What it does:**
- `-d` - runs container in detached mode (background)
- `-p 8000:8000` - maps container port 8000 to host port 8000
- `--name fake-news-api` - assigns container name
- Downloads EfficientNet weights on first run (~20.5MB)

**Container startup:**
1. Initializes Python environment
2. Loads model checkpoint
3. Downloads pretrained weights if needed
4. Starts uvicorn server
5. Ready to accept requests

---

### 3.3 Monitor Docker Logs

**Commands:**
```bash
# View all logs
docker logs fdbd5f6f049e

# Follow logs in real-time
docker logs -f fdbd5f6f049e

# Check running containers
docker ps
```

**What it does:**
- `docker logs` - displays container output and errors
- `-f` flag - follows log output (like tail -f)
- `docker ps` - lists running containers with status

**Useful for:**
- Debugging startup issues
- Monitoring API requests
- Checking model loading status

---

## 4. ONNX Model Export

### 4.1 Export PyTorch Model to ONNX

**Command:**
```bash
python export_onnx.py
```

**What it does:**
- Converts PyTorch model to ONNX format
- Loads checkpoint from `src/training/checkpoints/best_model_1.7.pth`
- Exports to `multimodal_model.onnx`
- Optimizes model graph with 126 pattern rewrite rules

**Benefits of ONNX:**
- **Smaller size**: Reduced inference runtime dependencies
- **Faster inference**: Optimized execution graph
- **Cross-platform**: Works with ONNX Runtime on multiple devices
- **No PyTorch needed**: Lighter deployment

**Output:**
```
✅ Success! Model exported to: multimodal_model.onnx
```

---

### 4.2 Build Docker Image (ONNX Version)

**Command:**
```bash
docker build -t fake-news-api:v1_onnx -f Dockerfile-onnx .
```

**What it does:**
- Builds lightweight container with ONNX Runtime
- Image size: ~754MB (much smaller than PyTorch version)
- Uses optimized ONNX model for inference
- No PyTorch dependencies needed

**Advantages:**
- 94% smaller than PyTorch version (754MB vs 12.1GB)
- Faster startup time
- Lower memory footprint
- Better for production deployment

---

### 4.3 Run ONNX Container

**Command:**
```bash
docker run -d -p 8000:8000 --name fake-news-api fake-news-api:v1_onnx
```

**What it does:**
- Same as PyTorch container but uses ONNX Runtime
- Loads `multimodal_model.onnx` instead of .pth checkpoint
- Provides identical API interface
- Better performance for CPU inference

---

## 5. Kubernetes Deployment

### 5.1 Check Minikube Status

**Commands:**
```bash
minikube status
minikube start
```

**What it does:**
- `minikube status` - checks if local Kubernetes cluster is running
- `minikube start` - starts local Kubernetes cluster
- Creates single-node cluster for testing

---

### 5.2 Load Docker Image into Minikube

**Command:**
```bash
minikube image load fake-news-api:v1_onnx
```

**What it does:**
- Transfers Docker image from local registry to Minikube's Docker daemon
- Required because Minikube uses separate Docker environment
- Verifies image is available in cluster

**Verify:**
```bash
minikube ssh -- docker images
```

---

### 5.3 Deploy to Kubernetes

**Command:**
```bash
kubectl apply -f k8s-onnx-deployment.yaml
```

**What it does:**
- Creates Deployment (manages pods running the API)
- Creates Service (exposes the API)
- Deploys 1 replica of the fake-news-api pod
- Sets up NodePort service on port 31018

**Resources created:**
- **Pod**: fake-news-deployment-6479788fb4-n2kpb
- **Service**: fake-news-service (NodePort)
- **Deployment**: fake-news-deployment

---

### 5.4 Check Kubernetes Resources

**Command:**
```bash
kubectl get all
```

**What it does:**
- Lists all Kubernetes resources (pods, services, deployments)
- Shows status (Running, Ready, Available)
- Displays ports and endpoints

**Example output:**
```
NAME                                        READY   STATUS    RESTARTS   AGE
pod/fake-news-deployment-6479788fb4-n2kpb   1/1     Running   0          81s

NAME                        TYPE        CLUSTER-IP       EXTERNAL-IP   PORT(S)
service/fake-news-service   NodePort    10.110.221.161   <none>        80:31018/TCP
```

---

### 5.5 Access the Service

**Option 1: Port Forwarding**
```bash
kubectl port-forward service/fake-news-service 8000:80
```

**What it does:**
- Forwards local port 8000 to service port 80
- Allows accessing API at http://localhost:8000
- Keeps connection open (Ctrl+C to stop)

---

**Option 2: Minikube Service URL**
```bash
minikube service fake-news-service --url
```

**What it does:**
- Returns the NodePort URL: `http://192.168.49.2:31018`
- Access API directly via Minikube IP

---

**Option 3: Minikube Tunnel**
```bash
minikube tunnel
```

**What it does:**
- Creates network tunnel for LoadBalancer services
- Provides external IP access
- Requires sudo/admin privileges

---

### 5.6 Test Kubernetes Deployment

**Command:**
```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'text=This frog sitting on a light' \
  -F 'image=@/mnt/d/workspace/.../c7jxj5.jpg'
```

**Response:**
```json
{
  "filename": "c7jxj5.jpg",
  "prediction": "FAKE",
  "probability_fake": 0.6504932641983032,
  "confidence": 0.6504932641983032,
  "label_id": 1
}
```

**What it does:**
- Tests the deployed API in Kubernetes
- Verifies model inference is working
- Confirms networking and port forwarding

---

## Summary of Deployment Options

| Method | Use Case | Size | Speed | Complexity |
|--------|----------|------|-------|------------|
| **Local Script** | Development, testing | N/A | Fast | Low |
| **API (Local)** | Development, testing | N/A | Fast | Low |
| **Docker (PyTorch)** | Full features, debugging | 12.1GB | Medium | Medium |
| **Docker (ONNX)** | Production, efficiency | 754MB | Fast | Medium |
| **Kubernetes** | Scalability, production | 754MB | Fast | High |

---

## Recommended Workflow

1. **Development**: Use `python inference.py` for quick testing
2. **API Testing**: Run `uvicorn api:app --reload` locally
3. **Containerization**: Build ONNX Docker image for deployment
4. **Production**: Deploy to Kubernetes for scalability

---

## Prerequisites Summary

- Python 3.11
- PyTorch, torchvision
- FastAPI, uvicorn
- Docker (for containerization)
- Kubernetes/Minikube (for cluster deployment)
- Trained model checkpoint
- ONNX Runtime (for optimized deployment)
