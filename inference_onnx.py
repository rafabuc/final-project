import os
import numpy as np
import onnxruntime as ort
from PIL import Image
from transformers import DistilBertTokenizer

class OnnxFakeNewsDetector:
    def __init__(self, model_path, log_enabled=True):
        """
        Detector optimized using ONNX Runtime.
        No requires PyTorch.
        """
        self.log_enabled = log_enabled
        if self.log_enabled:
            print(f"Loading ONNX model from: {model_path}")
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        # 1. Initialize ONNX session (Inference engine)
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        
        # 2. Initialize Tokenizer (HuggingFace)
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.max_length = 128

    def preprocess_image(self, image_path):
        """
        Replicates torchvision.transforms using NumPy.
        Resize -> Normalize (ImageNet) -> Transpose
        """
        # Open image
        img = Image.open(image_path).convert('RGB')
        
        # Resize (224, 224) - Uses BILINEAR to match PyTorch
        img = img.resize((224, 224), Image.Resampling.BILINEAR)
        
        # Convert to float32 array [0-255]
        img_data = np.array(img).astype(np.float32)
        
        # Normalize to [0-1]
        img_data = img_data / 255.0
        
        # Normalize with ImageNet means and std
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        img_data = (img_data - mean) / std
        
        # Transpose from [H, W, C] to [C, H, W]
        img_data = img_data.transpose(2, 0, 1)
        
        # Add batch dimension [1, C, H, W]
        img_data = np.expand_dims(img_data, axis=0)
        
        return img_data

    def preprocess_text(self, text):
        """Tokenizes the text and returns numpy arrays."""
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )
        
        # Convert list to numpy arrays with batch dimension [1, 128]
        input_ids = np.array(encoding['input_ids'], dtype=np.int64).reshape(1, self.max_length)
        attention_mask = np.array(encoding['attention_mask'], dtype=np.int64).reshape(1, self.max_length)
        
        return input_ids, attention_mask

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict_single(self, image_path, text, root_dir=""):
        """
        Performs the prediction.
        Maintains the method signature for compatibility with api.py.
        """
        # Handle paths
        if root_dir and not os.path.isabs(image_path):
            full_path = os.path.join(root_dir, image_path)
        else:
            full_path = image_path

        # A. Preprocess
        img_input = self.preprocess_image(full_path)
        input_ids, attention_mask = self.preprocess_text(text)

        # B. ONNX Inference
        inputs_onnx = {
            'image': img_input,
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        
        # Run! (returns a list, we take the first element which are the logits)
        logits = self.session.run(None, inputs_onnx)[0] 
        
        # C. Post-processing
        prob = self.sigmoid(logits)[0][0] # Scalar
        
        prediction = "FAKE" if prob > 0.5 else "REAL"
        confidence = prob if prob > 0.5 else 1 - prob
        
        return {
            'prediction': prediction,
            'label': 1 if prediction == "FAKE" else 0,
            'probability': float(prob),
            'confidence': float(confidence)
        }