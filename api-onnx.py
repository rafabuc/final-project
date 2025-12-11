import os
import shutil
import logging
from typing import Dict, Any

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from contextlib import asynccontextmanager

# Import existing inference class
from inference_onnx import OnnxFakeNewsDetector


# Global dictionary to hold the model instance in memory
model_instance: Dict[str, Any] = {}

# --- Lifespan Context Manager ---
# This ensures the model is loaded ONLY ONCE when the app starts,
# preventing memory overhead on every request.
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting API and loading models...")
    
    try:

  
        model_path = "multimodal_model.onnx"

        # 3. Instantiate the Detector
        # This loads the weights (approx 500MB+) into RAM
        detector = OnnxFakeNewsDetector(model_path=model_path)
        
        # Store in the global dictionary
        model_instance["detector"] = detector
      
        
        print("Model loaded successfully. API is ready.")
        yield
        
    except Exception as e:
        print(f"Critical error loading the model: {e}")
        raise e
    finally:
        print("Shutting down API...")
        model_instance.clear()

# --- App Definition ---
app = FastAPI(title="Fake News Detector API with ONNX", lifespan=lifespan)

@app.get("/health")
def health_check():
    """Endpoint to check if the service is running and model is loaded."""
    if "detector" not in model_instance:
        raise HTTPException(status_code=503, detail="Model not initialized")
    return {"status": "ok", "device": str(model_instance["detector"].device)}

@app.post("/predict")
async def predict(
    text: str = Form(..., description="The text content of the news article"),
    image: UploadFile = File(..., description="The image associated with the news")
):
    """
    Analyzes an Image + Text pair to detect Fake News.
    """
    detector = model_instance.get("detector")
    if not detector:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # 1. Save uploaded image to a temporary file
    # The FakeNewsDetector expects a file path (str), not raw bytes.
    temp_filename = f"temp_{image.filename}"
    
    try:
        # Write bytes to disk
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
            
        # 2. Call the existing prediction method
        # root_dir="" is empty because we are providing a direct relative path
        result = detector.predict_single(
            image_path=temp_filename, 
            text=text, 
            root_dir=""
        )
        
        # 3. Return JSON response
        # Convert numpy/torch types to Python native types for JSON serialization
        return {
            "filename": image.filename,
            "prediction": result['prediction'],
            "probability_fake": float(result.get('probability', 0.0)),
            "confidence": float(result.get('confidence', 0.0)),
            "label_id": int(result['label'])
        }

    except Exception as e:
        #model_instance["logger"].error(f"API Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # 4. Cleanup: Delete the temporary file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

if __name__ == "__main__":
    import uvicorn
    # Run development server
    uvicorn.run(app, host="0.0.0.0", port=8000)