"""Inference script for single image-text pair."""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import DistilBertTokenizer

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.multimodal_net import MultimodalNet
from src.data.transforms import get_image_transforms


def load_model(checkpoint_path: str, device: str = 'cuda'):
    """Load trained model from checkpoint."""
    model = MultimodalNet(
        embedding_dim=512,
        fusion_hidden_dim=256,
        dropout_rate=0.4
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Validation accuracy: {checkpoint['val_accuracy']:.4f}")

    return model


def predict(
    model: torch.nn.Module,
    image_path: str,
    text: str,
    device: str = 'cuda',
    max_length: int = 128
):
    """
    Make prediction for a single image-text pair.

    Args:
        model: Trained MultimodalNet model
        image_path: Path to image file
        text: Text content
        device: Device to run inference on
        max_length: Maximum token length

    Returns:
        Prediction probability and label
    """
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    transform = get_image_transforms(mode='val')
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    # Tokenize text
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    # Prepare batch
    batch = {
        'image': image_tensor,
        'input_ids': encoded['input_ids'].to(device),
        'attention_mask': encoded['attention_mask'].to(device)
    }

    # Make prediction
    with torch.no_grad():
        output = model(batch)
        probability = output.item()
        label = 'FAKE' if probability >= 0.5 else 'REAL'

    return probability, label


def main():
    parser = argparse.ArgumentParser(description='Fake News Detection Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to image file')
    parser.add_argument('--text', type=str, required=True,
                       help='Text content')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use for inference')

    args = parser.parse_args()

    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, args.device)

    # Make prediction
    print(f"\nProcessing image: {args.image}")
    print(f"Text: {args.text[:100]}...")

    probability, label = predict(
        model,
        args.image,
        args.text,
        device=args.device
    )

    # Display results
    print("\n" + "="*50)
    print("PREDICTION RESULTS")
    print("="*50)
    print(f"Prediction: {label}")
    print(f"Confidence: {probability:.4f}")
    print(f"Fake probability: {probability:.2%}")
    print(f"Real probability: {(1 - probability):.2%}")
    print("="*50)


if __name__ == "__main__":
    main()
