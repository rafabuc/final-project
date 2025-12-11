"""Inference script for trained Multimodal Fake News Detection model using Hydra configuration."""

import os
import sys
from pathlib import Path
from typing import Dict, Union, Optional

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import pandas as pd
from PIL import Image
from transformers import DistilBertTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from src.models.multimodal_net import MultimodalNet
from src.data.transforms import get_image_transforms
from src.data.dataset import MultimodalDataset
from src.utils.logging import setup_logger


class FakeNewsDetector:
    """
    Inference wrapper for Multimodal Fake News Detection.

    Args:
        cfg: Hydra configuration object
        logger: Logger instance
    """

    def __init__(self, cfg: DictConfig, logger):
        self.cfg = cfg
        self.logger = logger

        # Set device
        if cfg.inference.checkpoint.device:
            self.device = torch.device(cfg.inference.checkpoint.device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.logger.info(f"Using device: {self.device}")

        # Load checkpoint
        # Load checkpoint from src/training/checkpoints/best_model_1.7.pth       
        checkpoint_path = cfg.inference.checkpoint.path
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        self.logger.info(f"Loading model from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.checkpoint_config = checkpoint.get('config', {})

        # Initialize model with config from checkpoint or cfg
        self.model = MultimodalNet(
            embedding_dim=self.checkpoint_config.get('embedding_dim', cfg.model.embedding_dim),
            fusion_hidden_dim=self.checkpoint_config.get('fusion_hidden_dim', cfg.model.fusion_hidden_dim),
            dropout_rate=cfg.inference.dropout_rate,#cfg.model.dropout_rate,
            freeze_vision=cfg.inference.freeze_vision,#cfg.model.freeze_vision,
            freeze_text=cfg.inference.freeze_text,#cfg.model.freeze_text
        ).to(self.device)

        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        epoch = checkpoint.get('epoch', 'N/A')
        val_acc = checkpoint.get('val_accuracy', 0.0)
        self.logger.info(f"Model loaded successfully! (Epoch: {epoch}, Val Acc: {val_acc:.4f})")

        # Initialize tokenizer
        tokenizer_name = cfg.inference.text.tokenizer
        self.logger.info(f"Loading tokenizer: {tokenizer_name}")
        self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_name)
        self.max_length = cfg.inference.text.max_length

        # Initialize image transforms
        self.image_transform = get_image_transforms(
            mode=cfg.inference.image.mode,
            image_size=cfg.inference.image.size
        )

    def preprocess_image(self, image_path: str, root_dir: str = "") -> torch.Tensor:
        """Preprocess image for model input."""
        try:
            # Handle relative and absolute paths
            if root_dir and not os.path.isabs(image_path):
                full_path = os.path.join(root_dir, image_path)
            else:
                full_path = image_path

            image = Image.open(full_path).convert('RGB')
            return self.image_transform(image)
        except Exception as e:
            raise ValueError(f"Error loading image {image_path}: {e}")

    def preprocess_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Preprocess text for model input."""
        encoding = self.tokenizer(#encode_plus
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }

    @torch.no_grad()
    def predict_single(self, image_path: str, text: str, root_dir: str = "") -> Dict[str, Union[float, str]]:
        """
        Make prediction for a single image-text pair.

        Args:
            image_path: Path to image file
            text: News article text
            root_dir: Root directory for images (if path is relative)

        Returns:
            Dictionary with prediction results
        """
        # Preprocess inputs
        image = self.preprocess_image(image_path, root_dir).unsqueeze(0).to(self.device)
        text_encoding = self.preprocess_text(text)

        # Create batch
        batch = {
            'image': image,
            'input_ids': text_encoding['input_ids'].unsqueeze(0).to(self.device),
            'attention_mask': text_encoding['attention_mask'].unsqueeze(0).to(self.device)
        }

        # Forward pass
        output = self.model(batch)
        probability = torch.sigmoid(output).item()  # Apply sigmoid if using BCEWithLogitsLoss

        # Determine prediction using threshold from config
        threshold = self.cfg.inference.output.threshold
        prediction = "FAKE" if probability > threshold else "REAL"
        confidence = probability if probability > threshold else (1 - probability)

        result = {
            'prediction': prediction,
            'label': 1 if prediction == "FAKE" else 0
        }

        if self.cfg.inference.output.save_probabilities:
            result['probability'] = probability

        if self.cfg.inference.output.save_confidence:
            result['confidence'] = confidence

        return result

    def predict_batch(self, csv_file: str, output_file: str, root_dir: str = "", n_rows: int = 100) -> pd.DataFrame:
        """
        Make predictions for a batch of samples from CSV.

        Args:
            csv_file: Path to CSV with required columns
            output_file: Path to save predictions
            root_dir: Root directory for images

        Returns:
            DataFrame with predictions
        """
        self.logger.info(f"Loading data from {csv_file}...")
        df = pd.read_csv(csv_file, nrows=n_rows)

        # Check required columns (using column names from data config)
        image_col = self.cfg.data.image_column
        text_col = self.cfg.data.text_column

        if image_col not in df.columns or text_col not in df.columns:
            raise ValueError(f"CSV must contain '{image_col}' and '{text_col}' columns")

        predictions = []
        self.logger.info(f"Processing {len(df)} samples...")

        # Process with progress bar
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Predicting"):
            try:
                result = self.predict_single(row[image_col], row[text_col], root_dir)
                predictions.append(result)

            except Exception as e:
                self.logger.error(f"Error processing row {idx}: {e}")
                predictions.append({
                    'prediction': 'ERROR',
                    'probability': -1.0,
                    'confidence': -1.0,
                    'label': -1
                })

        # Create results DataFrame
        results_df = pd.DataFrame(predictions)
        results_df = pd.concat([df, results_df], axis=1)

        # Save to file
        results_df.to_csv(output_file, index=False)
        self.logger.info(f"Predictions saved to {output_file}")

        # Print summary statistics
        self.logger.info("\n=== Prediction Summary ===")
        self.logger.info(f"Total samples: {len(results_df)}")
        self.logger.info(f"Predicted FAKE: {(results_df['prediction'] == 'FAKE').sum()}")
        self.logger.info(f"Predicted REAL: {(results_df['prediction'] == 'REAL').sum()}")

        if self.cfg.inference.output.save_confidence:
            valid_conf = results_df[results_df['confidence'] >= 0]['confidence']
            if len(valid_conf) > 0:
                self.logger.info(f"Average confidence: {valid_conf.mean():.4f}")

        return results_df


@hydra.main(version_base=None, config_path="configs", config_name="config_inference")
def main(cfg: DictConfig) -> None:
    """
    Main inference function with Hydra configuration.

    Args:
        cfg: Hydra configuration object
    """
    # Setup logger
    logger = setup_logger('InferencePipeline', log_file='inference.log')
    logger.info("Starting Multimodal Fake News Detection Inference")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Initialize detector
    detector = FakeNewsDetector(cfg, logger)

    # Determine mode
    mode = cfg.inference.mode

    if mode == "single":
        # Single sample prediction
        image_path = cfg.inference.single.image_path
        text = cfg.inference.single.text

        if not image_path or not text:
            raise ValueError(
                "For single mode, both 'inference.single.image_path' and "
                "'inference.single.text' must be provided in config or via command line.\n"
                "Example: python inference.py inference.single.image_path=path/to/image.jpg "
                "inference.single.text='Your news text here'"
            )

        logger.info("\n=== Single Sample Prediction ===")
        logger.info(f"Image: {image_path}")
        logger.info(f"Text: {text[:100]}..." if len(text) > 100 else f"Text: {text}")

        result = detector.predict_single(image_path, text)

        logger.info("\n=== Results ===")
        logger.info(f"Prediction: {result['prediction']}")
        if cfg.inference.output.save_probabilities:
            logger.info(f"Probability (Fake): {result['probability']:.4f}")
        if cfg.inference.output.save_confidence:
            logger.info(f"Confidence: {result['confidence']:.4f}")

    elif mode == "batch":
        # Batch prediction
        csv_file = cfg.inference.batch.csv_file
        output_file = cfg.inference.batch.output_file
        root_dir = cfg.inference.batch.get('root_dir', '')
        n_rows = cfg.inference.batch.get('n_rows', 100)

        if not csv_file:
            raise ValueError(
                "For batch mode, 'inference.batch.csv_file' must be provided.\n"
                "Example: python inference.py inference.mode=batch "
                "inference.batch.csv_file=data/test.csv"
            )

        logger.info("\n=== Batch Prediction ===")
        results_df = detector.predict_batch(csv_file, output_file, root_dir=root_dir, n_rows=n_rows)

        # Display sample predictions if verbose
        if cfg.inference.output.verbose:
            logger.info("\n=== Sample Predictions ===")
            display_cols = ['prediction']
            if cfg.inference.output.save_probabilities:
                display_cols.append('probability')
            if cfg.inference.output.save_confidence:
                display_cols.append('confidence')

            print(results_df[display_cols].head(10))

    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'single' or 'batch'")

    logger.info("\nInference completed successfully!")


if __name__ == "__main__":
    main()
