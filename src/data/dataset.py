"""Custom PyTorch Dataset for Multimodal Fake News Detection."""

import os
from typing import Dict, Optional, Callable

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer


class MultimodalDataset(Dataset):
    """
    PyTorch Dataset for paired image-text fake news detection.

    Args:
        csv_file: Path to CSV file containing [image_path, text_content, label]
        root_dir: Root directory for images (if paths in CSV are relative)
        tokenizer: HuggingFace tokenizer for text processing
        max_length: Maximum sequence length for text tokenization
        image_transform: Torchvision transforms for image preprocessing
        text_column: Name of the text column in CSV (default: 'text_content')
        image_column: Name of the image path column in CSV (default: 'image_path')
        label_column: Name of the label column in CSV (default: 'label')
    """

    def __init__(
        self,
        csv_file: str,
        root_dir: Optional[str] = None,
        tokenizer: Optional[DistilBertTokenizer] = None,
        max_length: int = 128,
        image_transform: Optional[Callable] = None,
        text_column: str = "text_content",
        image_column: str = "image_path",
        label_column: str = "label",
        n_rows : int = -1
    ):

        if n_rows==-1:
            self.data_frame = pd.read_csv(csv_file)
        else:
            self.data_frame = pd.read_csv(csv_file, nrows=n_rows)
            
        self.root_dir = root_dir or ""
        self.max_length = max_length
        self.image_transform = image_transform
        self.text_column = text_column
        self.image_column = image_column
        self.label_column = label_column

        # Initialize tokenizer
        if tokenizer is None:
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        else:
            self.tokenizer = tokenizer

        # Validate required columns
        required_cols = [text_column, image_column, label_column]
        missing_cols = [col for col in required_cols if col not in self.data_frame.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.data_frame)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load and preprocess a single sample.

        Args:
            idx: Index of the sample

        Returns:
            Dictionary containing:
                - 'image': Preprocessed image tensor [C, H, W]
                - 'input_ids': Tokenized text input IDs [max_length]
                - 'attention_mask': Attention mask for text [max_length]
                - 'label': Binary label (0 or 1)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load image
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx][self.image_column])

        try:
            image = Image.open(img_name)#.convert('RGB')

             # ✅ FIX: Convert palette images with transparency to RGBA
            if image.mode == 'P' and 'transparency' in image.info:
                image = image.convert('RGBA')
            
            # ✅ Convert all images to RGB (handles RGBA, L, etc.)
            if image.mode != 'RGB':
                image = image.convert('RGB')

        except Exception as e:
            raise RuntimeError(f"Error loading image at {img_name}: {str(e)}")

        # Apply image transforms
        if self.image_transform:
            image = self.image_transform(image)
        else:
            # Default: convert to tensor
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        # Load and tokenize text
        text = str(self.data_frame.iloc[idx][self.text_column])

        # Handle missing or NaN text
        if pd.isna(text) or text.strip() == "":
            text = "[EMPTY]"  # Placeholder for empty text

        # Tokenize text
        encoded_text = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        # Load label
        label = torch.tensor(
            self.data_frame.iloc[idx][self.label_column],
            dtype=torch.float32
        )

        sample = {
            'image': image,
            'input_ids': encoded_text['input_ids'].squeeze(0),  # Remove batch dimension
            'attention_mask': encoded_text['attention_mask'].squeeze(0),
            'label': label
        }

        return sample

    def get_class_distribution(self) -> Dict[int, int]:
        """Return the distribution of classes in the dataset."""
        return self.data_frame[self.label_column].value_counts().to_dict()
