"""Multimodal Network for Fake News Detection with Late Fusion."""

import torch
import torch.nn as nn

from .vision_branch import VisionBranch
from .text_branch import TextBranch


class MultimodalNet(nn.Module):
    """
    Multimodal fake news detection network using late fusion.

    Architecture:
        1. Vision Branch: EfficientNet-B0 -> Projection -> Image Embeddings
        2. Text Branch: DistilBERT -> Projection -> Text Embeddings
        3. Fusion: Concatenate embeddings -> Dropout -> Dense -> Sigmoid

    Args:
        embedding_dim: Dimension of embeddings from each branch (default: 512)
        fusion_hidden_dim: Hidden dimension in fusion layer (default: 256)
        dropout_rate: Dropout probability in fusion layer (default: 0.4)
        freeze_vision: Whether to freeze vision backbone (default: False)
        freeze_text: Whether to freeze text backbone (default: False)
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        fusion_hidden_dim: int = 256,
        dropout_rate: float = 0.4,
        freeze_vision: bool = False,
        freeze_text: bool = False
    ):
        super(MultimodalNet, self).__init__()

        self.embedding_dim = embedding_dim

        # Initialize vision and text branches
        self.vision_branch = VisionBranch(
            embedding_dim=embedding_dim,
            pretrained=True,
            freeze_backbone=freeze_vision
        )

        self.text_branch = TextBranch(
            embedding_dim=embedding_dim,
            pretrained_model='distilbert-base-uncased',
            freeze_backbone=freeze_text
        )

        # Late fusion layer
        # Concatenated embeddings: [batch_size, 2 * embedding_dim]
        self.fusion = nn.Sequential(
            nn.Linear(2 * embedding_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(128, 1)#,
            #nn.Sigmoid()  BCEWithLogitsLoss applies sigmoide internally
        )

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Forward pass through the multimodal network.

        Args:
            batch: Dictionary containing:
                - 'image': Image tensor [batch_size, 3, H, W]
                - 'input_ids': Tokenized text [batch_size, seq_len]
                - 'attention_mask': Attention mask [batch_size, seq_len]

        Returns:
            Predictions: Sigmoid probabilities [batch_size, 1]
        """
        # Extract image embeddings
        image_embeddings = self.vision_branch(batch['image'])  # [batch_size, embedding_dim]

        # Extract text embeddings
        text_embeddings = self.text_branch(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )  # [batch_size, embedding_dim]

        # Concatenate embeddings (late fusion)
        fused_embeddings = torch.cat(
            [image_embeddings, text_embeddings],
            dim=1
        )  # [batch_size, 2 * embedding_dim]

        # Pass through fusion layer
        output = self.fusion(fused_embeddings)  # [batch_size, 1]

        return output

    def get_embeddings(self, batch: dict) -> dict:
        """
        Extract separate embeddings without fusion (useful for analysis).

        Args:
            batch: Dictionary containing image and text data

        Returns:
            Dictionary with image_embeddings, text_embeddings, and fused_embeddings
        """
        with torch.no_grad():
            image_embeddings = self.vision_branch(batch['image'])
            text_embeddings = self.text_branch(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            fused_embeddings = torch.cat([image_embeddings, text_embeddings], dim=1)

        return {
            'image_embeddings': image_embeddings,
            'text_embeddings': text_embeddings,
            'fused_embeddings': fused_embeddings
        }

    def count_parameters(self) -> dict:
        """Count trainable and total parameters in the model."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': total_params - trainable_params
        }
