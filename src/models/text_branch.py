"""Text branch using DistilBERT."""

import torch
import torch.nn as nn
from transformers import DistilBertModel


class TextBranch(nn.Module):
    """
    Text encoder using pretrained DistilBERT.

    Extracts semantic features from text and projects them
    to a fixed-dimensional embedding space.

    Args:
        embedding_dim: Dimension of output embeddings (default: 512)
        pretrained_model: HuggingFace model name (default: 'distilbert-base-uncased')
        freeze_backbone: Whether to freeze DistilBERT layers (default: False)
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        pretrained_model: str = 'distilbert-base-uncased',
        freeze_backbone: bool = False
    ):
        super(TextBranch, self).__init__()

        # Load pretrained DistilBERT
        self.backbone = DistilBertModel.from_pretrained(pretrained_model)

        # DistilBERT hidden size is 768
        bert_hidden_size = self.backbone.config.hidden_size

        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Projection head to embedding space
        self.projection = nn.Sequential(
            nn.Linear(bert_hidden_size, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through text branch.

        Args:
            input_ids: Tokenized input IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Text embeddings [batch_size, embedding_dim]
        """
        # Extract features from DistilBERT
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Use [CLS] token representation (first token)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [batch_size, 768]

        # Project to embedding space
        embeddings = self.projection(cls_embedding)  # [batch_size, embedding_dim]

        return embeddings
