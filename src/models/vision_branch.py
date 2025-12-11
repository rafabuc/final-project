"""Vision branch using EfficientNet-B0."""

import torch
import torch.nn as nn
from torchvision import models


class VisionBranch(nn.Module):
    """
    Vision encoder using pretrained EfficientNet-B0.

    Extracts visual features from images and projects them
    to a fixed-dimensional embedding space.

    Args:
        embedding_dim: Dimension of output embeddings (default: 512)
        pretrained: Whether to use pretrained ImageNet weights (default: True)
        freeze_backbone: Whether to freeze EfficientNet layers (default: False)
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        super(VisionBranch, self).__init__()

        # Load pretrained EfficientNet-B0
        self.backbone = models.efficientnet_b0(pretrained=pretrained)

        # EfficientNet-B0 has 1280 features before the classifier
        in_features = self.backbone.classifier[1].in_features

        # Remove the original classifier
        self.backbone.classifier = nn.Identity()

        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Projection head to embedding space
        self.projection = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through vision branch.

        Args:
            images: Batch of images [batch_size, 3, H, W]

        Returns:
            Image embeddings [batch_size, embedding_dim]
        """
        # Extract features from EfficientNet
        features = self.backbone(images)  # [batch_size, 1280]

        # Project to embedding space
        embeddings = self.projection(features)  # [batch_size, embedding_dim]

        return embeddings
