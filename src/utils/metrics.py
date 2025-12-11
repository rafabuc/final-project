"""Metrics calculation utilities."""

from typing import Dict

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


def calculate_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics.

    Args:
        predictions: Model predictions (probabilities) [batch_size, 1] or [batch_size]
        labels: Ground truth labels [batch_size]
        threshold: Classification threshold (default: 0.5)

    Returns:
        Dictionary containing accuracy, precision, recall, f1, and AUC
    """
    # Convert to numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    # Flatten arrays
    predictions = predictions.flatten()
    labels = labels.flatten()

    # Apply threshold for binary predictions
    binary_predictions = (predictions >= threshold).astype(int)

    # Calculate metrics
    metrics = {
        'accuracy': float(accuracy_score(labels, binary_predictions)),
        'precision': float(precision_score(labels, binary_predictions, zero_division=0)),
        'recall': float(recall_score(labels, binary_predictions, zero_division=0)),
        'f1': float(f1_score(labels, binary_predictions, zero_division=0)),
    }

    # Add AUC if we have both classes
    try:
        metrics['auc'] = float(roc_auc_score(labels, predictions))
    except ValueError:
        metrics['auc'] = 0.0

    return metrics


def get_confusion_matrix(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Calculate confusion matrix.

    Args:
        predictions: Model predictions (probabilities)
        labels: Ground truth labels
        threshold: Classification threshold

    Returns:
        Confusion matrix as numpy array
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    predictions = predictions.flatten()
    labels = labels.flatten()
    binary_predictions = (predictions >= threshold).astype(int)

    return confusion_matrix(labels, binary_predictions)
