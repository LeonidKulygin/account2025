# src/evaluation/metrics.py
import numpy as np
from sklearn.metrics import (
    roc_auc_score, 
    roc_curve,
    precision_recall_curve,
    f1_score,
    accuracy_score,
    gini
)
from typing import Dict


def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict:
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    predictions_binary = (predictions >= 0.5).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(targets, predictions_binary),
        'auc': roc_auc_score(targets, predictions),
        'gini': gini(targets, predictions),
        'f1': f1_score(targets, predictions_binary),
    }
    
    return metrics
