"""
Metrics calculation module for toxicity classification evaluation.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import config


def calculate_metrics(results_df):
    """
    Calculate comprehensive classification metrics.

    Args:
        results_df: DataFrame with 'true_label' and 'predicted_label' columns

    Returns:
        Dictionary containing all metrics
    """
    # Filter valid predictions
    valid_df = results_df[results_df['predicted_label'] != -1].copy()

    if len(valid_df) == 0:
        print("No valid predictions to evaluate!")
        return None

    y_true = valid_df['true_label'].values
    y_pred = valid_df['predicted_label'].values

    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)

    # Per-class metrics
    precision_toxic = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    precision_nontoxic = precision_score(y_true, y_pred, pos_label=0, zero_division=0)

    recall_toxic = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall_nontoxic = recall_score(y_true, y_pred, pos_label=0, zero_division=0)

    f1_toxic = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_nontoxic = f1_score(y_true, y_pred, pos_label=0, zero_division=0)

    # Macro averages
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Classification report
    class_report = classification_report(
        y_true, y_pred,
        target_names=['non-toxic', 'toxic'],
        output_dict=True
    )

    # Count predictions
    total_samples = len(results_df)
    valid_samples = len(valid_df)
    invalid_samples = total_samples - valid_samples

    # Error counts
    false_positives = cm[0, 1] if cm.shape == (2, 2) else 0  # Non-toxic classified as toxic
    false_negatives = cm[1, 0] if cm.shape == (2, 2) else 0  # Toxic classified as non-toxic
    true_positives = cm[1, 1] if cm.shape == (2, 2) else 0
    true_negatives = cm[0, 0] if cm.shape == (2, 2) else 0

    metrics = {
        'total_samples': total_samples,
        'valid_samples': valid_samples,
        'invalid_samples': invalid_samples,
        'accuracy': accuracy,
        'precision_toxic': precision_toxic,
        'precision_nontoxic': precision_nontoxic,
        'precision_macro': precision_macro,
        'recall_toxic': recall_toxic,
        'recall_nontoxic': recall_nontoxic,
        'recall_macro': recall_macro,
        'f1_toxic': f1_toxic,
        'f1_nontoxic': f1_nontoxic,
        'f1_macro': f1_macro,
        'confusion_matrix': cm,
        'true_positives': true_positives,
        'true_negatives': true_negatives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'classification_report': class_report
    }

    return metrics


def print_metrics_report(metrics):
    """
    Print a formatted metrics report.
    """
    if metrics is None:
        return

    print("\n" + "="*60)
    print("CLASSIFICATION METRICS REPORT")
    print("="*60)

    print(f"\n{'='*60}")
    print("SAMPLE STATISTICS")
    print(f"{'='*60}")
    print(f"Total samples:       {metrics['total_samples']}")
    print(f"Valid predictions:   {metrics['valid_samples']} ({metrics['valid_samples']/metrics['total_samples']*100:.1f}%)")
    print(f"Invalid predictions: {metrics['invalid_samples']} ({metrics['invalid_samples']/metrics['total_samples']*100:.1f}%)")

    print(f"\n{'='*60}")
    print("OVERALL METRICS")
    print(f"{'='*60}")
    print(f"Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"F1 (Macro): {metrics['f1_macro']*100:.2f}%")

    print(f"\n{'='*60}")
    print("PER-CLASS METRICS")
    print(f"{'='*60}")
    print(f"\n{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 48)
    print(f"{'Non-toxic':<12} {metrics['precision_nontoxic']*100:>8.2f}%    {metrics['recall_nontoxic']*100:>8.2f}%    {metrics['f1_nontoxic']*100:>8.2f}%")
    print(f"{'Toxic':<12} {metrics['precision_toxic']*100:>8.2f}%    {metrics['recall_toxic']*100:>8.2f}%    {metrics['f1_toxic']*100:>8.2f}%")
    print("-" * 48)
    print(f"{'Macro Avg':<12} {metrics['precision_macro']*100:>8.2f}%    {metrics['recall_macro']*100:>8.2f}%    {metrics['f1_macro']*100:>8.2f}%")

    print(f"\n{'='*60}")
    print("CONFUSION MATRIX")
    print(f"{'='*60}")
    cm = metrics['confusion_matrix']
    print(f"\n{'':>15} {'Predicted':<20}")
    print(f"{'':>15} {'Non-toxic':<10} {'Toxic':<10}")
    print(f"{'Actual':<8} {'Non-toxic':<7} {cm[0,0]:>6}     {cm[0,1]:>6}")
    print(f"{'':>8} {'Toxic':<7} {cm[1,0]:>6}     {cm[1,1]:>6}")

    print(f"\n{'='*60}")
    print("ERROR ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"True Positives (Toxic correctly identified):     {metrics['true_positives']}")
    print(f"True Negatives (Non-toxic correctly identified): {metrics['true_negatives']}")
    print(f"False Positives (Non-toxic marked as toxic):     {metrics['false_positives']}")
    print(f"False Negatives (Toxic marked as non-toxic):     {metrics['false_negatives']}")


def save_metrics_to_csv(metrics, output_path=None):
    """
    Save metrics to a CSV file.
    """
    output_path = output_path or f"{config.RESULTS_DIR}/metrics.csv"

    metrics_flat = {k: v for k, v in metrics.items()
                   if not isinstance(v, (np.ndarray, dict))}

    df = pd.DataFrame([metrics_flat])
    df.to_csv(output_path, index=False)
    print(f"\nMetrics saved to {output_path}")


def calculate_and_report(results_df, save=True):
    """
    Calculate metrics and print report.

    Args:
        results_df: DataFrame with predictions
        save: Whether to save metrics to file

    Returns:
        Dictionary of metrics
    """
    metrics = calculate_metrics(results_df)
    print_metrics_report(metrics)

    if save and metrics:
        import os
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        save_metrics_to_csv(metrics)

    return metrics


if __name__ == "__main__":
    # Load results and calculate metrics
    try:
        results_df = pd.read_csv(f"{config.RESULTS_DIR}/predictions.csv")
        metrics = calculate_and_report(results_df)
    except FileNotFoundError:
        print("Results file not found. Please run inference.py first.")
