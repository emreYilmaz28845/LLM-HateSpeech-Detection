# visualization.py - makes all the charts and graphs

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import config
from metrics import calculate_metrics


def setup_style():
    """sets up the plot style so everything looks consistent"""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12


def plot_confusion_matrix(metrics, save_path=None):
    """
    makes a confusion matrix heatmap
    """
    cm = metrics['confusion_matrix']

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Non-toxic', 'Toxic'],
        yticklabels=['Non-toxic', 'Toxic'],
        ax=ax,
        annot_kws={'size': 16}
    )

    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix - Toxicity Classification', fontsize=14, fontweight='bold')

    # add percentages to each cell
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            percentage = cm[i, j] / total * 100
            current_text = ax.texts[i * 2 + j]
            current_text.set_text(f'{cm[i, j]}\n({percentage:.1f}%)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")

    return fig


def plot_metrics_comparison(metrics, save_path=None):
    """
    bar chart comparing precision, recall, f1 for both classes
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(3)  # precision, recall, f1
    width = 0.35

    nontoxic_values = [
        metrics['precision_nontoxic'] * 100,
        metrics['recall_nontoxic'] * 100,
        metrics['f1_nontoxic'] * 100
    ]

    toxic_values = [
        metrics['precision_toxic'] * 100,
        metrics['recall_toxic'] * 100,
        metrics['f1_toxic'] * 100
    ]

    bars1 = ax.bar(x - width/2, nontoxic_values, width, label='Non-toxic', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x + width/2, toxic_values, width, label='Toxic', color='#e74c3c', alpha=0.8)

    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Classification Metrics by Class', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Precision', 'Recall', 'F1-Score'])
    ax.legend()
    ax.set_ylim(0, 105)

    # put the values on top of bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)

    add_labels(bars1)
    add_labels(bars2)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics comparison saved to {save_path}")

    return fig


def plot_prediction_distribution(results_df, save_path=None):
    """
    shows how many predictions were correct vs incorrect
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # pie chart for correct vs incorrect
    valid_df = results_df[results_df['predicted_label'] != -1]
    correct_count = valid_df['correct'].sum()
    incorrect_count = len(valid_df) - correct_count
    invalid_count = len(results_df) - len(valid_df)

    colors = ['#2ecc71', '#e74c3c', '#95a5a6']
    sizes = [correct_count, incorrect_count, invalid_count]
    labels = [f'Correct\n({correct_count})', f'Incorrect\n({incorrect_count})', f'Invalid\n({invalid_count})']

    # dont show zero values
    non_zero = [(s, l, c) for s, l, c in zip(sizes, labels, colors) if s > 0]
    if non_zero:
        sizes, labels, colors = zip(*non_zero)

    axes[0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                startangle=90, explode=[0.05] * len(sizes))
    axes[0].set_title('Prediction Accuracy Distribution', fontsize=14, fontweight='bold')

    # bar chart breakdown
    categories = ['True\nPositive', 'True\nNegative', 'False\nPositive', 'False\nNegative']

    # count up each type
    tp = ((valid_df['true_label'] == 1) & (valid_df['predicted_label'] == 1)).sum()
    tn = ((valid_df['true_label'] == 0) & (valid_df['predicted_label'] == 0)).sum()
    fp = ((valid_df['true_label'] == 0) & (valid_df['predicted_label'] == 1)).sum()
    fn = ((valid_df['true_label'] == 1) & (valid_df['predicted_label'] == 0)).sum()

    values = [tp, tn, fp, fn]
    colors = ['#27ae60', '#3498db', '#e67e22', '#c0392b']

    bars = axes[1].bar(categories, values, color=colors, alpha=0.8)
    axes[1].set_xlabel('Category', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('Prediction Breakdown', fontsize=14, fontweight='bold')

    # add counts on top
    for bar, val in zip(bars, values):
        axes[1].annotate(str(val),
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prediction distribution saved to {save_path}")

    return fig


def plot_class_distribution(results_df, save_path=None):
    """
    compares true labels vs predicted labels
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    valid_df = results_df[results_df['predicted_label'] != -1]

    # true label distribution
    true_counts = valid_df['true_label'].value_counts().sort_index()
    axes[0].bar(['Non-toxic', 'Toxic'], [true_counts.get(0, 0), true_counts.get(1, 0)],
                color=['#2ecc71', '#e74c3c'], alpha=0.8)
    axes[0].set_title('True Label Distribution', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Count', fontsize=12)

    # predicted label distribution
    pred_counts = valid_df['predicted_label'].value_counts().sort_index()
    axes[1].bar(['Non-toxic', 'Toxic'], [pred_counts.get(0, 0), pred_counts.get(1, 0)],
                color=['#2ecc71', '#e74c3c'], alpha=0.8)
    axes[1].set_title('Predicted Label Distribution', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Count', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class distribution saved to {save_path}")

    return fig


def plot_accuracy_summary(metrics, save_path=None):
    """
    makes a nice summary of the key metrics
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics_names = ['Accuracy', 'Precision\n(Macro)', 'Recall\n(Macro)', 'F1-Score\n(Macro)']
    values = [
        metrics['accuracy'] * 100,
        metrics['precision_macro'] * 100,
        metrics['recall_macro'] * 100,
        metrics['f1_macro'] * 100
    ]

    colors = plt.cm.viridis(np.linspace(0.3, 0.8, len(metrics_names)))
    bars = ax.bar(metrics_names, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)

    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Overall Model Performance', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 105)

    # add value labels
    for bar, val in zip(bars, values):
        ax.annotate(f'{val:.1f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=12, fontweight='bold')

    # add threshold lines for reference
    ax.axhline(y=70, color='orange', linestyle='--', alpha=0.5, label='70% threshold')
    ax.axhline(y=80, color='green', linestyle='--', alpha=0.5, label='80% threshold')
    ax.legend(loc='lower right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Accuracy summary saved to {save_path}")

    return fig


def generate_all_visualizations(results_df, metrics=None):
    """
    generates and saves all the visualizations
    """
    setup_style()

    # make sure output folder exists
    os.makedirs(config.FIGURES_DIR, exist_ok=True)

    # calculate metrics if we dont have them
    if metrics is None:
        metrics = calculate_metrics(results_df)

    if metrics is None:
        print("Cannot generate visualizations: no valid metrics")
        return

    print("\n" + "="*50)
    print("GENERATING VISUALIZATIONS")
    print("="*50)

    # make all the plots
    plot_confusion_matrix(metrics, f"{config.FIGURES_DIR}/confusion_matrix.png")
    plot_metrics_comparison(metrics, f"{config.FIGURES_DIR}/metrics_comparison.png")
    plot_prediction_distribution(results_df, f"{config.FIGURES_DIR}/prediction_distribution.png")
    plot_class_distribution(results_df, f"{config.FIGURES_DIR}/class_distribution.png")
    plot_accuracy_summary(metrics, f"{config.FIGURES_DIR}/accuracy_summary.png")

    print(f"\nAll visualizations saved to {config.FIGURES_DIR}/")

    # close everything to save memory
    plt.close('all')


if __name__ == "__main__":
    # load results and make visualizations
    try:
        results_df = pd.read_csv(f"{config.RESULTS_DIR}/predictions.csv")
        generate_all_visualizations(results_df)
    except FileNotFoundError:
        print("Results file not found. Please run inference.py first.")
