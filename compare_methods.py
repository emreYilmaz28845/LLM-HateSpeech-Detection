"""
Compare Few-Shot vs Zero-Shot performance for toxicity classification.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import ollama
import re
import time

import config
from prompt_design import PromptManager, create_zero_shot_prompt
from inference import parse_response
from metrics import calculate_metrics


def classify_text_with_prompt(text, prompt, model_name=None, retries=3):
    """
    Classify text using a given prompt.
    """
    model_name = model_name or config.MODEL_NAME

    for attempt in range(retries):
        try:
            response = ollama.generate(
                model=model_name,
                prompt=prompt,
                options={
                    'temperature': 0.1,
                    'num_predict': 20,
                }
            )
            raw_response = response['response']
            prediction = parse_response(raw_response)
            return {
                'prediction': prediction,
                'raw_response': raw_response.strip()
            }
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)

    return {'prediction': -1, 'raw_response': 'Error'}


def evaluate_method(test_df, prompt_func, method_name, verbose=True):
    """
    Evaluate a classification method on the test dataset.

    Args:
        test_df: DataFrame with 'text' and 'label' columns
        prompt_func: Function that takes text and returns a prompt
        method_name: Name of the method for display
        verbose: Show progress bar

    Returns:
        DataFrame with predictions
    """
    results = []
    iterator = tqdm(test_df.iterrows(), total=len(test_df),
                   desc=f"{method_name}") if verbose else test_df.iterrows()

    for idx, row in iterator:
        text = row['text']
        true_label = row['label']

        prompt = prompt_func(text)
        result = classify_text_with_prompt(text, prompt)

        results.append({
            'text': text,
            'true_label': true_label,
            'predicted_label': result['prediction'],
            'raw_response': result['raw_response'],
            'correct': true_label == result['prediction'] if result['prediction'] != -1 else False
        })

    return pd.DataFrame(results)


def run_comparison():
    """
    Run comparison between few-shot and zero-shot methods.
    """
    print("\n" + "="*70)
    print("   FEW-SHOT vs ZERO-SHOT COMPARISON")
    print("   Toxicity Classification with llama3.2:3b")
    print("="*70 + "\n")

    # Load data
    print("Loading data...")
    train_df = pd.read_csv(config.TRAIN_DATA_PATH)
    test_df = pd.read_csv(config.TEST_DATA_PATH)
    print(f"Test samples: {len(test_df)}")

    # Initialize prompt manager for few-shot
    prompt_manager = PromptManager(train_df, num_examples=5)

    # Define prompt functions
    def few_shot_prompt(text):
        return prompt_manager.get_prompt(text, use_few_shot=True)

    def zero_shot_prompt(text):
        return create_zero_shot_prompt(text)

    # Run evaluations
    print("\n" + "="*50)
    print("Running Zero-Shot Evaluation...")
    print("="*50)
    zero_shot_results = evaluate_method(test_df, zero_shot_prompt, "Zero-Shot")

    print("\n" + "="*50)
    print("Running Few-Shot Evaluation (5 examples)...")
    print("="*50)
    few_shot_results = evaluate_method(test_df, few_shot_prompt, "Few-Shot")

    # Calculate metrics for both
    print("\n" + "="*70)
    print("CALCULATING METRICS")
    print("="*70)

    zero_shot_metrics = calculate_metrics(zero_shot_results)
    few_shot_metrics = calculate_metrics(few_shot_results)

    # Create comparison table
    comparison_data = {
        'Metric': ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)',
                   'Valid Predictions %', 'False Positives', 'False Negatives'],
        'Zero-Shot': [
            f"{zero_shot_metrics['accuracy']*100:.2f}%",
            f"{zero_shot_metrics['precision_macro']*100:.2f}%",
            f"{zero_shot_metrics['recall_macro']*100:.2f}%",
            f"{zero_shot_metrics['f1_macro']*100:.2f}%",
            f"{zero_shot_metrics['valid_samples']/zero_shot_metrics['total_samples']*100:.1f}%",
            zero_shot_metrics['false_positives'],
            zero_shot_metrics['false_negatives']
        ],
        'Few-Shot (5 examples)': [
            f"{few_shot_metrics['accuracy']*100:.2f}%",
            f"{few_shot_metrics['precision_macro']*100:.2f}%",
            f"{few_shot_metrics['recall_macro']*100:.2f}%",
            f"{few_shot_metrics['f1_macro']*100:.2f}%",
            f"{few_shot_metrics['valid_samples']/few_shot_metrics['total_samples']*100:.1f}%",
            few_shot_metrics['false_positives'],
            few_shot_metrics['false_negatives']
        ]
    }

    comparison_df = pd.DataFrame(comparison_data)

    # Print comparison table
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    print("\n" + comparison_df.to_string(index=False))

    # Calculate improvement
    accuracy_improvement = (few_shot_metrics['accuracy'] - zero_shot_metrics['accuracy']) * 100
    f1_improvement = (few_shot_metrics['f1_macro'] - zero_shot_metrics['f1_macro']) * 100

    print("\n" + "="*70)
    print("IMPROVEMENT SUMMARY")
    print("="*70)
    print(f"\nAccuracy improvement: {accuracy_improvement:+.2f}%")
    print(f"F1-Score improvement: {f1_improvement:+.2f}%")

    if accuracy_improvement > 0:
        print(f"\nFew-shot learning improved accuracy by {accuracy_improvement:.2f} percentage points!")
    elif accuracy_improvement < 0:
        print(f"\nZero-shot actually performed better by {-accuracy_improvement:.2f} percentage points!")
    else:
        print("\nBoth methods performed equally!")

    # Save results
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.FIGURES_DIR, exist_ok=True)

    zero_shot_results.to_csv(f"{config.RESULTS_DIR}/zero_shot_predictions.csv", index=False)
    few_shot_results.to_csv(f"{config.RESULTS_DIR}/few_shot_predictions.csv", index=False)
    comparison_df.to_csv(f"{config.RESULTS_DIR}/method_comparison.csv", index=False)

    print(f"\nResults saved to {config.RESULTS_DIR}/")

    # Generate comparison visualization
    generate_comparison_plot(zero_shot_metrics, few_shot_metrics)

    return {
        'zero_shot': {'results': zero_shot_results, 'metrics': zero_shot_metrics},
        'few_shot': {'results': few_shot_results, 'metrics': few_shot_metrics},
        'comparison': comparison_df
    }


def generate_comparison_plot(zero_shot_metrics, few_shot_metrics):
    """
    Generate a comparison visualization.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Bar chart comparison
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    zero_shot_values = [
        zero_shot_metrics['accuracy'] * 100,
        zero_shot_metrics['precision_macro'] * 100,
        zero_shot_metrics['recall_macro'] * 100,
        zero_shot_metrics['f1_macro'] * 100
    ]
    few_shot_values = [
        few_shot_metrics['accuracy'] * 100,
        few_shot_metrics['precision_macro'] * 100,
        few_shot_metrics['recall_macro'] * 100,
        few_shot_metrics['f1_macro'] * 100
    ]

    x = np.arange(len(metrics_names))
    width = 0.35

    bars1 = axes[0].bar(x - width/2, zero_shot_values, width, label='Zero-Shot', color='#3498db', alpha=0.8)
    bars2 = axes[0].bar(x + width/2, few_shot_values, width, label='Few-Shot', color='#e74c3c', alpha=0.8)

    axes[0].set_ylabel('Score (%)', fontsize=12)
    axes[0].set_title('Zero-Shot vs Few-Shot Performance', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics_names)
    axes[0].legend()
    axes[0].set_ylim(0, 105)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        axes[0].annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        axes[0].annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    # Plot 2: Confusion matrix comparison (side by side)
    cm_zero = zero_shot_metrics['confusion_matrix']
    cm_few = few_shot_metrics['confusion_matrix']

    # Create combined data for grouped bar
    categories = ['True\nNegative', 'False\nPositive', 'False\nNegative', 'True\nPositive']
    zero_values = [cm_zero[0,0], cm_zero[0,1], cm_zero[1,0], cm_zero[1,1]]
    few_values = [cm_few[0,0], cm_few[0,1], cm_few[1,0], cm_few[1,1]]

    x = np.arange(len(categories))
    bars1 = axes[1].bar(x - width/2, zero_values, width, label='Zero-Shot', color='#3498db', alpha=0.8)
    bars2 = axes[1].bar(x + width/2, few_values, width, label='Few-Shot', color='#e74c3c', alpha=0.8)

    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('Confusion Matrix Comparison', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(categories)
    axes[1].legend()

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        axes[1].annotate(f'{int(height)}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar in bars2:
        height = bar.get_height()
        axes[1].annotate(f'{int(height)}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{config.FIGURES_DIR}/method_comparison.png", dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to {config.FIGURES_DIR}/method_comparison.png")
    plt.close()


if __name__ == "__main__":
    results = run_comparison()
