"""
Report generation module for toxicity classification project.
"""

import pandas as pd
import os
from datetime import datetime
import config
from metrics import calculate_metrics


def generate_markdown_report(results_df, metrics=None, output_path=None):
    """
    Generate a comprehensive markdown report.
    """
    output_path = output_path or f"{config.REPORTS_DIR}/final_report.md"
    os.makedirs(config.REPORTS_DIR, exist_ok=True)

    if metrics is None:
        metrics = calculate_metrics(results_df)

    report = []

    # Header
    report.append("# Toxicity Detection System - Final Report")
    report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"\n**Model:** {config.MODEL_NAME}")
    report.append("\n---\n")

    # Table of Contents
    report.append("## Table of Contents")
    report.append("1. [Project Overview](#project-overview)")
    report.append("2. [Methodology](#methodology)")
    report.append("3. [Dataset Statistics](#dataset-statistics)")
    report.append("4. [Model Configuration](#model-configuration)")
    report.append("5. [Performance Metrics](#performance-metrics)")
    report.append("6. [Visualizations](#visualizations)")
    report.append("7. [Error Analysis](#error-analysis)")
    report.append("8. [Conclusions](#conclusions)")
    report.append("9. [Future Improvements](#future-improvements)")
    report.append("\n---\n")

    # Project Overview
    report.append("## Project Overview")
    report.append("""
This project implements a toxicity/hate speech detection system using a Large Language Model (LLM)
with few-shot learning. The system classifies text as either "toxic" or "non-toxic" using the
Ollama framework with the llama3.2:3b model.

### Objectives
- Build a binary text classifier for toxicity detection
- Utilize few-shot prompting for classification without fine-tuning
- Evaluate model performance using standard classification metrics
- Analyze errors to understand model limitations
""")

    # Methodology
    report.append("\n## Methodology")
    report.append("""
### Approach: Few-Shot Learning

Instead of traditional supervised learning requiring thousands of labeled examples, this project
uses **few-shot learning** where the model learns the task from just a handful of examples
embedded directly in the prompt.

### Pipeline Steps:
1. **Data Preparation**: Download and preprocess a toxicity dataset
2. **Balanced Sampling**: Create a balanced test set with equal toxic/non-toxic samples
3. **Few-Shot Prompt Design**: Select representative examples and create structured prompts
4. **Inference**: Run classification on test samples using Ollama API
5. **Evaluation**: Calculate metrics and generate visualizations
6. **Error Analysis**: Identify patterns in misclassifications
""")

    # Dataset Statistics
    report.append("\n## Dataset Statistics")
    report.append(f"""
| Metric | Value |
|--------|-------|
| Total Samples | {metrics['total_samples']} |
| Valid Predictions | {metrics['valid_samples']} |
| Invalid Predictions | {metrics['invalid_samples']} |
| Samples per Class | {config.SAMPLE_SIZE_PER_CLASS} (balanced) |
| Few-Shot Examples | {config.NUM_FEW_SHOT_EXAMPLES} |
""")

    # Model Configuration
    report.append("\n## Model Configuration")
    report.append(f"""
| Parameter | Value |
|-----------|-------|
| Model | {config.MODEL_NAME} |
| Framework | Ollama |
| Temperature | 0.1 (low for consistency) |
| Max Tokens | 20 |
| Prompt Type | Few-shot ({config.NUM_FEW_SHOT_EXAMPLES} examples) |
""")

    # Performance Metrics
    report.append("\n## Performance Metrics")
    report.append("\n### Overall Metrics")
    report.append(f"""
| Metric | Score |
|--------|-------|
| **Accuracy** | {metrics['accuracy']*100:.2f}% |
| **Precision (Macro)** | {metrics['precision_macro']*100:.2f}% |
| **Recall (Macro)** | {metrics['recall_macro']*100:.2f}% |
| **F1-Score (Macro)** | {metrics['f1_macro']*100:.2f}% |
""")

    report.append("\n### Per-Class Metrics")
    report.append("""
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|""")
    report.append(f"| Non-toxic | {metrics['precision_nontoxic']*100:.2f}% | {metrics['recall_nontoxic']*100:.2f}% | {metrics['f1_nontoxic']*100:.2f}% |")
    report.append(f"| Toxic | {metrics['precision_toxic']*100:.2f}% | {metrics['recall_toxic']*100:.2f}% | {metrics['f1_toxic']*100:.2f}% |")

    report.append("\n### Confusion Matrix")
    cm = metrics['confusion_matrix']
    report.append("""
|  | Predicted Non-toxic | Predicted Toxic |
|--|---------------------|-----------------|""")
    report.append(f"| **Actual Non-toxic** | {cm[0,0]} (TN) | {cm[0,1]} (FP) |")
    report.append(f"| **Actual Toxic** | {cm[1,0]} (FN) | {cm[1,1]} (TP) |")

    # Visualizations
    report.append("\n## Visualizations")
    report.append("""
The following visualizations were generated and saved to the `results/figures/` directory:

1. **Confusion Matrix** (`confusion_matrix.png`)
   - Heatmap showing prediction distribution

2. **Metrics Comparison** (`metrics_comparison.png`)
   - Bar chart comparing precision, recall, F1 for each class

3. **Prediction Distribution** (`prediction_distribution.png`)
   - Pie chart of correct/incorrect/invalid predictions

4. **Class Distribution** (`class_distribution.png`)
   - Comparison of true vs predicted label distributions

5. **Accuracy Summary** (`accuracy_summary.png`)
   - Overview of key performance metrics
""")

    # Error Analysis
    report.append("\n## Error Analysis")
    report.append(f"""
### Error Summary
| Error Type | Count | Percentage |
|------------|-------|------------|
| False Positives | {metrics['false_positives']} | {metrics['false_positives']/metrics['valid_samples']*100:.1f}% |
| False Negatives | {metrics['false_negatives']} | {metrics['false_negatives']/metrics['valid_samples']*100:.1f}% |
| Invalid Responses | {metrics['invalid_samples']} | {metrics['invalid_samples']/metrics['total_samples']*100:.1f}% |

### Analysis
- **False Positives**: Non-toxic content incorrectly flagged as toxic
- **False Negatives**: Toxic content missed by the classifier
- **Invalid Responses**: Model outputs that couldn't be parsed

See `results/reports/error_analysis.txt` for detailed error analysis with examples.
""")

    # Conclusions
    report.append("\n## Conclusions")

    accuracy = metrics['accuracy'] * 100
    if accuracy >= 80:
        performance_assessment = "The model demonstrates strong performance"
    elif accuracy >= 70:
        performance_assessment = "The model shows moderate performance"
    else:
        performance_assessment = "The model has room for improvement"

    report.append(f"""
### Key Findings

1. **Overall Performance**: {performance_assessment} with {accuracy:.1f}% accuracy

2. **Class Balance**:
   - Precision for toxic: {metrics['precision_toxic']*100:.1f}%
   - Precision for non-toxic: {metrics['precision_nontoxic']*100:.1f}%

3. **Error Patterns**:
   - False Positives: {metrics['false_positives']} cases
   - False Negatives: {metrics['false_negatives']} cases

4. **Model Reliability**: {metrics['valid_samples']}/{metrics['total_samples']} ({metrics['valid_samples']/metrics['total_samples']*100:.1f}%) responses were valid and parseable
""")

    # Future Improvements
    report.append("\n## Future Improvements")
    report.append("""
### Potential Enhancements

1. **Prompt Engineering**
   - Experiment with different prompt structures
   - Add more diverse few-shot examples
   - Include edge cases in examples

2. **Model Selection**
   - Try larger models (llama3.1:8b, mistral:7b)
   - Compare performance across different models

3. **Data Augmentation**
   - Increase test set size
   - Include more diverse toxic content types

4. **Fine-tuning**
   - Consider fine-tuning on domain-specific data
   - Use LoRA for efficient adaptation

5. **Ensemble Methods**
   - Combine predictions from multiple prompts
   - Implement confidence-based filtering

6. **Multi-class Extension**
   - Extend to multi-label classification
   - Detect specific toxicity types (hate, threat, insult, etc.)
""")

    # Footer
    report.append("\n---")
    report.append(f"\n*Report generated by AI525 Project 4 - Toxicity Detection System*")
    report.append(f"\n*Model: {config.MODEL_NAME} via Ollama*")

    report_text = "\n".join(report)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\nMarkdown report saved to {output_path}")

    return report_text


def generate_summary_table(results_df, metrics=None, output_path=None):
    """
    Generate a summary CSV table of all results.
    """
    output_path = output_path or f"{config.REPORTS_DIR}/summary.csv"

    if metrics is None:
        metrics = calculate_metrics(results_df)

    summary_data = {
        'Metric': [
            'Total Samples',
            'Valid Predictions',
            'Invalid Predictions',
            'Accuracy',
            'Precision (Toxic)',
            'Precision (Non-toxic)',
            'Precision (Macro)',
            'Recall (Toxic)',
            'Recall (Non-toxic)',
            'Recall (Macro)',
            'F1-Score (Toxic)',
            'F1-Score (Non-toxic)',
            'F1-Score (Macro)',
            'True Positives',
            'True Negatives',
            'False Positives',
            'False Negatives'
        ],
        'Value': [
            metrics['total_samples'],
            metrics['valid_samples'],
            metrics['invalid_samples'],
            f"{metrics['accuracy']*100:.2f}%",
            f"{metrics['precision_toxic']*100:.2f}%",
            f"{metrics['precision_nontoxic']*100:.2f}%",
            f"{metrics['precision_macro']*100:.2f}%",
            f"{metrics['recall_toxic']*100:.2f}%",
            f"{metrics['recall_nontoxic']*100:.2f}%",
            f"{metrics['recall_macro']*100:.2f}%",
            f"{metrics['f1_toxic']*100:.2f}%",
            f"{metrics['f1_nontoxic']*100:.2f}%",
            f"{metrics['f1_macro']*100:.2f}%",
            metrics['true_positives'],
            metrics['true_negatives'],
            metrics['false_positives'],
            metrics['false_negatives']
        ]
    }

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_path, index=False)
    print(f"Summary table saved to {output_path}")

    return summary_df


def generate_all_reports(results_df):
    """
    Generate all reports.
    """
    print("\n" + "="*50)
    print("GENERATING REPORTS")
    print("="*50)

    metrics = calculate_metrics(results_df)

    if metrics is None:
        print("Cannot generate reports: no valid metrics")
        return

    os.makedirs(config.REPORTS_DIR, exist_ok=True)

    generate_markdown_report(results_df, metrics)
    generate_summary_table(results_df, metrics)

    print(f"\nAll reports saved to {config.REPORTS_DIR}/")


if __name__ == "__main__":
    try:
        results_df = pd.read_csv(f"{config.RESULTS_DIR}/predictions.csv")
        generate_all_reports(results_df)
    except FileNotFoundError:
        print("Results file not found. Please run inference.py first.")
