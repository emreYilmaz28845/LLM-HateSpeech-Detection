# Toxicity Detection System - Final Report

**Generated:** 2026-01-12 00:32:40

**Model:** llama3.2:3b

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Methodology](#methodology)
3. [Dataset Statistics](#dataset-statistics)
4. [Model Configuration](#model-configuration)
5. [Performance Metrics](#performance-metrics)
6. [Visualizations](#visualizations)
7. [Error Analysis](#error-analysis)
8. [Conclusions](#conclusions)
9. [Future Improvements](#future-improvements)

---

## Project Overview

This project implements a toxicity/hate speech detection system using a Large Language Model (LLM)
with few-shot learning. The system classifies text as either "toxic" or "non-toxic" using the
Ollama framework with the llama3.2:3b model.

### Objectives
- Build a binary text classifier for toxicity detection
- Utilize few-shot prompting for classification without fine-tuning
- Evaluate model performance using standard classification metrics
- Analyze errors to understand model limitations


## Methodology

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


## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 160 |
| Valid Predictions | 155 |
| Invalid Predictions | 5 |
| Samples per Class | 100 (balanced) |
| Few-Shot Examples | 5 |


## Model Configuration

| Parameter | Value |
|-----------|-------|
| Model | llama3.2:3b |
| Framework | Ollama |
| Temperature | 0.1 (low for consistency) |
| Max Tokens | 20 |
| Prompt Type | Few-shot (5 examples) |


## Performance Metrics

### Overall Metrics

| Metric | Score |
|--------|-------|
| **Accuracy** | 93.55% |
| **Precision (Macro)** | 93.54% |
| **Recall (Macro)** | 93.54% |
| **F1-Score (Macro)** | 93.54% |


### Per-Class Metrics

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Non-toxic | 93.75% | 93.75% | 93.75% |
| Toxic | 93.33% | 93.33% | 93.33% |

### Confusion Matrix

|  | Predicted Non-toxic | Predicted Toxic |
|--|---------------------|-----------------|
| **Actual Non-toxic** | 75 (TN) | 5 (FP) |
| **Actual Toxic** | 5 (FN) | 70 (TP) |

## Visualizations

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


## Error Analysis

### Error Summary
| Error Type | Count | Percentage |
|------------|-------|------------|
| False Positives | 5 | 3.2% |
| False Negatives | 5 | 3.2% |
| Invalid Responses | 5 | 3.1% |

### Analysis
- **False Positives**: Non-toxic content incorrectly flagged as toxic
- **False Negatives**: Toxic content missed by the classifier
- **Invalid Responses**: Model outputs that couldn't be parsed

See `results/reports/error_analysis.txt` for detailed error analysis with examples.


## Conclusions

### Key Findings

1. **Overall Performance**: The model demonstrates strong performance with 93.5% accuracy

2. **Class Balance**:
   - Precision for toxic: 93.3%
   - Precision for non-toxic: 93.8%

3. **Error Patterns**:
   - False Positives: 5 cases
   - False Negatives: 5 cases

4. **Model Reliability**: 155/160 (96.9%) responses were valid and parseable


## Future Improvements

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


---

*Report generated by AI525 Project 4 - Toxicity Detection System*

*Model: llama3.2:3b via Ollama*