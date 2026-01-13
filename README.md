# Toxicity/Hate Speech Detection System

A toxicity detection system using Large Language Models (LLMs) with few-shot learning via Ollama.

## Overview

This project implements a binary text classifier that detects toxic/hate speech content using the `llama3.2:3b` model. It demonstrates both zero-shot and few-shot prompting approaches and provides comprehensive evaluation metrics.

## Features

- **Few-Shot Learning**: Classify text using examples embedded in the prompt
- **Zero-Shot Classification**: Classify without examples for comparison
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- **Visualizations**: Auto-generated charts and graphs
- **Error Analysis**: Detailed analysis of misclassifications
- **Report Generation**: Markdown reports with all findings

## Results

| Method | Accuracy | F1-Score | Valid Predictions |
|--------|----------|----------|-------------------|
| Zero-Shot | 95.60% | 95.60% | 99.4% |
| Few-Shot (5 examples) | 94.23% | 94.23% | 97.5% |

## Project Structure

```
AI525_Project4/
├── main.py              # Main pipeline orchestrator
├── config.py            # Configuration settings
├── data_loader.py       # Dataset download & preprocessing
├── prompt_design.py     # Few-shot prompt engineering
├── inference.py         # Model inference with Ollama
├── metrics.py           # Performance metrics calculation
├── visualization.py     # Chart/graph generation
├── error_analysis.py    # Error pattern analysis
├── report_generator.py  # Report generation
├── compare_methods.py   # Few-shot vs Zero-shot comparison
├── requirements.txt     # Python dependencies
├── data/                # Dataset files (auto-generated)
└── results/             # Output files (auto-generated)
    ├── predictions.csv
    ├── metrics.csv
    ├── figures/         # Visualization PNGs
    └── reports/         # Analysis reports
```

## Usage

### Run Full Pipeline
```bash
python main.py
```
This will:
1. Download and preprocess the toxicity dataset
2. Run few-shot classification on test samples
3. Calculate metrics and generate visualizations
4. Perform error analysis
5. Generate final report

### Run Only Inference
```bash
python main.py --mode inference
```

### Run Only Analysis (if predictions exist)
```bash
python main.py --mode analysis
```

### Compare Few-Shot vs Zero-Shot
```bash
python compare_methods.py
```

## Configuration

Edit `config.py` to customize:

```python
MODEL_NAME = "llama3.2:3b"      # Ollama model to use
SAMPLE_SIZE_PER_CLASS = 100     # Samples per class for testing
NUM_FEW_SHOT_EXAMPLES = 5       # Number of examples in prompt
TEST_SPLIT_RATIO = 0.8          # Train/test split ratio
```

## Output Files

After running the pipeline:

- `results/predictions.csv` - All predictions with raw model responses
- `results/metrics.csv` - Performance metrics summary
- `results/figures/` - Visualization images:
  - `confusion_matrix.png`
  - `metrics_comparison.png`
  - `prediction_distribution.png`
  - `class_distribution.png`
  - `accuracy_summary.png`
  - `method_comparison.png` (after running compare_methods.py)
- `results/reports/` - Analysis reports:
  - `final_report.md`
  - `error_analysis.txt`
  - `false_positives.csv`
  - `false_negatives.csv`
  - `summary.csv`

## Dataset

Uses the [Surge AI Toxicity Dataset](https://github.com/surge-ai/toxicity) containing 1000 English text samples labeled as "Toxic" or "Not Toxic".

## Methodology

### Few-Shot Prompting
The system embeds 5 representative examples (mix of toxic and non-toxic) directly in the prompt to guide the model's classification.

### Zero-Shot Prompting
The model classifies text based only on task instructions without any examples.

### Evaluation
- Balanced test set (equal toxic/non-toxic samples)
- Standard classification metrics
- Confusion matrix analysis
- Error pattern identification

## Requirements

- Python 3.8+
- Ollama running locally
- llama3.2:3b model (or modify config for different model)

## License

MIT License
