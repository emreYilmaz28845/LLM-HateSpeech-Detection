"""
Runs the whole toxicity project end to end, from loading data to writing reports.
"""

import os
import sys
import time
from datetime import datetime

# make sure Python can find the local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from data_loader import prepare_dataset
from prompt_design import PromptManager
from inference import run_inference_pipeline
from metrics import calculate_and_report
from visualization import generate_all_visualizations
from error_analysis import run_error_analysis
from report_generator import generate_all_reports
import pandas as pd


def print_header():
    """Print the little banner for the run."""
    print("\n" + "="*70)
    print("   TOXICITY/HATE SPEECH DETECTION SYSTEM")
    print("   Using Few-Shot Learning with Ollama (llama3.2:3b)")
    print("="*70)
    print(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")


def check_ollama():
    """Check that Ollama is up and the model is already pulled."""
    print("Checking Ollama setup...")
    try:
        import ollama
        models_response = ollama.list()
        # handles both response shapes from the client
        if hasattr(models_response, 'models'):
            models = models_response.models
            model_names = [m.model if hasattr(m, 'model') else m.get('model', '') for m in models]
        else:
            models = models_response.get('models', [])
            model_names = [m.get('name', m.get('model', '')) for m in models]

        if config.MODEL_NAME not in model_names and f"{config.MODEL_NAME}:latest" not in model_names:
            # also check names without the version tag
            base_name = config.MODEL_NAME.split(':')[0]
            found = any(base_name in name for name in model_names)
            if not found:
                print(f"WARNING: Model {config.MODEL_NAME} not found!")
                print(f"Available models: {model_names}")
                print(f"Please run: ollama pull {config.MODEL_NAME}")
                return False

        print(f"Ollama is running")
        print(f"Model {config.MODEL_NAME} is available")
        return True
    except Exception as e:
        print(f"ERROR: Could not connect to Ollama: {e}")
        print("Please ensure Ollama is running (ollama serve)")
        return False


def run_step(step_num, step_name, func, *args, **kwargs):
    """Run one pipeline step, time it, and surface any errors."""
    print(f"\n{'='*70}")
    print(f"STEP {step_num}: {step_name}")
    print(f"{'='*70}")

    start_time = time.time()
    try:
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        print(f"\nStep {step_num} completed in {elapsed:.2f} seconds")
        return result
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\nStep {step_num} failed after {elapsed:.2f} seconds")
        print(f"Error: {e}")
        raise


def run_full_pipeline():
    """
    Run everything for the toxicity project in one go.
    """
    print_header()
    start_time = time.time()

    # make sure Ollama and the model are ready before doing anything else
    if not check_ollama():
        print("\nPlease fix Ollama setup and try again.")
        return None

    # set up output folders if they are missing
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # Step 1: data prep
    train_df, test_df = run_step(
        1, "Data Preparation",
        prepare_dataset
    )

    # Step 2: run model inference
    results_df = run_step(
        2, "Model Inference",
        run_inference_pipeline
    )

    # Step 3: compute metrics
    metrics = run_step(
        3, "Metrics Calculation",
        calculate_and_report,
        results_df
    )

    # Step 4: draw the figures
    run_step(
        4, "Visualization Generation",
        generate_all_visualizations,
        results_df, metrics
    )

    # Step 5: dig into mistakes
    run_step(
        5, "Error Analysis",
        run_error_analysis,
        results_df
    )

    # Step 6: write out the reports
    run_step(
        6, "Report Generation",
        generate_all_reports,
        results_df
    )

    # wrap-up summary
    total_time = time.time() - start_time
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"\nTotal execution time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    print(f"\nOutput files:")
    print(f"  - Predictions: {config.RESULTS_DIR}/predictions.csv")
    print(f"  - Metrics: {config.RESULTS_DIR}/metrics.csv")
    print(f"  - Figures: {config.FIGURES_DIR}/")
    print(f"  - Error Analysis: {config.REPORTS_DIR}/error_analysis.txt")
    print(f"  - Final Report: {config.REPORTS_DIR}/final_report.md")

    if metrics:
        print(f"\n{'='*70}")
        print("FINAL RESULTS SUMMARY")
        print(f"{'='*70}")
        print(f"  Accuracy:     {metrics['accuracy']*100:.2f}%")
        print(f"  F1-Score:     {metrics['f1_macro']*100:.2f}%")
        print(f"  Precision:    {metrics['precision_macro']*100:.2f}%")
        print(f"  Recall:       {metrics['recall_macro']*100:.2f}%")

    return results_df, metrics


def run_inference_only():
    """Just run inference (assumes the data is already prepped)."""
    print_header()

    if not check_ollama():
        return None

    # if test data is missing, prep it first
    if not os.path.exists(config.TEST_DATA_PATH):
        print("Test data not found. Running data preparation first...")
        prepare_dataset()

    results_df = run_inference_pipeline()
    metrics = calculate_and_report(results_df)

    return results_df, metrics


def run_analysis_only():
    """Only do analysis and visuals (needs existing predictions)."""
    print_header()

    results_path = f"{config.RESULTS_DIR}/predictions.csv"
    if not os.path.exists(results_path):
        print(f"Results file not found at {results_path}")
        print("Please run inference first using: python main.py")
        return None

    results_df = pd.read_csv(results_path)

    metrics = calculate_and_report(results_df)
    generate_all_visualizations(results_df, metrics)
    run_error_analysis(results_df)
    generate_all_reports(results_df)

    return results_df, metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Toxicity Detection System')
    parser.add_argument('--mode', choices=['full', 'inference', 'analysis'],
                       default='full',
                       help='Pipeline mode: full (default), inference-only, or analysis-only')

    args = parser.parse_args()

    if args.mode == 'full':
        run_full_pipeline()
    elif args.mode == 'inference':
        run_inference_only()
    elif args.mode == 'analysis':
        run_analysis_only()
