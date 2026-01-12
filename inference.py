"""
Inference module for toxicity classification using Ollama.
"""

import ollama
import re
import time
import pandas as pd
from tqdm import tqdm
import config
from prompt_design import PromptManager


def parse_response(response_text):
    """
    Parse and clean the model output to extract classification.
    Returns: 1 for toxic, 0 for non-toxic, -1 for invalid response
    """
    # Clean the response
    cleaned = response_text.strip().lower()

    # Remove any punctuation and extra whitespace
    cleaned = re.sub(r'[^\w\s-]', '', cleaned)
    cleaned = cleaned.strip()

    # Check for toxic/non-toxic
    if 'non-toxic' in cleaned or 'non toxic' in cleaned or 'nontoxic' in cleaned:
        return 0
    elif 'toxic' in cleaned:
        return 1
    else:
        # Try to find the first word
        first_word = cleaned.split()[0] if cleaned.split() else ""
        if first_word in ['toxic']:
            return 1
        elif first_word in ['non-toxic', 'nontoxic', 'non']:
            return 0

    return -1  # Invalid response


def classify_text(text, prompt_manager, model_name=None, retries=None):
    """
    Classify a single text using the Ollama model.

    Args:
        text: The text to classify
        prompt_manager: PromptManager instance with loaded examples
        model_name: Name of the model to use
        retries: Number of retries on failure

    Returns:
        dict with 'prediction' (0, 1, or -1) and 'raw_response'
    """
    model_name = model_name or config.MODEL_NAME
    retries = retries or config.MAX_RETRIES

    prompt = prompt_manager.get_prompt(text)

    for attempt in range(retries):
        try:
            response = ollama.generate(
                model=model_name,
                prompt=prompt,
                options={
                    'temperature': 0.1,  # Low temperature for consistent results
                    'num_predict': 20,   # Short response expected
                }
            )

            raw_response = response['response']
            prediction = parse_response(raw_response)

            return {
                'prediction': prediction,
                'raw_response': raw_response.strip()
            }

        except Exception as e:
            print(f"Attempt {attempt + 1}/{retries} failed: {e}")
            if attempt < retries - 1:
                time.sleep(2)  # Wait before retry

    return {
        'prediction': -1,
        'raw_response': f"Error after {retries} attempts"
    }


def evaluate_dataset(test_df, prompt_manager, model_name=None, verbose=True):
    """
    Evaluate the model on the entire test dataset.

    Args:
        test_df: DataFrame with 'text' and 'label' columns
        prompt_manager: PromptManager instance
        model_name: Name of the model
        verbose: Whether to show progress bar

    Returns:
        DataFrame with predictions and evaluation results
    """
    model_name = model_name or config.MODEL_NAME

    results = []
    iterator = tqdm(test_df.iterrows(), total=len(test_df), desc="Classifying") if verbose else test_df.iterrows()

    for idx, row in iterator:
        text = row['text']
        true_label = row['label']

        result = classify_text(text, prompt_manager, model_name)

        results.append({
            'text': text,
            'true_label': true_label,
            'predicted_label': result['prediction'],
            'raw_response': result['raw_response'],
            'correct': true_label == result['prediction'] if result['prediction'] != -1 else False
        })

    results_df = pd.DataFrame(results)

    # Calculate summary stats
    valid_predictions = results_df[results_df['predicted_label'] != -1]
    total = len(results_df)
    valid = len(valid_predictions)
    invalid = total - valid
    correct = results_df['correct'].sum()
    accuracy = correct / total if total > 0 else 0

    print(f"\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Total samples: {total}")
    print(f"Valid predictions: {valid} ({valid/total*100:.1f}%)")
    print(f"Invalid predictions: {invalid} ({invalid/total*100:.1f}%)")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy*100:.2f}%")

    return results_df


def run_inference_pipeline(train_path=None, test_path=None, output_path=None):
    """
    Run the complete inference pipeline.

    Args:
        train_path: Path to training data (for few-shot examples)
        test_path: Path to test data
        output_path: Path to save results

    Returns:
        DataFrame with all results
    """
    train_path = train_path or config.TRAIN_DATA_PATH
    test_path = test_path or config.TEST_DATA_PATH
    output_path = output_path or f"{config.RESULTS_DIR}/predictions.csv"

    print("\n" + "="*50)
    print("STARTING INFERENCE PIPELINE")
    print("="*50)

    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"Train samples (for few-shot): {len(train_df)}")
    print(f"Test samples: {len(test_df)}")

    # Initialize prompt manager
    print("\nInitializing prompt manager...")
    prompt_manager = PromptManager(train_df)
    prompt_manager.display_examples()

    # Run evaluation
    print(f"\nRunning evaluation with {config.MODEL_NAME}...")
    results_df = evaluate_dataset(test_df, prompt_manager)

    # Save results
    import os
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    return results_df


if __name__ == "__main__":
    results = run_inference_pipeline()
