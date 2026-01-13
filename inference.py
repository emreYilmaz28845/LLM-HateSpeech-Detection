# inference.py - this is where the actual model calls happen

import ollama
import re
import time
import pandas as pd
from tqdm import tqdm
import config
from prompt_design import PromptManager


def parse_response(response_text):
    """
    takes the model output and figures out if its toxic or not
    returns 1 for toxic, 0 for non-toxic, -1 if we cant tell
    """
    # clean up the response first
    cleaned = response_text.strip().lower()

    # get rid of punctuation and extra spaces
    cleaned = re.sub(r'[^\w\s-]', '', cleaned)
    cleaned = cleaned.strip()

    # check if its non-toxic first (has to come before toxic check)
    if 'non-toxic' in cleaned or 'non toxic' in cleaned or 'nontoxic' in cleaned:
        return 0
    elif 'toxic' in cleaned:
        return 1
    else:
        # sometimes the model just says one word
        first_word = cleaned.split()[0] if cleaned.split() else ""
        if first_word in ['toxic']:
            return 1
        elif first_word in ['non-toxic', 'nontoxic', 'non']:
            return 0

    return -1  # couldnt figure out what the model meant


def classify_text(text, prompt_manager, model_name=None, retries=None):
    """
    sends one text to the model and gets a classification back

    text: what we want to classify
    prompt_manager: has all our few-shot examples ready
    model_name: which model to use (defaults to config)
    retries: how many times to try if it fails

    returns a dict with prediction (0, 1, or -1) and the raw model response
    """
    model_name = model_name or config.MODEL_NAME
    retries = retries or config.MAX_RETRIES

    prompt = prompt_manager.get_prompt(text)

    for attempt in range(retries):
        try:
            # this is the actual ollama call
            response = ollama.generate(
                model=model_name,
                prompt=prompt,
                options={
                    'temperature': 0.1,  # low temp = more consistent answers
                    'num_predict': 20,   # we only need a short answer (max tokens)
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
                time.sleep(2)  # wait a bit before trying again

    return {
        'prediction': -1,
        'raw_response': f"Error after {retries} attempts"
    }


def evaluate_dataset(test_df, prompt_manager, model_name=None, verbose=True):
    """
    runs the model on all the test data

    test_df: dataframe with text and label columns
    prompt_manager: our prompt manager with examples
    model_name: which model
    verbose: show progress bar or not

    returns dataframe with all predictions
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

    # print out how we did
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
    runs everything from start to finish

    train_path: where to get training data (for few-shot examples)
    test_path: where to get test data
    output_path: where to save results
    """
    train_path = train_path or config.TRAIN_DATA_PATH
    test_path = test_path or config.TEST_DATA_PATH
    output_path = output_path or f"{config.RESULTS_DIR}/predictions.csv"

    print("\n" + "="*50)
    print("STARTING INFERENCE PIPELINE")
    print("="*50)

    # load the data
    print("\nLoading data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"Train samples (for few-shot): {len(train_df)}")
    print(f"Test samples: {len(test_df)}")

    # set up the prompt manager
    print("\nInitializing prompt manager...")
    prompt_manager = PromptManager(train_df)
    prompt_manager.display_examples()

    # run the actual evaluation
    print(f"\nRunning evaluation with {config.MODEL_NAME}...")
    results_df = evaluate_dataset(test_df, prompt_manager)

    # save everything
    import os
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    return results_df


if __name__ == "__main__":
    results = run_inference_pipeline()
