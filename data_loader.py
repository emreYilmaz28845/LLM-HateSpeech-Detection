"""
Data loading and preprocessing module for toxicity detection.
"""

import pandas as pd
import numpy as np
import os
import requests
from sklearn.model_selection import train_test_split
import config


def download_dataset():
    """
    Download the toxicity dataset from the specified URL.
    Returns the path to the downloaded file.
    """
    os.makedirs("data", exist_ok=True)

    if os.path.exists(config.RAW_DATA_PATH):
        print(f"Dataset already exists at {config.RAW_DATA_PATH}")
        return config.RAW_DATA_PATH

    print(f"Downloading dataset from {config.DATASET_URL}...")
    response = requests.get(config.DATASET_URL)
    response.raise_for_status()

    with open(config.RAW_DATA_PATH, 'w', encoding='utf-8') as f:
        f.write(response.text)

    print(f"Dataset saved to {config.RAW_DATA_PATH}")
    return config.RAW_DATA_PATH


def load_and_explore_data(filepath):
    """
    Load dataset and perform basic EDA.
    """
    print("\n" + "="*50)
    print("LOADING AND EXPLORING DATA")
    print("="*50)

    df = pd.read_csv(filepath)

    print(f"\nDataset shape: {df.shape}")
    print(f"\nColumn names: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())

    print(f"\nData types:")
    print(df.dtypes)

    print(f"\nMissing values:")
    print(df.isnull().sum())

    # Check for label column - might be named differently
    label_col = None
    for col in ['label', 'is_toxic', 'toxic', 'toxicity', 'class']:
        if col in df.columns:
            label_col = col
            break

    if label_col:
        print(f"\nLabel distribution (column: {label_col}):")
        print(df[label_col].value_counts())
        print(f"\nLabel proportions:")
        print(df[label_col].value_counts(normalize=True))

    return df, label_col


def preprocess_data(df, text_col='text', label_col='is_toxic'):
    """
    Clean and preprocess the dataset.
    """
    print("\n" + "="*50)
    print("PREPROCESSING DATA")
    print("="*50)

    original_size = len(df)

    # Remove null values
    df = df.dropna(subset=[text_col, label_col])
    print(f"Removed {original_size - len(df)} rows with null values")

    # Remove duplicates
    size_before = len(df)
    df = df.drop_duplicates(subset=[text_col])
    print(f"Removed {size_before - len(df)} duplicate texts")

    # Standardize labels to binary (0 = non-toxic, 1 = toxic)
    if df[label_col].dtype == 'object':
        # Normalize labels - handle various formats
        df['label'] = df[label_col].str.lower().str.strip().map({
            'toxic': 1, 'not toxic': 0, 'non-toxic': 0, 'nontoxic': 0,
            'yes': 1, 'no': 0, '1': 1, '0': 0
        })
    else:
        df['label'] = df[label_col].astype(int)

    # Create text column if different name
    if text_col != 'text':
        df['text'] = df[text_col]

    # Keep only relevant columns
    df = df[['text', 'label']].copy()

    print(f"\nFinal dataset size: {len(df)}")
    print(f"Class distribution:")
    print(df['label'].value_counts())

    return df


def create_balanced_subset(df, samples_per_class=100):
    """
    Create a balanced subset with equal samples from each class.
    """
    print("\n" + "="*50)
    print("CREATING BALANCED SUBSET")
    print("="*50)

    np.random.seed(config.RANDOM_SEED)

    toxic_samples = df[df['label'] == 1].sample(
        n=min(samples_per_class, len(df[df['label'] == 1])),
        random_state=config.RANDOM_SEED
    )

    non_toxic_samples = df[df['label'] == 0].sample(
        n=min(samples_per_class, len(df[df['label'] == 0])),
        random_state=config.RANDOM_SEED
    )

    balanced_df = pd.concat([toxic_samples, non_toxic_samples]).sample(
        frac=1, random_state=config.RANDOM_SEED
    ).reset_index(drop=True)

    print(f"Balanced subset size: {len(balanced_df)}")
    print(f"Toxic samples: {len(toxic_samples)}")
    print(f"Non-toxic samples: {len(non_toxic_samples)}")

    return balanced_df


def split_data(df, test_ratio=0.8):
    """
    Split data into train (for few-shot examples) and test sets.
    """
    print("\n" + "="*50)
    print("SPLITTING DATA")
    print("="*50)

    train_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        stratify=df['label'],
        random_state=config.RANDOM_SEED
    )

    print(f"Train set (for few-shot examples): {len(train_df)}")
    print(f"Test set (for evaluation): {len(test_df)}")

    return train_df, test_df


def save_processed_data(processed_df, train_df, test_df):
    """
    Save processed datasets to CSV files.
    """
    os.makedirs("data", exist_ok=True)

    processed_df.to_csv(config.PROCESSED_DATA_PATH, index=False)
    train_df.to_csv(config.TRAIN_DATA_PATH, index=False)
    test_df.to_csv(config.TEST_DATA_PATH, index=False)

    print(f"\nSaved processed data to {config.PROCESSED_DATA_PATH}")
    print(f"Saved train data to {config.TRAIN_DATA_PATH}")
    print(f"Saved test data to {config.TEST_DATA_PATH}")


def prepare_dataset():
    """
    Main function to download, process, and prepare the dataset.
    """
    # Download dataset
    filepath = download_dataset()

    # Load and explore
    df, label_col = load_and_explore_data(filepath)

    # Determine text column
    text_col = 'text' if 'text' in df.columns else df.columns[0]
    if label_col is None:
        label_col = 'is_toxic' if 'is_toxic' in df.columns else df.columns[1]

    # Preprocess
    processed_df = preprocess_data(df, text_col, label_col)

    # Create balanced subset
    balanced_df = create_balanced_subset(
        processed_df,
        samples_per_class=config.SAMPLE_SIZE_PER_CLASS
    )

    # Split data
    train_df, test_df = split_data(balanced_df, test_ratio=config.TEST_SPLIT_RATIO)

    # Save
    save_processed_data(balanced_df, train_df, test_df)

    return train_df, test_df


if __name__ == "__main__":
    train_df, test_df = prepare_dataset()
    print("\n" + "="*50)
    print("DATA PREPARATION COMPLETE")
    print("="*50)
