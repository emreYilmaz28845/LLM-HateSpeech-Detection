# data_loader.py - handles downloading and preparing the dataset

import pandas as pd
import numpy as np
import os
import requests
from sklearn.model_selection import train_test_split
import config


def download_dataset():
    # downloads the csv file from github if we dont have it already
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
    # loads the csv and prints some basic stats about it
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

    # try to find which column has the labels
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
    # cleans up the data - removes nulls, duplicates, and converts labels to 0/1
    print("\n" + "="*50)
    print("PREPROCESSING DATA")
    print("="*50)

    original_size = len(df)

    # get rid of rows with missing values
    df = df.dropna(subset=[text_col, label_col])
    print(f"Removed {original_size - len(df)} rows with null values")

    # get rid of duplicate texts
    size_before = len(df)
    df = df.drop_duplicates(subset=[text_col])
    print(f"Removed {size_before - len(df)} duplicate texts")

    # convert labels to 0 and 1
    # the dataset uses "Toxic" and "Not Toxic" so we need to handle that
    if df[label_col].dtype == 'object':
        df['label'] = df[label_col].str.lower().str.strip().map({
            'toxic': 1, 'not toxic': 0, 'non-toxic': 0, 'nontoxic': 0,
            'yes': 1, 'no': 0, '1': 1, '0': 0
        })
    else:
        df['label'] = df[label_col].astype(int)

    # make sure we have a 'text' column
    if text_col != 'text':
        df['text'] = df[text_col]

    # only keep the columns we need
    df = df[['text', 'label']].copy()

    print(f"\nFinal dataset size: {len(df)}")
    print(f"Class distribution:")
    print(df['label'].value_counts())

    return df


def create_balanced_subset(df, samples_per_class=100):
    # takes equal number of toxic and non-toxic samples
    # this way the model isnt biased towards one class
    print("\n" + "="*50)
    print("CREATING BALANCED SUBSET")
    print("="*50)

    np.random.seed(config.RANDOM_SEED)

    # grab toxic samples
    toxic_samples = df[df['label'] == 1].sample(
        n=min(samples_per_class, len(df[df['label'] == 1])),
        random_state=config.RANDOM_SEED
    )

    # grab non-toxic samples
    non_toxic_samples = df[df['label'] == 0].sample(
        n=min(samples_per_class, len(df[df['label'] == 0])),
        random_state=config.RANDOM_SEED
    )

    # combine and shuffle them
    balanced_df = pd.concat([toxic_samples, non_toxic_samples]).sample(
        frac=1, random_state=config.RANDOM_SEED
    ).reset_index(drop=True)

    print(f"Balanced subset size: {len(balanced_df)}")
    print(f"Toxic samples: {len(toxic_samples)}")
    print(f"Non-toxic samples: {len(non_toxic_samples)}")

    return balanced_df


def split_data(df, test_ratio=0.8):
    # splits data into train and test
    # train set is small because we only use it for few-shot examples
    # most of the data goes to test set for evaluation
    print("\n" + "="*50)
    print("SPLITTING DATA")
    print("="*50)

    train_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        stratify=df['label'],  # keeps the class balance in both sets
        random_state=config.RANDOM_SEED
    )

    print(f"Train set (for few-shot examples): {len(train_df)}")
    print(f"Test set (for evaluation): {len(test_df)}")

    return train_df, test_df


def save_processed_data(processed_df, train_df, test_df):
    # saves everything to csv files
    os.makedirs("data", exist_ok=True)

    processed_df.to_csv(config.PROCESSED_DATA_PATH, index=False)
    train_df.to_csv(config.TRAIN_DATA_PATH, index=False)
    test_df.to_csv(config.TEST_DATA_PATH, index=False)

    print(f"\nSaved processed data to {config.PROCESSED_DATA_PATH}")
    print(f"Saved train data to {config.TRAIN_DATA_PATH}")
    print(f"Saved test data to {config.TEST_DATA_PATH}")


def prepare_dataset():
    # main function that runs all the steps

    # step 1: download
    filepath = download_dataset()

    # step 2: load and look at the data
    df, label_col = load_and_explore_data(filepath)

    # figure out which columns to use
    text_col = 'text' if 'text' in df.columns else df.columns[0]
    if label_col is None:
        label_col = 'is_toxic' if 'is_toxic' in df.columns else df.columns[1]

    # step 3: clean it up
    processed_df = preprocess_data(df, text_col, label_col)

    # step 4: make it balanced
    balanced_df = create_balanced_subset(
        processed_df,
        samples_per_class=config.SAMPLE_SIZE_PER_CLASS
    )

    # step 5: split into train/test
    train_df, test_df = split_data(balanced_df, test_ratio=config.TEST_SPLIT_RATIO)

    # step 6: save to files
    save_processed_data(balanced_df, train_df, test_df)

    return train_df, test_df


if __name__ == "__main__":
    train_df, test_df = prepare_dataset()
    print("\n" + "="*50)
    print("DATA PREPARATION COMPLETE")
    print("="*50)
