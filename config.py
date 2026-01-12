"""
Configuration settings for the Toxicity Detection project.
"""

# Model settings
MODEL_NAME = "llama3.2:3b"
OLLAMA_HOST = "http://localhost:11434"

# Dataset settings
DATASET_URL = "https://raw.githubusercontent.com/surge-ai/toxicity/main/toxicity_en.csv"
RAW_DATA_PATH = "data/raw_toxicity.csv"
PROCESSED_DATA_PATH = "data/processed_toxicity.csv"
TRAIN_DATA_PATH = "data/train.csv"
TEST_DATA_PATH = "data/test.csv"

# Sampling settings
SAMPLE_SIZE_PER_CLASS = 100  # Number of samples per class for balanced testing
TEST_SPLIT_RATIO = 0.8  # 80% for test, 20% for few-shot examples
RANDOM_SEED = 42

# Few-shot settings
NUM_FEW_SHOT_EXAMPLES = 5  # Number of examples to include in prompt

# Output paths
RESULTS_DIR = "results"
FIGURES_DIR = "results/figures"
REPORTS_DIR = "results/reports"

# Inference settings
MAX_RETRIES = 3
TIMEOUT = 60  # seconds
