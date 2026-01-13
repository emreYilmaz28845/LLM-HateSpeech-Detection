# config.py - all the settings for the project

# which model to use
MODEL_NAME = "llama3.2:3b"
OLLAMA_HOST = "http://localhost:11434"

# where to get the data from
DATASET_URL = "https://raw.githubusercontent.com/surge-ai/toxicity/main/toxicity_en.csv"
RAW_DATA_PATH = "data/raw_toxicity.csv"
PROCESSED_DATA_PATH = "data/processed_toxicity.csv"
TRAIN_DATA_PATH = "data/train.csv"
TEST_DATA_PATH = "data/test.csv"

# how much data to use
SAMPLE_SIZE_PER_CLASS = 100  # 100 toxic + 100 non-toxic
TEST_SPLIT_RATIO = 0.8  # 80% goes to test set, rest is for few-shot examples
RANDOM_SEED = 42  # so results are reproducible

# few-shot stuff
NUM_FEW_SHOT_EXAMPLES = 5  # how many examples to put in the prompt

# where to save outputs
RESULTS_DIR = "results"
FIGURES_DIR = "results/figures"
REPORTS_DIR = "results/reports"

# inference params
MAX_RETRIES = 3  # try 3 times if ollama fails
TIMEOUT = 60  # max wait time in seconds
