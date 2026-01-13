# prompt_design.py - creates the prompts we send to the LLM

import pandas as pd
import config


def select_few_shot_examples(train_df, num_examples=5):
    # picks examples from training data to use in the prompt
    # tries to get a mix of toxic and non-toxic
    num_toxic = num_examples // 2 + num_examples % 2  # if odd, give extra to toxic
    num_non_toxic = num_examples // 2

    toxic_examples = train_df[train_df['label'] == 1].head(num_toxic)
    non_toxic_examples = train_df[train_df['label'] == 0].head(num_non_toxic)

    # mix them together so its not all toxic then all non-toxic
    examples = []
    toxic_list = toxic_examples.to_dict('records')
    non_toxic_list = non_toxic_examples.to_dict('records')

    while toxic_list or non_toxic_list:
        if non_toxic_list:
            examples.append(non_toxic_list.pop(0))
        if toxic_list:
            examples.append(toxic_list.pop(0))

    return examples


def format_example(text, label):
    # formats one example for the prompt
    label_str = "toxic" if label == 1 else "non-toxic"
    return f'Text: "{text}"\nClassification: {label_str}'


def create_few_shot_prompt(examples, new_text):
    # builds the full prompt with instructions, examples, and the text to classify
    prompt = """You are a toxicity classifier. Your task is to classify text as either "toxic" or "non-toxic".

Toxic text includes: hate speech, insults, threats, profanity, harassment, discrimination, or any content that is offensive, harmful, or inappropriate.

Non-toxic text is: respectful, constructive, neutral, or positive communication that does not contain harmful content.

Here are some examples:

"""
    # add the few-shot examples
    for i, example in enumerate(examples, 1):
        prompt += f"Example {i}:\n"
        prompt += format_example(example['text'], example['label'])
        prompt += "\n\n"

    # add the text we want to classify
    prompt += f"""Now classify the following text. Respond with ONLY the word "toxic" or "non-toxic" - nothing else.

Text: "{new_text}"
Classification:"""

    return prompt


def create_zero_shot_prompt(new_text):
    # same as above but without any examples
    # used for comparing few-shot vs zero-shot performance
    prompt = f"""You are a toxicity classifier. Classify the following text as either "toxic" or "non-toxic".

Toxic text includes: hate speech, insults, threats, profanity, harassment, discrimination, or any content that is offensive, harmful, or inappropriate.

Non-toxic text is: respectful, constructive, neutral, or positive communication that does not contain harmful content.

Respond with ONLY the word "toxic" or "non-toxic" - nothing else.

Text: "{new_text}"
Classification:"""

    return prompt


class PromptManager:
    # this class handles everything related to prompts
    # it stores the few-shot examples and builds prompts when needed

    def __init__(self, train_df=None, num_examples=None):
        self.num_examples = num_examples or config.NUM_FEW_SHOT_EXAMPLES
        self.examples = []

        if train_df is not None:
            self.load_examples(train_df)

    def load_examples(self, train_df):
        # loads examples from the training dataframe
        self.examples = select_few_shot_examples(train_df, self.num_examples)
        print(f"Loaded {len(self.examples)} few-shot examples")

    def load_examples_from_file(self, filepath=None):
        # loads examples from a csv file
        filepath = filepath or config.TRAIN_DATA_PATH
        train_df = pd.read_csv(filepath)
        self.load_examples(train_df)

    def get_prompt(self, text, use_few_shot=True):
        # returns the prompt for a given text
        # can do either few-shot or zero-shot
        if use_few_shot and self.examples:
            return create_few_shot_prompt(self.examples, text)
        else:
            return create_zero_shot_prompt(text)

    def display_examples(self):
        # prints out the examples we're using (for debugging)
        print("\n" + "="*50)
        print("FEW-SHOT EXAMPLES")
        print("="*50)

        for i, example in enumerate(self.examples, 1):
            label_str = "toxic" if example['label'] == 1 else "non-toxic"
            print(f"\nExample {i} ({label_str}):")
            print(f"  {example['text'][:100]}..." if len(example['text']) > 100 else f"  {example['text']}")


def demo_prompt():
    # just a demo function to test if everything works
    try:
        train_df = pd.read_csv(config.TRAIN_DATA_PATH)
    except FileNotFoundError:
        print("Training data not found. Please run data_loader.py first.")
        return

    pm = PromptManager(train_df)
    pm.display_examples()

    sample_text = "This is a test message to classify."
    prompt = pm.get_prompt(sample_text)

    print("\n" + "="*50)
    print("SAMPLE PROMPT")
    print("="*50)
    print(prompt)


if __name__ == "__main__":
    demo_prompt()
