import ast
import json
import os

from tqdm import tqdm
from openai import OpenAI
import openai

# OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()


def gpt3_5(message):
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    messages.extend(message)
    response = client.responses.create(
        model="gpt-3.5-turbo",
        input=messages,
    )
    return response.output_text


# In context samples
instruct = "Which object does given expressions refer to? Answer in dict format"
sample = """{"1": "next to the red, brick building and before the little tree resides the white truck.", "2": "to the left of the young, happy, and smiling boy stands the young girl who, in turn, is positioned to the left of the white dishwasher.", "3": "this tree can be found on the left side of the white building."}"""
response = """{"1": "the white truck", "2": "the young girl", "3": "this tree."}"""


def split_dict(d, k):
    """Split a dictionary into batches"""
    dict_list = []
    sub_dict = {}
    i = 0
    for key, value in d.items():
        sub_dict[key] = value
        i += 1
        if i == k:
            dict_list.append(sub_dict)
            sub_dict = {}
            i = 0
    if sub_dict:
        dict_list.append(sub_dict)
    return dict_list


def load_batches(json_path, batch_size):
    """Load data from JSON and split into batches"""
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)
    return split_dict(data, batch_size)


def batch_extract(
    input_path,
    batch_size=8,
    error_log_path='wrong_extraction.txt'
):
    """Extract target objects in batches"""
    dataset = load_batches(input_path, batch_size)
    results = {}
    for i, prompt in tqdm(enumerate(dataset), total=len(dataset)):
        messages = [
            {"role": "user", "content": instruct + sample},
            {"role": "assistant", "content": response},
            {"role": "user", "content": json.dumps(prompt)}
        ]
        truncated_text = gpt3_5(messages)
        start_index = truncated_text.find('{')
        if start_index != -1:
            truncated_text = truncated_text[start_index:]
        try:
            result_dict = ast.literal_eval(truncated_text)
        except Exception as e:
            with open(error_log_path, 'a', encoding='utf-8') as file:
                file.write(f"{truncated_text}\n")
            continue
        try:
            results.update(result_dict)
        except (TypeError, ValueError) as e:
            print(f"Error updating results: {e}")
            with open(error_log_path, 'a', encoding='utf-8') as file:
                file.write(f"{result_dict}\n")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract targets from expressions using GPT.")
    parser.add_argument('--input', required=True, help='Path to input JSON file.')
    parser.add_argument('--output', required=True, help='Path to save the output JSON file.')
    parser.add_argument('--error_log', default='wrong_extraction.txt', help='Path to save error logs.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for extraction.')
    args = parser.parse_args()

    final_results = batch_extract(
        input_path=args.input,
        batch_size=args.batch_size,
        error_log_path=args.error_log
    )
    with open(args.output, "w", encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    print(f'Saved to {args.output}')
