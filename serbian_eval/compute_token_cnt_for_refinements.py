import os
import json
import cyrtranslit
from tqdm import tqdm
import tiktoken
enc = tiktoken.encoding_for_model("gpt-4")
assert enc.decode(enc.encode("hello world")) == "hello world"


def get_tokens(text):
    return len(enc.encode(text))


def get_keys_of_interest(task_name):
    task_to_keys_of_interest = {
        # Common Sense
        "hellaswag": ["query", "choices"],
        "winogrande": ["sentence", "option1", "option2"],
        "piqa": ["goal", "choices"],
        "openbookqa": ["query", "choices"],
        "arc_easy": ["query", "choices"],
        "arc_challenge": ["query", "choices"],
        # Knowledge
        "nq_open": ["question", "answer"],
        "triviaqa": ["question", "answer/value", "answer/aliases"],
        # Reading Comprehension
        "boolq": ["passage", "question"],
    }

    return task_to_keys_of_interest[task_name]


num_tokens = 0
root_dir = "/home/aleksa/Projects/eval/english/lm-evaluation-harness/serbian_eval/transliterated"
num_json_objects = 0

for file in os.listdir(root_dir):
    if file.endswith(".jsonl"):
        print('*' * 80)
        print(file)
        print('*' * 80)

        # Copilot: grab the prefix before the first mention of "train" or "test"
        if "train" in file:
            task_name = file.split("_train")[0]
        elif "test" in file:
            task_name = file.split("_test")[0]
        else:
            raise ValueError("Unexpected file name")

        assert task_name in ["arc_easy", "arc_challenge", "boolq", "hellaswag", "nq_open", "openbookqa", "piqa", "triviaqa", "winogrande"]

        keys_of_interest = get_keys_of_interest(task_name)
        first_level_keys = set([x.split('/')[0] for x in keys_of_interest])
        second_level_keys = set([x.split('/')[1] for x in keys_of_interest if len(x.split('/')) > 1])

        # Open the file and then read one json object from each line at a time
        file_path = os.path.join(root_dir, file)
        with open(file_path, 'r') as f:
            json_objects = []
            for line in f:
                json_objects.append(json.loads(line))

            num_json_objects += len(json_objects)

        for json_obj in tqdm(json_objects):

            for outer_key, outer_value in json_obj.items():
                if not outer_key in first_level_keys:
                    continue

                if isinstance(outer_value, str):
                    num_tokens += get_tokens(outer_value)

                elif isinstance(outer_value, list):
                    assert all(isinstance(x, str) for x in outer_value), f'Expected all values in list to be str, but found {outer_value}'

                    for _, inner_value in enumerate(outer_value):
                        num_tokens += get_tokens(inner_value)

                elif isinstance(outer_value, dict):
                    for inner_key, inner_value in outer_value.items():
                        if not inner_key in second_level_keys:
                            continue

                        if isinstance(inner_value, str):
                            num_tokens += get_tokens(inner_value)

                        elif isinstance(inner_value, list):
                            assert all(isinstance(x, str) for x in inner_value)

                            for _, x in enumerate(inner_value):
                                num_tokens += get_tokens(x)
                        else:
                            raise RuntimeError(f"Unexpected value type in doc in {outer_key} -> {inner_key}")
                else:
                    raise RuntimeError("Unexpected value type in doc")


print(f'Number of json objects: {num_json_objects}')

print(f"Total number of input tokens: {num_tokens}")
print(f"Total number of output tokens: {num_tokens}")


PRICE_PER_1K_OUT_TOKEN_GPT_4_TURBO = 0.03
PRICE_PER_1K_IN_TOKEN_GPT_4_TURBO = 0.01
PRICE_PER_1K_OUT_TOKEN_GPT_4 = 0.06
PRICE_PER_1K_IN_TOKEN_GPT_4 = 0.03


print(f"Total cost of input tokens: ${num_tokens / 1000 * PRICE_PER_1K_IN_TOKEN_GPT_4_TURBO}")
print(f"Total cost of output tokens: ${num_tokens / 1000 * PRICE_PER_1K_OUT_TOKEN_GPT_4_TURBO}")
print(f"Total cost of input tokens: ${num_tokens / 1000 * PRICE_PER_1K_IN_TOKEN_GPT_4}")
print(f"Total cost of output tokens: ${num_tokens / 1000 * PRICE_PER_1K_OUT_TOKEN_GPT_4}")
