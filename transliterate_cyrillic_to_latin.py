import os
import json
import cyrtranslit
from tqdm import tqdm


# get current directory of this file
root_dir = os.path.dirname(os.path.realpath(__file__), 'serbian_eval')
out_dir = os.path.join(root_dir, 'transliterated')
os.makedirs(out_dir, exist_ok=True)


def transliterate_to_lat_fn(text):
    return cyrtranslit.to_latin(text)


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

            transliterated_json_objs = []
            for json_obj in tqdm(json_objects):
                transliterated_json_obj = json_obj.copy()  # create a copy of the doc

                for outer_key, outer_value in json_obj.items():
                    if not outer_key in first_level_keys:
                        continue

                    if isinstance(outer_value, str):
                        transliterated_json_obj[outer_key] = transliterate_to_lat_fn(outer_value)

                    elif isinstance(outer_value, list):
                        assert all(isinstance(x, str) for x in outer_value), f'Expected all values in list to be str, but found {outer_value}'

                        translated_list = []
                        for _, inner_value in enumerate(outer_value):
                            translated_list.append(transliterate_to_lat_fn(inner_value))
                        transliterated_json_obj[outer_key] = translated_list

                    elif isinstance(outer_value, dict):
                        for inner_key, inner_value in outer_value.items():
                            if not inner_key in second_level_keys:
                                continue

                            if isinstance(inner_value, str):
                                transliterated_json_obj[outer_key][inner_key] = transliterate_to_lat_fn(inner_value)

                            elif isinstance(inner_value, list):
                                assert all(isinstance(x, str) for x in inner_value)

                                translated_list = []
                                for _, x in enumerate(inner_value):
                                    translated_list.append(transliterate_to_lat_fn(x))
                                transliterated_json_obj[outer_key][inner_key] = translated_list
                            else:
                                raise RuntimeError(f"Unexpected value type in doc in {outer_key} -> {inner_key}")
                    else:
                        raise RuntimeError("Unexpected value type in doc")

                transliterated_json_objs.append(transliterated_json_obj)

        # Write the transliterated json objects to a new file
        out_file = os.path.join(out_dir, file)
        with open(out_file, 'w') as f:
            for json_obj in transliterated_json_objs:
                f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')