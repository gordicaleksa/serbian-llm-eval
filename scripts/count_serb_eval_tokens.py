import json

import numpy as np
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4")
assert enc.decode(enc.encode("hello world")) == "hello world"

def get_tokens(text):
    return len(enc.encode(text))

num_tokens_per_request = "/home/aleksa/Projects/eval/english/lm-evaluation-harness/serbian_eval/transliterated/nq_open_train_partial_0_87924_end.jsonl"

with open(num_tokens_per_request, "r") as f:
    data = f.readlines()
    data = [json.loads(line) for line in data]

with open("/home/aleksa/Projects/eval/english/lm-evaluation-harness/serbian_eval/prompts/nq_open.txt", "r") as f:
    prompt_data = f.read()

prompt_tokens = get_tokens(prompt_data)

total_num_tokens = 0
num_tokens_per_request = []
for d in data:
    num_tokens_request = 0
    num_tokens_request += get_tokens(d["question"]) + prompt_tokens + sum([get_tokens(a) for a in d["answer"]])
    num_tokens_per_request.append(num_tokens_request)

    total_num_tokens += get_tokens(d["question"])
    for a in d["answer"]:
        total_num_tokens += get_tokens(a)

num_tokens_per_request = np.asarray(num_tokens_per_request)
print(np.average(num_tokens_per_request))
print(np.median(num_tokens_per_request))
print(np.max(num_tokens_per_request))
print(np.min(num_tokens_per_request))

print(2 * total_num_tokens + prompt_tokens * len(data))  # input + output + prompt