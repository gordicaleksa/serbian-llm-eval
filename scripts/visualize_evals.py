import json
import os
from collections import defaultdict


import matplotlib.pyplot as plt
import numpy as np


categories_str = "arc_easy,arc_challenge,boolq,hellaswag,openbookqa,piqa,winogrande,triviaqa,nq_open"
categories = categories_str.split(",")

results_dir = "/home/aleksa/Projects/eval/english/lm-evaluation-harness/results_jsons"

results = defaultdict(list)
for result_file in os.listdir(results_dir):
    if result_file.startswith('triviaqa'):
        name = result_file.split(".json")[0].split("_")[1]
    else:
        name = result_file.split(".json")[0].split("_")[0]

    with open(os.path.join(results_dir, result_file), "r") as f:
        data = json.load(f)["results"]
        for category in categories:
            if category in data:
                results[name].append((category, data[category]["acc"] if "acc" in data[category] else data[category]["em"]))

# sort the results according to categories
for key in results.keys():
    results[key].sort(key=lambda x: categories.index(x[0]))

mistral_7B = [t[1] for t in results["mistral"]]
llama_2_7B = [t[1] for t in results["llama2"]]
gpt_orao = [t[1] for t in results["orao"]]
yugo_gpt = [t[1] for t in results["yugo"]]

bar_width = 0.2  # width of the bars

r1 = np.arange(len(mistral_7B))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]
r4 = [x + bar_width for x in r3]

# Creating the figure and the bar plot
fig, ax = plt.subplots(figsize=(14, 8))

ax.bar(r1, yugo_gpt, color='#E9A426', width=bar_width, label='YugoGPT')
ax.bar(r2, mistral_7B, color='#B45C5C', width=bar_width, label='Mistral 7B')
ax.bar(r3, llama_2_7B, color='#89C8C2', width=bar_width, label='LLaMA 2 7B')
ax.bar(r4, gpt_orao, color='#C1DA75', width=bar_width, label='gpt2-orao')

# Adding labels and title
# ax.set_xlabel('Benchmark', fontweight='bold', fontsize=15)
ax.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=15)
ax.set_title('Performance of YugoGPT against Mistral 7B and other baselines on Serbian LLM Eval', fontsize=18)
ax.set_xticks([r + bar_width for r in range(len(mistral_7B))], categories, fontweight='bold')

# Creating legend & Show plot
ax.legend()
plt.xticks(rotation=0)
plt.tight_layout()

# Saving the figure
plt.savefig('bar_chart_comparison.png')
plt.show()