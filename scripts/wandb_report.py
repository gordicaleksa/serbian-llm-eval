import json
import wandb

wandb.init(project="serbian_llm_eval", name="winogrande_test_serbian")

winogrande_file_srp = "/home/aleksa/Projects/eval/english/lm-evaluation-harness/serbian_eval/refined_no_triviaqa/winogrande_test_partial_0_1266_end.jsonl"
winogrande_file_eng = "/home/aleksa/Projects/eval/english/lm-evaluation-harness/wino_test.jsonl"

num_samples = 100

with open(winogrande_file_srp, "r") as f:
    winogrande = [json.loads(line) for line in f.readlines()]
    winogrande = winogrande[:num_samples]

table = wandb.Table(
    columns=[
        "id",
        "sentence",
        "option1",
        "option2",
        "answer",
    ]
)
for i, sample in enumerate(winogrande):
    table.add_data(
        i,
        sample["sentence"],
        sample["option1"],
        sample["option2"],
        sample["answer"],
    )

wandb.run.log({f"Winogrande Serbian": table})

with open(winogrande_file_eng, "r") as f:
    winogrande = [json.loads(line) for line in f.readlines()]
    winogrande = winogrande[:num_samples]

table = wandb.Table(
    columns=[
        "id",
        "sentence",
        "option1",
        "option2",
        "answer",
    ]
)

for i, sample in enumerate(winogrande):
    table.add_data(
        i,
        sample["sentence"],
        sample["option1"],
        sample["option2"],
        sample["answer"],
    )

wandb.run.log({f"Winogrande English": table})

artifact = wandb.Artifact(name="winogrande_prompts", type="prompt")
artifact.add_file(local_path="/home/aleksa/Projects/eval/english/lm-evaluation-harness/serbian_eval/prompts/winogrande_with_reasoning.txt")
artifact.add_file(local_path="/home/aleksa/Projects/eval/english/lm-evaluation-harness/serbian_eval/prompts/winogrande.txt")
wandb.log_artifact(artifact)