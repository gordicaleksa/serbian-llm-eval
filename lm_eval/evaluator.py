import asyncio
import collections
import itertools
import random
import re

import lm_eval.metrics
import lm_eval.models
import lm_eval.tasks
import lm_eval.base
from lm_eval.utils import positional_deprecated, run_task_tests
from lm_eval.models.gpt2 import HFLM

import numpy as np
import openai
from openai import OpenAI
import transformers

import json
import os
from google.cloud import translate
from tqdm import tqdm
from lm_eval.openai_handler import OpenAIHandler

@positional_deprecated
def simple_evaluate(
    model,
    model_args=None,
    tasks=[],
    num_fewshot=0,
    batch_size=None,
    max_batch_size=None,
    device=None,
    no_cache=False,
    limit=None,
    bootstrap_iters=100000,
    description_dict=None,
    check_integrity=False,
    decontamination_ngrams_path=None,
    write_out=False,
    output_base_path=None,
    translation_project_id=None,
    char_limit=500000,
    start_from_doc_index=None,
):
    """Instantiate and evaluate a model on a list of tasks.

    :param model: Union[str, LM]
        Name of model, transformers.PreTrainedModel object, or LM object, see lm_eval.models.get_model
    :param model_args: Optional[str]
        String arguments for each model class, see LM.create_from_arg_string.
        Ignored if `model` argument is a LM object.
    :param tasks: list[Union[str, Task]]
        List of task names or Task objects. Task objects will be taken to have name task.EVAL_HARNESS_NAME if defined and type(task).__name__ otherwise.
    :param num_fewshot: int
        Number of examples in few-shot context
    :param batch_size: int or str, optional
        Batch size for model
    :param max_batch_size: int, optional
        Maximal batch size to try with automatic batch size detection
    :param device: str, optional
        PyTorch device (e.g. "cpu" or "cuda:0") for running models
    :param no_cache: bool
        Whether or not to cache
    :param limit: int or float, optional
        Limit the number of examples per task (only use this for testing), If <1, limit is a percentage of the total number of examples.
    :param bootstrap_iters:
        Number of iterations for bootstrap statistics
    :param description_dict: dict[str, str]
        Dictionary of custom task descriptions of the form: `task_name: description`
    :param check_integrity: bool
        Whether to run the relevant part of the test suite for the tasks
    :param write_out: bool
        If True, write details about prompts and logits to json for all tasks
    :param output_base_path: str, optional
        Directory to which detailed eval info will be written. Defaults to present working dir.
    :return
        Dictionary of results
    """
    random.seed(1234)
    np.random.seed(1234)

    assert tasks != [], "No tasks specified"

    if isinstance(model, str):
        if model_args is None:
            model_args = ""
        # lm = lm_eval.models.get_model(model).create_from_arg_string(
        #     model_args,
        #     {
        #         "batch_size": batch_size,
        #         "max_batch_size": max_batch_size,
        #         "device": device,
        #     },
        # )
        lm = None
    elif isinstance(model, transformers.PreTrainedModel):
        lm = lm_eval.models.get_model("hf-causal")(
            pretrained=model,
            batch_size=batch_size,
            max_batch_size=max_batch_size,
        )
        no_cache = True
    else:
        assert isinstance(model, lm_eval.base.LM)
        lm = model

    # if not no_cache:
    #     lm = lm_eval.base.CachingLM(
    #         lm,
    #         "lm_cache/"
    #         + (model if isinstance(model, str) else model.model.config._name_or_path)
    #         + "_"
    #         + model_args.replace("=", "-").replace(",", "_").replace("/", "-")
    #         + ".db",
    #     )

    task_dict = lm_eval.tasks.get_task_dict(tasks)

    if check_integrity:
        run_task_tests(task_list=tasks)

    results = evaluate(
        lm=lm,
        task_dict=task_dict,
        num_fewshot=num_fewshot,
        limit=limit,
        bootstrap_iters=bootstrap_iters,
        description_dict=description_dict,
        decontamination_ngrams_path=decontamination_ngrams_path,
        write_out=write_out,
        output_base_path=output_base_path,
        translation_project_id=translation_project_id,
        char_limit=char_limit,
        start_from_doc_index=start_from_doc_index,
    )

    # add info about the model and few shot config
    model_name = None
    if isinstance(model, str):
        model_name = model
    elif isinstance(model, transformers.PreTrainedModel):
        model_name = "pretrained=" + model.config._name_or_path
    results["config"] = {
        "model": model_name,
        "model_args": model_args,
        "num_fewshot": num_fewshot,
        "batch_size": batch_size,
        "batch_sizes": list(lm.batch_sizes.values())
        if hasattr(lm, "batch_sizes")
        else [],
        "device": device,
        "no_cache": no_cache,
        "limit": limit,
        "bootstrap_iters": bootstrap_iters,
        "description_dict": description_dict,
    }

    return results


decontaminate_suffix = "_decontaminate"


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


def get_translate_fn(client, parent, target_lang, debug_mode=False):
    def translate_helper_fn(src_str):
        assert isinstance(src_str, str), f"Expected str, got {src_str}"

        if debug_mode:  # not spending money :)
            return "dummy"
        else:
            response = client.translate_text(
                request={
                    "parent": parent,
                    "contents": [src_str],
                    "mime_type": "text/plain",
                    "source_language_code": "en-US",
                    "target_language_code": target_lang,
                }
            )
            trg_str = response.translations[0].translated_text
            return trg_str

    return translate_helper_fn


def translate_dataset(task_name, translate_fn, first_level_keys, second_level_keys, all_docs, out_dir, is_test=True, start_num_chars=None, end_num_chars=None, char_limit=500000, start_from_doc_index=None):
    num_chars_dataset_current = 0
    exit_flag = False
    out_filename = f'{task_name}{"_test" if is_test else "_train"}.jsonl'
    out_path = os.path.join(out_dir, out_filename)
    is_debug = start_num_chars is None or end_num_chars is None
    try:
        with open(out_path, 'w', encoding="utf-8") as f:

            if not is_debug:
                progress_bar = tqdm(all_docs, total=end_num_chars-start_num_chars, initial=start_num_chars)

            for doc_index, doc in enumerate(all_docs):

                num_chars_dataset_old = num_chars_dataset_current

                if start_from_doc_index is not None and doc_index < start_from_doc_index:
                    continue

                translated_doc = doc.copy()  # create a copy of the doc

                if exit_flag:
                    raise Exception(f"Char limit exceeded {char_limit}. Exiting...")

                for outer_key, outer_value in doc.items():
                    if not outer_key in first_level_keys:
                        continue

                    if isinstance(outer_value, str):
                        num_chars_dataset_current += len(outer_value)
                        if start_num_chars is not None and start_num_chars + num_chars_dataset_current > char_limit:
                            exit_flag=True
                            break
                        translated_doc[outer_key] = translate_fn(outer_value)

                    elif isinstance(outer_value, list):
                        assert all(isinstance(x, str) for x in outer_value), f'Expected all values in list to be str, but found {outer_value}'

                        num_chars_dataset_current += sum(len(x) for x in outer_value)
                        if start_num_chars is not None and start_num_chars + num_chars_dataset_current > char_limit:
                            exit_flag=True
                            break

                        translated_list = []
                        for _, inner_value in enumerate(outer_value):
                            translated_list.append(translate_fn(inner_value))
                        translated_doc[outer_key] = translated_list

                    elif isinstance(outer_value, dict):
                        for inner_key, inner_value in outer_value.items():
                            if not inner_key in second_level_keys:
                                continue

                            if isinstance(inner_value, str):
                                num_chars_dataset_current += len(inner_value)
                                if start_num_chars is not None and start_num_chars + num_chars_dataset_current > char_limit:
                                    exit_flag=True
                                    break
                                translated_doc[outer_key][inner_key] = translate_fn(inner_value)

                            elif isinstance(inner_value, list):
                                assert all(isinstance(x, str) for x in inner_value)

                                num_chars_dataset_current += sum(len(x) for x in inner_value)
                                if start_num_chars is not None and start_num_chars + num_chars_dataset_current > char_limit:
                                    exit_flag=True
                                    break

                                translated_list = []
                                for _, x in enumerate(inner_value):
                                    translated_list.append(translate_fn(x))
                                translated_doc[outer_key][inner_key] = translated_list
                            else:
                                raise RuntimeError(f"Unexpected value type in doc in {outer_key} -> {inner_key}")
                    else:
                        raise RuntimeError("Unexpected value type in doc")

                if not exit_flag and not is_debug:
                    progress_bar.update(num_chars_dataset_current - num_chars_dataset_old)

                if not exit_flag and not is_debug:
                    f.write(json.dumps(translated_doc, ensure_ascii=False) + "\n")  # write the translated doc to file
                    f.flush()

            print(f"Task: {task_name}; number of chars: {num_chars_dataset_current}")

    except Exception as e:  # char limit exceeded
        # Rename output file to indicate that it is incomplete and add doc_id, after that raise again
        print(e)
        new_filename = f'{task_name}{"_test" if is_test else "_train"}_partial_{start_from_doc_index}_{doc_index-2}.jsonl'
        os.rename(out_path, os.path.join(out_dir, new_filename))
        raise e

    if not is_debug:
        new_filename = f'{task_name}{"_test" if is_test else "_train"}_partial_{start_from_doc_index}_{len(all_docs) - 1}_end.jsonl'
        os.rename(out_path, os.path.join(out_dir, new_filename))

    return num_chars_dataset_current


def translate_eval(task_dict_items, target_lang="sr", project_id=None, char_limit=500000, start_from_doc_index=None):
    assert project_id is not None, "Project ID must be specified"
    assert target_lang == "sr", "Only Serbian (Cyrillic) is supported for now"

    client = translate.TranslationServiceClient()
    location = "global"
    parent = f"projects/{project_id}/locations/{location}"

    this_file_path = os.path.dirname(os.path.realpath(__file__))
    parent_dir_path = os.path.abspath(os.path.join(this_file_path, os.pardir))
    out_dir = os.path.join(parent_dir_path, "serbian_eval")
    os.makedirs(out_dir, exist_ok=True)

    debug_translate_fn = get_translate_fn(client, parent, target_lang, debug_mode=True)
    translate_fn = get_translate_fn(client, parent, target_lang)

    num_chars_total = 0
    try:
        for task_name, task in task_dict_items:
            print('*' * 50)
            print(f"Translating task: {task_name}")
            print('*' * 50)

            if task.has_test_docs():
                task_doc_func = task.test_docs
            elif task.has_validation_docs():
                task_doc_func = task.validation_docs
            else:
                raise RuntimeError("Task has neither test_docs nor validation_docs")

            keys_of_interest = get_keys_of_interest(task_name)
            first_level_keys = set([x.split('/')[0] for x in keys_of_interest])
            second_level_keys = set([x.split('/')[1] for x in keys_of_interest if len(x.split('/')) > 1])

            task_docs = list(task_doc_func())
            num_chars_pred_total = translate_dataset(task_name, debug_translate_fn, first_level_keys, second_level_keys, task_docs, out_dir, is_test=True)
            num_chars_total += translate_dataset(task_name, translate_fn, first_level_keys, second_level_keys, task_docs, out_dir, is_test=True, start_num_chars=num_chars_total, end_num_chars=num_chars_total+num_chars_pred_total, char_limit=char_limit, start_from_doc_index=start_from_doc_index)

            if task_name in ["nq_open", "triviaqa"]:
                task_docs = list(task.training_docs())
                num_chars_pred_total = translate_dataset(task_name, debug_translate_fn, first_level_keys, second_level_keys, task_docs, out_dir, is_test=False)
                num_chars_total += translate_dataset(task_name, translate_fn, first_level_keys, second_level_keys, task_docs, out_dir, is_test=False, start_num_chars=num_chars_total, end_num_chars=num_chars_total+num_chars_pred_total, char_limit=char_limit, start_from_doc_index=start_from_doc_index)
    except Exception as e:  # char limit exceeded
        print(e)
        exit(0)
    print(f"Total number of chars: {num_chars_total}")


def get_prompt_template(prompt_templates_dir, task_name):
    path = os.path.join(prompt_templates_dir, f"{task_name}.txt")
    with open(path) as infile:
        template = infile.read()

    return template


def save_human_readable(f_human_readable, reasoning, doc_eng, old_doc_srp, doc_srp, doc_index):
    doc_eng_str = json.dumps(doc_eng, indent=4, ensure_ascii=False)
    doc_srp_str = json.dumps(doc_srp, indent=4, ensure_ascii=False)
    old_doc_srp_str = json.dumps(old_doc_srp, indent=4, ensure_ascii=False)

    f_human_readable.write('*' * 50)
    f_human_readable.write("\n")
    f_human_readable.write(f"{doc_index}. original doc (English):\n")
    f_human_readable.write(doc_eng_str)
    f_human_readable.write("\n\n")
    f_human_readable.write(f"Google Translate (Serbian):\n")
    f_human_readable.write(old_doc_srp_str)
    f_human_readable.write("\n\n")
    f_human_readable.write(f"GPT-4 reasoning:\n")
    f_human_readable.write(reasoning)
    f_human_readable.write("\n\n")
    f_human_readable.write(f"Translated doc (Serbian):\n")
    f_human_readable.write(doc_srp_str)
    f_human_readable.write("\n")
    f_human_readable.write('*' * 50)
    f_human_readable.write("\n\n")
    f_human_readable.flush()


async def refine_dataset(instructor, task_docs, task_docs_serbian, task_name, prompt_templates_dir, out_dir, in_dir, is_test, start_from_doc_index=None):
    template = get_prompt_template(prompt_templates_dir, task_name)

    NUM_ATTEMPTS_GPT4 = 3

    out_filename = f'{task_name}{"_test" if is_test else "_train"}.jsonl'
    out_path = os.path.join(out_dir, out_filename)
    out_filename_human_readable = f'{task_name}{"_test" if is_test else "_train"}_human_readable.jsonl'
    out_path_human_readable = os.path.join(out_dir, out_filename_human_readable)

    progress_bar = tqdm(task_docs, total=len(task_docs), initial=start_from_doc_index)

    api_params = {
        "temperature": 1.0,
        "top_p": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0
    }

    try:
        with open(out_path, 'w', encoding="utf-8") as f, open(out_path_human_readable, 'w', encoding="utf-8") as f_human_readable:
            for doc_index, (doc_eng, doc_srp) in enumerate(zip(task_docs, task_docs_serbian)):

                if start_from_doc_index is not None and doc_index < start_from_doc_index:
                    continue

                old_doc_srp = doc_srp.copy()  # create a copy of the doc

                if task_name in ["piqa"]:
                    # Eval method: loglikelihood of choices (this is similar to all those that were in the format question + answer, arc etc.)
                    src_goal = doc_eng["goal"]
                    src_choices = doc_eng["choices"]
                    trg_goal = doc_srp["goal"]
                    trg_choices = doc_srp["choices"]
                    assert len(src_choices) == len(trg_choices) == 2, f"Expected same number of choices, but found {len(src_choices)} and {len(trg_choices)}"
                    prompt = template.format(
                        src_goal=src_goal,
                        src_choice1=src_choices[0],
                        src_choice2=src_choices[1],
                        trg_goal=trg_goal,
                        trg_choice1=trg_choices[0],
                        trg_choice2=trg_choices[1]
                    )

                    num_attempts = NUM_ATTEMPTS_GPT4
                    while num_attempts > 0:
                        response = await instructor.gen_with_retry(
                            prompt, **api_params
                        )

                        if response is None:
                            num_attempts -= 1
                            continue

                        matches = re.findall(
                            r"REASONING:\s*(.*?)\s*SERBIAN:\s*\"goal\":\s*(.*?)\n\s*\"choice1\":\s*(.*?)\n\s*\"choice2\":\s*(.*?)\s*(?=REASONING:|$)",
                            response,
                            re.DOTALL
                        )

                        if len(matches) != 1:
                            print(f"Expected exactly one match, but found {matches}")
                            num_attempts -= 1
                            continue

                        reasoning, goal, choice1, choice2 = matches[0]

                        doc_srp["goal"] = goal
                        doc_srp["choices"] = [choice1, choice2]
                        break
                elif task_name in ["hellaswag", "openbookqa", "triviaqa"]:
                    # Eval method: loglikelihood of choices
                    src_query = doc_eng["query"]
                    src_choices = doc_eng["choices"]
                    trg_query = doc_srp["query"]
                    trg_choices = doc_srp["choices"]
                    assert len(src_choices) == len(trg_choices) == 4, f"Expected same number of choices, but found {len(src_choices)} and {len(trg_choices)}"
                    prompt = template.format(
                        src_query=src_query,
                        src_choice1=src_choices[0],
                        src_choice2=src_choices[1],
                        src_choice3=src_choices[2],
                        src_choice4=src_choices[3],
                        trg_query=trg_query,
                        trg_choice1=trg_choices[0],
                        trg_choice2=trg_choices[1],
                        trg_choice3=trg_choices[2],
                        trg_choice4=trg_choices[3],
                    )

                    num_attempts = NUM_ATTEMPTS_GPT4
                    while num_attempts > 0:
                        response = await instructor.gen_with_retry(
                            prompt, **api_params
                        )

                        if response is None:
                            num_attempts -= 1
                            continue

                        matches = re.findall(
                            r"REASONING:\s*(.*?)\s*SERBIAN:\s*\"query\":\s*(.*?)\n\s*\"choice1\":\s*(.*?)\n\s*\"choice2\":\s*(.*?)\n\s*\"choice3\":\s*(.*?)\n\s*\"choice4\":\s*(.*?)\s*(?=REASONING:|$)",
                            response,
                            re.DOTALL
                        )

                        if len(matches) != 1:
                            print(f"Expected exactly one match, but found {matches}")
                            num_attempts -= 1
                            continue

                        reasoning, query, choice1, choice2, choice3, choice4 = matches[0]

                        doc_srp["query"] = query
                        doc_srp["choices"] = [choice1, choice2, choice3, choice4]
                        break
                elif task_name in ["winogrande"]:
                    # Eval method: plugin option 1 and 2 where the "_" is  and do loglikelihood of everything right of the "_"
                    src_sentence = doc_eng["sentence"]
                    src_option1 = doc_eng["option1"]
                    src_option2 = doc_eng["option2"]
                    trg_sentence = doc_srp["sentence"]
                    trg_option1 = doc_srp["option1"]
                    trg_option2 = doc_srp["option2"]
                    prompt = template.format(
                        src_sentence=src_sentence,
                        trg_sentence=trg_sentence,
                        src_option1=src_option1,
                        src_option2=src_option2,
                        trg_option1=trg_option1,
                        trg_option2=trg_option2
                    )

                    num_attempts = NUM_ATTEMPTS_GPT4
                    while num_attempts > 0:
                        response = await instructor.gen_with_retry(
                            prompt, **api_params
                        )

                        if response is None:
                            num_attempts -= 1
                            continue

                        matches = re.findall(
                            r"REASONING:\s*(.*?)\s*SERBIAN:\s*\"sentence\":\s*(.*?)\n\s*\"option1\":\s*(.*?)\n\s*\"option2\":\s*(.*?)\s*(?=REASONING:|$)",
                            response,
                            re.DOTALL
                        )

                        if len(matches) != 1:
                            print(f"Expected exactly one match, but found {matches}")
                            num_attempts -= 1
                            continue

                        reasoning, sentence, option1, option2 = matches[0]

                        num_underscores = len(re.findall("_", sentence))

                        if num_underscores != 1:
                            print(f"Expected exactly one underscore, but found {num_underscores}")
                            num_attempts -= 1
                            continue

                        doc_srp["sentence"] = sentence
                        doc_srp["option1"] = option1
                        doc_srp["option2"] = option2
                        break

                elif task_name in ["boolq"]:
                    # Eval method: loglikelihood of "yes" and "no" given passage and question
                    src_passage = doc_eng["passage"]
                    src_question = doc_eng["question"]
                    trg_passage = doc_srp["passage"]
                    trg_question = doc_srp["question"]
                    prompt = template.format(
                        src_passage=src_passage,
                        src_question=src_question,
                        trg_passage=trg_passage,
                        trg_question=trg_question
                    )

                    num_attempts = NUM_ATTEMPTS_GPT4

                    while num_attempts > 0:
                        response = await instructor.gen_with_retry(
                            prompt, **api_params
                        )

                        if response is None:
                            num_attempts -= 1
                            continue

                        matches = re.findall(  # removed brackets compared to below
                            r"REASONING:\s*(.*?)\s*SERBIAN:\s*\"passage\":\s*(.*?)\n\s*\"question\":\s*(.*?)\s*(?=REASONING:|$)",
                            response,
                            re.DOTALL
                        )

                        if len(matches) != 1:
                            print(f"Expected exactly one match, but found {matches}")
                            num_attempts -= 1
                            continue

                        reasoning, passage, question = matches[0]

                        doc_srp["passage"] = passage
                        doc_srp["question"] = question
                        break
                elif task_name in ["arc_easy", "arc_challenge"]:
                    # Eval method: loglikelihood of choices
                    qe = doc_eng["query"]
                    qs = doc_srp["query"]
                    ce = doc_eng["choices"]
                    cs = doc_srp["choices"]
                    prompt = template.format(src_question=qe, src_answer=ce, trg_question=qs, trg_answer=cs)

                    num_attempts = NUM_ATTEMPTS_GPT4

                    while num_attempts > 0:
                        response = await instructor.gen_with_retry(
                            prompt, **api_params
                        )

                        if response is None:
                            num_attempts -= 1
                            continue

                        matches = re.findall(
                            r"REASONING:\s*(.*?)\s*SERBIAN:\s*\"question\":\s*(.*?)\n\s*\"answer\":\s*\[(.*?)\]\s*(?=REASONING:|$)",
                            response,
                            re.DOTALL
                        )

                        if len(matches) != 1:
                            print(f"Expected exactly one match, but found {matches}")
                            num_attempts -= 1
                            continue

                        reasoning, question, answer = matches[0]
                        pattern = r'\'(.*?[^\\])\'|"(.*?[^\\])"' # r'\'(.*?)\''
                        answers = [match[0] or match[1] for match in re.findall(pattern, answer)]

                        if len(answers) != len(doc_eng["choices"]) or not all(isinstance(a, str) for a in answers) or any(a == "" for a in answers):
                            print(f"Expected same number of answers, but found {len(answers)} and {len(doc_eng['choices'])}")
                            num_attempts -= 1
                            continue

                        doc_srp["query"] = question
                        doc_srp["choices"] = answers
                        break
                elif task_name == "nq_open":
                    # Eval method: greedy until ['\n', '.', ',']}
                    # In more detail:
                    # We sample from an LLM until EOT and then split on the above tokens (taking the 0th element from split method)
                    # That solution is then normalized (lowercased, stripped of whitespace, remove punctuation) and compared exactly against the answer
                    src_q = doc_eng["question"]
                    trg_q = doc_srp["question"]
                    src_a = str(doc_eng["answer"])
                    trg_a = str(doc_srp["answer"])
                    prompt = template.format(src_question=src_q, src_answer=src_a, trg_question=trg_q, trg_answer=trg_a)

                    num_attempts = NUM_ATTEMPTS_GPT4

                    while num_attempts > 0:
                        response = await instructor.gen_with_retry(
                            prompt, **api_params
                        )

                        if response is None:
                            num_attempts -= 1
                            continue

                        matches = re.findall(
                            r"REASONING:\s*(.*?)\s*SERBIAN:\s*\"question\":\s*(.*?)\n\s*\"answer\":\s*\[(.*?)\]\s*(?=REASONING:|$)",
                            response,
                            re.DOTALL
                        )

                        if len(matches) != 1:
                            print(f"Expected exactly one match, but found {matches}")
                            num_attempts -= 1
                            continue

                        reasoning, question, answer = matches[0]
                        pattern = r'\'(.*?[^\\])\'|"(.*?[^\\])"' # r'\'(.*?)\''
                        answers = [match[0] or match[1] for match in re.findall(pattern, answer)]

                        if len(answers) != len(doc_eng["answer"]) or not all(isinstance(a, str) for a in answers) or any(a == "" for a in answers):
                            print(f"Expected same number of answers, but found {len(answers)} and {len(doc_eng['answer'])}")
                            num_attempts -= 1
                            continue

                        doc_srp["question"] = question
                        doc_srp["answer"] = answers
                        break
                else:
                    raise RuntimeError("Unexpected task name")

                if num_attempts == 0:
                    raise RuntimeError("GPT-4 failed to generate a response")

                f.write(json.dumps(doc_srp, ensure_ascii=False) + "\n")  # write the translated doc to file
                f.flush()
                save_human_readable(f_human_readable, reasoning, doc_eng, old_doc_srp, doc_srp, doc_index)
                progress_bar.update(1)
    except Exception as e:
        # Rename output file to indicate that it is incomplete and add doc_id, after that raise again
        print(e)
        new_filename = f'{task_name}{"_test" if is_test else "_train"}_partial_{start_from_doc_index}_{doc_index-1}.jsonl'
        os.rename(out_path, os.path.join(out_dir, new_filename))

        new_filename_human_readable = f'{task_name}{"_test" if is_test else "_train"}_human_readable_partial_{start_from_doc_index}_{doc_index-1}.jsonl'
        os.rename(out_path_human_readable, os.path.join(out_dir, new_filename_human_readable))
        raise e

    new_filename = f'{task_name}{"_test" if is_test else "_train"}_partial_{start_from_doc_index}_{len(doc_eng) - 1}_end.jsonl'
    os.rename(out_path, os.path.join(out_dir, new_filename))

    new_filename_human_readable = f'{task_name}{"_test" if is_test else "_train"}_human_readable_partial_{start_from_doc_index}_{len(doc_eng) - 1}_end.jsonl'
    os.rename(out_path_human_readable, os.path.join(out_dir, new_filename_human_readable))


def get_serbian_docs(in_dir, task_name, is_test=False, get_only_file_name=False):
    suffix = "_test" if is_test else "_train"
    matches = [filename for filename in os.listdir(in_dir) if filename.startswith(f'{task_name}{suffix}')]
    assert len(matches) == 1, f"Expected exactly one match for {task_name}, but found {matches}"
    filename = matches[0]
    file_path = os.path.join(in_dir, filename)
    with open(file_path, 'r') as f:
        task_docs_serbian = []
        for line in f:
            task_docs_serbian.append(json.loads(line))

    if get_only_file_name:
        return filename
    else:
        return task_docs_serbian


async def refine_eval(task_dict_items, start_from_doc_index=None):

    this_file_path = os.path.dirname(os.path.realpath(__file__))
    parent_dir_path = os.path.abspath(os.path.join(this_file_path, os.pardir))
    out_dir = os.path.join(parent_dir_path, "serbian_eval", "refined")
    in_dir = os.path.join(parent_dir_path, "serbian_eval", "transliterated")
    prompt_templates_dir = os.path.join(parent_dir_path, "serbian_eval", "prompts")
    os.makedirs(out_dir, exist_ok=True)
    instructor = OpenAIHandler()

    try:
        for task_name, task in task_dict_items:
            print('*' * 50)
            print(f"Translating task: {task_name}")
            print('*' * 50)

            if task.has_test_docs():
                task_doc_func = task.test_docs
            elif task.has_validation_docs():
                task_doc_func = task.validation_docs
            else:
                raise RuntimeError("Task has neither test_docs nor validation_docs")

            # keys_of_interest = get_keys_of_interest(task_name)
            # first_level_keys = set([x.split('/')[0] for x in keys_of_interest])
            # second_level_keys = set([x.split('/')[1] for x in keys_of_interest if len(x.split('/')) > 1])

            task_docs_serbian = get_serbian_docs(in_dir, task_name, is_test=True)
            task_docs = list(task_doc_func())
            assert len(task_docs) == len(task_docs_serbian), f"Expected same number of docs, but found {len(task_docs)} and {len(task_docs_serbian)}"

            await refine_dataset(instructor, task_docs, task_docs_serbian, task_name, prompt_templates_dir, out_dir, in_dir, is_test=True, start_from_doc_index=start_from_doc_index)

            if task_name in ["nq_open", "triviaqa"]:
                task_docs_serbian = get_serbian_docs(in_dir, task_name, is_test=False)
                task_docs = list(task.training_docs())
                assert len(task_docs) == len(task_docs_serbian), f"Expected same number of docs, but found {len(task_docs)} and {len(task_docs_serbian)}"

                await refine_dataset(instructor, task_docs, task_docs_serbian, task_name, prompt_templates_dir, out_dir, in_dir, is_test=False, start_from_doc_index=start_from_doc_index)

    except Exception as e:
        print(e)
        exit(0)

    print(f'Ok, done!')


@positional_deprecated
def evaluate(
    lm,
    task_dict,
    provide_description=None,
    num_fewshot=0,
    limit=None,
    bootstrap_iters=100000,
    description_dict=None,
    decontamination_ngrams_path=None,
    write_out=False,
    output_base_path=None,
    translation_project_id=None,
    char_limit=500000,
    start_from_doc_index=None,
):
    """Instantiate and evaluate a model on a list of tasks.

    :param lm: obj
        Language Model
    :param task_dict: dict[str, Task]
        Dictionary of tasks. Tasks will be taken to have name task.EVAL_HARNESS_NAME if defined and type(task).__name__ otherwise.
    :param provide_description: bool
        Not implemented, and this option is deprecated and will be removed in a future version in favor of a different description providing method
    :param num_fewshot: int
        Number of examples in few-shot context
    :param limit: int, optional
        Limit the number of examples per task (only use this for testing)
    :param bootstrap_iters:
        Number of iterations for bootstrap statistics
    :param description_dict: dict[str, str]
        Dictionary of custom task descriptions of the form: `task_name: description`
    :param write_out: bool
        If True, write all prompts, logits and metrics to json for offline analysis
    :param output_base_path: str, optional
        Directory to which detailed eval info will be written. Defaults to present working dir
    :return
        Dictionary of results
    """
    # TODO: completely refactor this entire function to not be a huge mess, ideally breaking it down into smaller pieces

    # TODO: todo: implement proper description-providing system
    assert not provide_description  # not implemented.
    if provide_description is not None:
        # nudge people to not specify it at all
        print(
            "WARNING: provide_description is deprecated and will be removed in a future version in favor of description_dict"
        )

    decontaminate = decontamination_ngrams_path is not None

    task_dict_items = [
        (name, task)
        for name, task in task_dict.items()
        if (task.has_validation_docs() or task.has_test_docs())
    ]

    # translate_eval(task_dict_items, project_id=translation_project_id, char_limit=char_limit, start_from_doc_index=start_from_doc_index)
    asyncio.run(refine_eval(task_dict_items, start_from_doc_index=start_from_doc_index))
    exit(0)

    results = collections.defaultdict(dict)
    versions = collections.defaultdict(dict)

    requests = collections.defaultdict(list)
    requests_origin = collections.defaultdict(list)

    overlaps = collections.defaultdict(list)  # {task_name: contaminated_docs}

    # If we ever run into issues where the eval tasks don't fit in memory and we can't afford a machine with bigger
    # memory, we can always modify this plumbing to support that, but I didn't want to include it just yet because
    # over-engineering is bad (or we could make it write the requests to disk and then read them back out again
    #  - probably using an sqlite db because of all the moving parts we have

    # TODO: we need unit tests & sanity checks or something to ensure that the return of `validation_docs` is stable
    docs = {}
    write_out_info = {}

    docs_for_decontamination = collections.defaultdict(list)

    # get lists of each type of request
    for task_name, task in task_dict_items:
        versions[task_name] = task.VERSION
        # default to test doc, fall back to val doc if validation unavailable
        # TODO: the test-fallback-to-val system isn't final, we should revisit it at some point
        if task.has_test_docs():
            task_doc_func = task.test_docs
            task_set = "test"  # Required for caching in the decontamination
        elif task.has_validation_docs():
            task_set = "val"  # Required for caching in the decontamination
            task_doc_func = task.validation_docs
        else:
            raise RuntimeError("Task has neither test_docs nor validation_docs")

        # deterministically shuffle docs and chop off the first `limit` because sometimes docs are in some kind of order
        task_docs = list(task_doc_func())
        rnd = random.Random()
        rnd.seed(42)
        rnd.shuffle(task_docs)
        print(f"Task: {task_name}; number of docs: {len(task_docs)}")

        if write_out:
            prompt_details = []

        description = (
            description_dict[task_name]
            if description_dict and task_name in description_dict
            else ""
        )
        if limit is not None:
            limit = int(len(task_docs) * limit) if limit < 1.0 else int(limit)

        for doc_id, doc in enumerate(itertools.islice(task_docs, 0, limit)):
            if decontaminate and task.should_decontaminate():
                docs_for_decontamination[(task_name, task_set)].append(
                    task.doc_to_decontamination_query(doc)
                )

            docs[(task_name, doc_id)] = doc
            ctx = task.fewshot_context(
                doc=doc, num_fewshot=num_fewshot, rnd=rnd, description=description
            )
            reqs = task.construct_requests(doc, ctx)

            if write_out:
                prompt_details.append({"doc_id": doc_id})

            # print the prompt for the first few documents
            if doc_id < 1:
                print(
                    f"Task: {task_name}; document {doc_id}; context prompt (starting on next line):\n{ctx}\n(end of prompt on previous line)"
                )
                print("Requests:", reqs)

            if not isinstance(reqs, (list, tuple)):
                reqs = [reqs]
            for i, req in enumerate(reqs):
                requests[req.request_type].append(req)
                # i: index in requests for a single task instance
                # doc_id: unique id that we can get back to a doc using `docs`
                requests_origin[req.request_type].append((i, task_name, doc, doc_id))

                if write_out:
                    prompt_details[-1][f"prompt_{i}"] = "".join(
                        (map(lambda x: "".join(x), req.args))
                    )

        if write_out:
            write_out_info[task_name] = prompt_details

    # Compare all tasks/sets at once to ensure a single training set scan
    if decontaminate:
        from lm_eval.decontamination.decontaminate import get_train_overlap

        print("Finding train/test overlap, please wait...")
        overlaps = get_train_overlap(
            docs_for_decontamination, decontamination_ngrams_path, limit
        )

    # all responses for each (task, doc)
    process_res_queue = collections.defaultdict(list)

    # execute each type of request
    for reqtype, reqs in requests.items():
        # TODO: right now, this code runs multiple separate LM requests for multiple Requests differing
        #       only in index. We could implement some kind of caching, but that would be more of a band-aid
        #       solution. we could also implement some kind of auto-grouping here;
        #       they should end up next to each other.

        print("Running", reqtype, "requests")
        resps = getattr(lm, reqtype)([req.args for req in reqs])
        resps = [
            x if req.index is None else x[req.index] for x, req in zip(resps, reqs)
        ]

        for resp, (i, task_name, doc, doc_id) in zip(resps, requests_origin[reqtype]):
            process_res_queue[(task_name, doc_id)].append((i, resp))

            if write_out:
                write_out_info[task_name][doc_id][f"logit_{i}"] = resp
                task = task_dict[task_name]
                if isinstance(task, lm_eval.base.MultipleChoiceTask):
                    write_out_info[task_name][doc_id]["truth"] = doc["gold"]
                elif isinstance(task, lm_eval.tasks.winogrande.Winogrande):
                    write_out_info[task_name][doc_id]["truth"] = task.answer_to_num[
                        doc["answer"]
                    ]
                else:
                    write_out_info[task_name][doc_id]["truth"] = task.doc_to_target(doc)

    vals = collections.defaultdict(list)

    # unpack results and sort back in order and return control to Task
    for (task_name, doc_id), requests in process_res_queue.items():
        requests.sort(key=lambda x: x[0])
        requests = [x[1] for x in requests]

        task = task_dict[task_name]
        doc = docs[(task_name, doc_id)]

        metrics = task.process_results(doc, requests)
        for metric, value in metrics.items():
            vals[(task_name, metric)].append(value)

            if write_out:
                write_out_info[task_name][doc_id][metric] = str(value)

            # Re-use the evaluation for the decontaminated set by just ignoring the overlaps
            if decontaminate and task_name in overlaps:
                if doc_id not in overlaps[task_name]:
                    vals[(task_name, metric + decontaminate_suffix)].append(value)

    # aggregate results
    for (task_name, metric), items in vals.items():
        task = task_dict[task_name]
        real_metric = metric  # key when looking up the metric with task.aggregation
        if metric.endswith(decontaminate_suffix):
            real_metric = metric.replace(
                decontaminate_suffix, ""
            )  # decontaminated still uses the same metric
        results[task_name][metric] = task.aggregation()[real_metric](items)

        # hotfix: bleu, chrf, ter seem to be really expensive to bootstrap
        # so we run them less iterations. still looking for a cleaner way to do this

        stderr = lm_eval.metrics.stderr_for_metric(
            metric=task.aggregation()[real_metric],
            bootstrap_iters=min(bootstrap_iters, 1000)
            if metric in ["bleu", "chrf", "ter"]
            else bootstrap_iters,
        )

        if stderr is not None:
            results[task_name][metric + "_stderr"] = stderr(items)

    if write_out:
        import json
        import pathlib

        output_base_path = (
            pathlib.Path(output_base_path)
            if output_base_path is not None
            else pathlib.Path(".")
        )
        try:
            output_base_path.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            pass

        for task_name, _ in task_dict_items:
            with open(
                output_base_path.joinpath(f"{task_name}_write_out_info.json"),
                "w",
                encoding="utf8",
            ) as fp:
                json.dump(write_out_info[task_name], fp, indent=4, ensure_ascii=False)

    return {"results": dict(results), "versions": dict(versions)}


def make_table(result_dict):
    """Generate table of results."""
    from pytablewriter import MarkdownTableWriter, LatexTableWriter

    md_writer = MarkdownTableWriter()
    latex_writer = LatexTableWriter()
    md_writer.headers = ["Task", "Version", "Metric", "Value", "", "Stderr"]
    latex_writer.headers = ["Task", "Version", "Metric", "Value", "", "Stderr"]

    values = []

    for k, dic in result_dict["results"].items():
        version = result_dict["versions"][k]
        for m, v in dic.items():
            if m.endswith("_stderr"):
                continue

            if m + "_stderr" in dic:
                se = dic[m + "_stderr"]
                values.append([k, version, m, "%.4f" % v, "±", "%.4f" % se])
            else:
                values.append([k, version, m, "%.4f" % v, "", ""])
            k = ""
            version = ""
    md_writer.value_matrix = values
    latex_writer.value_matrix = values

    # todo: make latex table look good
    # print(latex_writer.dumps())

    return md_writer.dumps()
