# IMPORTANT

* running this this will eat your google cloud credits or will bill you if you're already in a bill mode (this happens after you spend free credits and then deliberately enable billing again).

* you can use your free credits to translate 500.000 chars / month!

* if this is the first time you're creating a gcloud project you'll have 300$ of free credits!

* only translate a subset of the above tasks, sync with Aleksa in [Discord](https://discord.gg/peBrCpheKE) in open-hbs-llm.

# Instructions for translating lm harness eval from English into Serbian

First let's setup a minimal Python program that makes sure you can run Google Translate on your local machine.

1. create a google console project (https://console.cloud.google.com/)
2. enable google translation API -> to enable it you have to setup the billing and input your credit card details (a note regarding safety: you'll have 300$ of free credit (if this is the first time you're doing it) and no one can spend money from your credit card unless all those free credits are spent and you re-enable the billing again! if you already had it setup in that case you have 500.000 chars/month for free!)
3. install `gcloud cli`  on your machine (see this: https://cloud.google.com/storage/docs/gsutil_install/)
4. create conda env - if you don't have conda install miniconda, see this: https://docs.conda.io/projects/miniconda/en/latest/
5. open a terminal (if on Windows type in `anaconda` in your search bar don't use `cmd`, if you're on Linux just use your terminal conda will already be in the PATH)
6. run `conda create -n open_nllb python=3.10 -y`
7. run `conda activate open_nllb` and then run  `pip install google-cloud-translate`

That's it! After that just create a Python file with the following code and run it:

```Python
from google.cloud import translate

client = translate.TranslationServiceClient()
location = "global"
project_id="<your project id from above>"
parent = f"projects/{project_id}/locations/{location}"

response = client.translate_text(
    request={
        "parent": parent,
        "contents": ["How do you do? Translate this."],
        "mime_type": "text/plain",
        "source_language_code": "en-US",
        "target_language_code": "sr",
    }
)
value_translated = response.translations[0].translated_text
print(value_translated)
```

# Running translation of evals from English into Serbian

Follow these instructions (see below for more details):
1. Create a Python env for this project
2. You'll find the program arguments are already specified inside `.vscode/launch.json`
3. Change `project_id` to the google project id you got in the previous section
4. Specify amount of characters you're willing to translate (500_000 is the usual free monthly limit)
5. Run the `main.py`

### Create Python environment

You can reuse the above conda env `open-nllb`, additionally do the following:

```
git clone https://github.com/gordicaleksa/lm-evaluation-harness-serbian
cd lm-evaluation-harness
pip install -e .
```

### Run translation

Finally run (note `model` and `model_args` are not important for us but we need to specify them):

```
python main.py \
    --model hf \
    --model_args pretrained=mistralai/Mistral-7B-v0.1 \
    --tasks hellaswag,winogrande,piqa,openbookqa,arc_easy,arc_challenge,nq_open,triviaqa,boolq \
    --translation_project_id <your project id>
    --char_limit 500000
```

or open `main.py` and run using vscode debugger.


