# Progress

Done: `arc_easy`, `arc_challenge`, `openbookqa`, `winogrande`, `piqa`, `boolq`, `hellaswag`

In progress: `triviaqa`, `nq_open` (see in open-hbs-llm channel on Discord)

todo: <empty> :)

(before setting a task sync with Aleksa first, see the rest of the README)

# IMPORTANT

* running this this will eat your google cloud credits or will bill you if you're already in a bill mode (this happens after you spend free credits and then deliberately enable billing again).

* you can use your free credits to translate 500.000 chars / month!

* if this is the first time you're creating a gcloud project you'll have 300$ of free credits!

* only translate a subset of the above tasks, sync with Aleksa in [Discord](https://discord.gg/peBrCpheKE) in open-hbs-llm channel.

# Prerequisites

Before you begin, ensure you meet the following requirements:

**For Linux Users:**

- [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) and [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/)

**For Windows Users:**
1. Windows Subsystem for Linux [(WSL2)](https://learn.microsoft.com/en-us/windows/wsl/install). If you don't have WSL2 installed, follow these steps in Windows cmd/powershell in administrator mode:

    ```bash
    wsl --install

   // Check version and distribution name.
   wsl -l -v       

    // Set the newly downloaded linux distro as default.
    wsl --set-default <distribution name>
    ```
2. Install [Git](https://learn.microsoft.com/en-us/windows/wsl/tutorials/wsl-git) from the WSL terminal.

    ```bash
    sudo apt update
    sudo apt install git
    git --version
    ```
3. Install [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) from the WSL terminal.
    ```bash
    mkdir -p ~/miniconda3

    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh

    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3

    rm -rf ~/miniconda3/miniconda.sh

    // Initialize conda with bash.
    ~/miniconda3/bin/conda init bash
    ```

4. Follow the instructions below on WSL.

# Instructions for translating lm harness eval from English into Serbian

First let's setup a minimal Python program that makes sure you can run Google Translate on your local machine.

1. Create a Google Console project (https://console.cloud.google.com/)
2. Enable Google Translation API -> to enable it you have to setup the billing and input your credit card details (a note regarding safety: you'll have 300$ of free credit (if this is the first time you're doing it) and no one can spend money from your credit card unless all those free credits are spent and you re-enable the billing again! if you already had it setup in that case you have 500.000 chars/month for free!)
3. Install Google Cloud CLI (gsutil) on your machine (see this: https://cloud.google.com/storage/docs/gsutil_install/)
    
    a.) Download the Linux archive file (find latest version from link above)

     `curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-455.0.0-linux-x86_64.tar.gz`

    b.) Extract the contents from the archive file above.

    `tar -xf google-cloud-cli-455.0.0-linux-x86_64.tar.gz`

    c.) Run installation script. `./google-cloud-sdk/install.sh`

    d.) Initiate and authenticate your account. `./google-cloud-sdk/bin/gcloud init`
    
    e.) Create a credentials file with `gcloud auth application-default login`
4. Create and setting up the conda env
    
    a.) Open a terminal (if on Windows use the `WSL` terminal, if you're on Linux just use your terminal conda will already be in the PATH)
    
    b.) Run `conda create -n open_nllb python=3.10 -y`

    c.) Run `conda activate open_nllb` 
    
    d.) Run  `pip install google-cloud-translate`

That's it! After that just create a `test.py` Python file with the following code and run with `Run and Debug` option in VS code after creating the launch.json file:

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
3. Change `translation_project_id` to the google project id you got in the previous section
4. Specify amount of characters you're willing to translate (500_000 is the usual free monthly limit)
5. Run the `main.py`

### Create Python environment

You can reuse the above conda env `open-nllb`, additionally do the following:

```
git clone https://github.com/gordicaleksa/lm-evaluation-harness-serbian
cd lm-evaluation-harness-serbian
pip install -e .
```

### Run translation

Finally run (note `model` and `model_args` are not important for us but we need to specify them):

```
python main.py \
    --model hf \
    --model_args pretrained=mistralai/Mistral-7B-v0.1 \
    --tasks hellaswag \
    --translation_project_id <your project id>
    --char_limit 500000
    --start_from_doc_index 0
```

or open `main.py` and run using vscode debugger.

Note:
* again please sync on Discord about which tasks you should help to translate! :)
* select only one task at a time, posssible options: `hellaswag,winogrande,piqa,openbookqa,arc_easy,arc_challenge,nq_open,triviaqa,boolq`
* `start_from_doc_index` is used if you want to resume and translate a particular task only starting from a certain document index (useful in a collaborative setting where multiple people are translating different portions of the task)
