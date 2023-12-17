"""
PIQA: Reasoning about Physical Commonsense in Natural Language
https://arxiv.org/pdf/1911.11641.pdf

Physical Interaction: Question Answering (PIQA) is a physical commonsense
reasoning and a corresponding benchmark dataset. PIQA was designed to investigate
the physical knowledge of existing models. To what extent are current approaches
actually learning about the world?

Homepage: https://yonatanbisk.com/piqa/
"""
from lm_eval.base import MultipleChoiceTask


_CITATION = """
@inproceedings{Bisk2020,
    author = {Yonatan Bisk and Rowan Zellers and
            Ronan Le Bras and Jianfeng Gao
            and Yejin Choi},
    title = {PIQA: Reasoning about Physical Commonsense in
           Natural Language},
    booktitle = {Thirty-Fourth AAAI Conference on
               Artificial Intelligence},
    year = {2020},
}
"""


class PiQA(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "piqa"
    DATASET_NAME = None

    def __init__(self, **kwargs):
        language = kwargs.get("language", "English")
        self._language = language
        if language == "Serbian":
            self.DATASET_PATH = "gordicaleksa/serbian-llm-eval-v1"
            self.DATASET_NAME = "piqa"
        super().__init__(**kwargs)

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(map(self._process_doc, self.dataset["train"]))
        return self._training_docs

    def validation_docs(self):
        return map(self._process_doc, self.dataset["test"] if self._language == "Serbian" else self.dataset["validation"])

    def _process_doc(self, doc):
        if self._language == "Serbian":
            return {
                "goal": doc["goal"],
                "choices": doc["choices"],
                "gold": doc["gold"],
            }
        else:
            out_doc = {
                "goal": doc["goal"],
                "choices": [doc["sol1"], doc["sol2"]],
                "gold": doc["label"],
            }
            return out_doc

    def doc_to_text(self, doc):
        if self._language == "Serbian":
            return "Pitanje: " + doc["goal"] + "\nOdgovor:"
        else:
            return "Question: " + doc["goal"] + "\nAnswer:"

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["goal"]
