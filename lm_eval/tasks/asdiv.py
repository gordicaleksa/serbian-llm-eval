"""
ASDiv: A Diverse Corpus for Evaluating and Developing English Math Word Problem Solvers
https://arxiv.org/abs/2106.15772

@misc{miao2021diverse,
      title={A Diverse Corpus for Evaluating and Developing English Math Word Problem Solvers},
      author={Shen-Yun Miao and Chao-Chun Liang and Keh-Yih Su},
      year={2021},
      eprint={2106.15772},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
"""
from lm_eval.base import Task
from pathlib import Path
from best_download import download_file 
import xml.etree.ElementTree as ET
from lm_eval.base import rf
from lm_eval.metrics import mean,perplexity
import numpy as np
from zipfile import ZipFile
import os 

#currently ignoring formula for answer generation

# given a subset, splits return the docs 
class Asdiv(Task):
    VERSION = 0
    DATASET_PATH = Path("data/asdiv")

    def download(self):
        if self.DATASET_PATH.exists():
            return
        Path.mkdir(self.DATASET_PATH)
        url = "https://github.com/chaochun/nlu-asdiv-dataset/archive/refs/heads/master.zip"
        checksum = "2f71f8003929d605369ad924be4b95c15879fc2bfac0d4d01a81f8aabceaad5c"
        zip_path = self.DATASET_PATH / "master.zip"
        download_file(url, str(zip_path), checksum)
        with ZipFile(zip_path, "r") as zip:
            zip.extractall(self.DATASET_PATH)
        os.remove(zip_path)

    def _convert_standard(self, problem):
        #TODO: include solution-type and formula
        out_doc = {
            "question" : problem.find('Question').text,
            "body" : problem.find('Body').text,
            "answer": problem.find('Answer').text
        }
        return out_doc

    def load_docs(self, textfilename, tfds=False):
        tree = ET.parse(textfilename)
        root = tree.getroot()
        for pid, problem in enumerate(root.iter('Problem')):
            out_doc = self._convert_standard(problem)
            yield out_doc

    def _strip_bracket(self,test_str):
        ret = ''
        skip1c = 0
        skip2c = 0
        for i in test_str:
            if i == '(':
                skip2c += 1
            elif i == ')'and skip2c > 0:
                skip2c -= 1
            elif skip1c == 0 and skip2c == 0:
                ret += i
        return ret

    def has_training_docs(self):
        return False
    
    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        raise NotImplementedError("This dataset has no training docs")

    def test_docs(self):
        raise NotImplementedError("This dataset has no test docs")

    def validation_docs(self):
        data_xml_path = self.DATASET_PATH / "nlu-asdiv-dataset-master/dataset/ASDiv.xml"
        return self.load_docs(data_xml_path)

    def fewshot_context(self, doc, num_fewshot, provide_description, rnd):
        assert num_fewshot == 0, "ASDiv is intended only for the zero-shot setting."
        return super().fewshot_context(doc, num_fewshot, provide_description, rnd)

    
    def fewshot_description(self):
        # TODO: add solution-type and formula
        desc = "information containing the context of the question\nQuestion: Text of a question.\nAnswer: Answer to the question, based on the passage.\n"
        return desc

    def doc_to_text(self, doc):
        # TODO: add solution-type
        return doc['body'] + '\n' + 'Question:' + doc['question'] + '\n' + 'Answer:'

    def doc_to_target(self, doc):
        # TODO: add formula

        answer = self._strip_bracket(doc['answer'])
        if len(answer)>0: # check if answer is present only in brackets
            return answer
        else:
            return doc['answer']

    def construct_requests(self, doc, ctx):
        ll, is_greedy = rf.loglikelihood(ctx, self.doc_to_target(doc))

        return ll, is_greedy
    
    def process_results(self, doc, results):
        ll, is_greedy = results

        return {
            'acc': int(is_greedy)
        }
        
    def aggregation(self):
        return {
            'acc': mean
        }

    def higher_is_better(self):
        return {
            'acc': True
        }

