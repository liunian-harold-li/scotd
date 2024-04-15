# use py as the config file because yaml throws a tone of errors when parsing strings
cfg = {
  "TYPE": ("commonsense", ),
  "SOURCE": "hf",
  "CLEAN_STRATEGY": "answer_split",
  "HF_IDENTIFIER": "quarel",
  "HF_NAME": "default",
  "FEW_SHOT_PROMPT": (),

  "USE_PROMPT_SOURCE": False,

  "MODULE_NAME": "configs_data.complete.quarel",
  "CLASS_NAME": "HFDataset",
  "INPUT_PREFIX": "Q: ",

#   "TRAIN": "train_r3",
    "VALID": "validation",
#   "TEST": "test_r3",

  "FEW_SHOT_COT_PROMPT": (
    ' ',
    ' ',
    ' ',

  ),

  "FEW_SHOT_PROMPT_INDEX": ()
}

from dataset.hf_datasets import CoTDataset, chunks
from torch import nn
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import pdb
from datasets import load_dataset
from configs_data._dataset_config import _C as DATASET_CFG
from tqdm import tqdm
import json
import random
from pprint import pprint
from promptsource.templates import DatasetTemplates

class HFDataset(CoTDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.dataset = load_dataset(path=self.dataset_cfg.HF_IDENTIFIER, name=self.dataset_cfg.HF_NAME, split=self.split_name_str)
        new_dataset = []
        for i in range(len(self.dataset)):
            example = self.dataset[i]
            new_dataset.append(self.convert_example(example))
        self.dataset = new_dataset
        pdb.set_trace()
        self.limit_size()
    
    def get_answer_string(self, datapoint):
        answer_index = datapoint["answer_index"]
        # st1 = datapoint["question"].find("(A)")
        # st2 = datapoint["question"].find("(B)")

        # if answer_index == 0:
        #     answer_string = datapoint["question"][st1+4: st2]
        # else:
        #     answer_string = datapoint["question"][st2+4: ]

        # if answer_string.endswith("or "):
        #     answer_string = answer_string[:-3]
        if answer_index == 0:
            answer_string = "(A)"
        else:
            answer_string = "(B)"

        return answer_string

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            answer_string = self.get_answer_string(datapoint)
            lines.append((datapoint["question"], answer_string.strip()))
        return lines

    def convert_example(self, datapoint):
        answer_string = self.get_answer_string(datapoint)
        return datapoint["question"], answer_string.strip(), None