# use py as the config file because yaml throws a tone of errors when parsing strings
cfg = {
  "TYPE": ("commonsense", ),
  "SOURCE": "hf",
  "CLEAN_STRATEGY": "answer_split",
  "HF_IDENTIFIER": "multi_nli",
  "HF_NAME": "default",
  "FEW_SHOT_PROMPT": (),

  "USE_PROMPT_SOURCE": False,

  "MODULE_NAME": "configs_data.complete.mnli",
  "CLASS_NAME": "MNLI",
  "INPUT_PREFIX": "",

  "TRAIN": "train",
  "VALID": "validation_matched",
  "TEST": "validation_matched",

  "FEW_SHOT_COT_PROMPT": (
    'Premise:\n"Conceptually cream skimming has two basic dimensions - product and geography."\nBased on this premise, can we conclude the hypothesis "Product and geography are what make cream skimming work." is true?\nOPTIONS:\n- yes\n- no\n- it is not possible to tell',
    'Based on "cream skimming has two basic dimensions" we can’t infer that these two dimensions are what make cream skimming work.',
    'it is not possible to tell.',


    'Premise:\n"One of our member will carry out your instructions minutely."\nBased on this premise, can we conclude the hypothesis "A member of my team will execute your orders with immense precision." is true?\nOPTIONS:\n- yes\n- no\n- it is not possible to tell',
    'one of" means the same as "a member of", "carry out" means the same as "execute", and "minutely" means the same as "immense precision".',
    'yes.',


    'Premise:\n"Fun for adults and children."\nBased on this premise, can we conclude the hypothesis "Fun for only children." is true?\nOPTIONS:\n- yes\n- no\n- it is not possible to tell',
    '"adults and children" contradicts "only children".',
    'no.',

    'Premise:\n"He turned and smiled at Vrenna."\nBased on this premise, can we conclude the hypothesis "He smiled at Vrenna who was walking slowly behind him with her mother." is true?\nOPTIONS:\n- yes\n- no\n- it is not possible to tell',
    'the premise does not say anything about "Vrenna was walking".',
    'it is not possible to tell.',

    'Premise:\n"well you see that on television also"\nBased on this premise, can we conclude the hypothesis "You can see that on television, as well." is true?\nOPTIONS:\n- yes\n- no\n- it is not possible to tell',
    '"also" and "as well" mean the same thing.',
    'yes.',


    'Premise:\n"Vrenna and I both fought him and he nearly took us."\nBased on this premise, can we conclude the hypothesis "Neither Vrenna nor myself have ever fought him." is true?\nOPTIONS:\n- yes\n- no\n- it is not possible to tell',
    '"Vrenna and I both" contradicts "neither Vrenna nor myself".', 
    'no.',
  )
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

class MNLI(CoTDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset = load_dataset(path=self.dataset_cfg.HF_IDENTIFIER, name=self.dataset_cfg.HF_NAME, split=self.split_name_str)
        new_dataset = []
        for i in range(len(self.dataset)):
            example = self.dataset[i]
            new_dataset.append(self.convert_example(example))
        self.dataset = new_dataset
        self.limit_size()
    
    def convert_example(self, example):
        # could have multiple ways 
        question = 'Premise:\n"{}"\nBased on this premise, can we conclude the hypothesis "{}" is true?\nOPTIONS:\n- yes\n- no\n- it is not possible to tell'.format(example["premise"], example["hypothesis"])
        if example["label"] == 0:
            answer_word = "yes."
        elif example["label"] == 1:
            answer_word = "no."
        else:
            answer_word = "it is not possible to tell."
        target = answer_word
        return question, target, None