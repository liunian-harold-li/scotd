# use py as the config file because yaml throws a tone of errors when parsing strings
cfg = {
  "TYPE": ("commonsense", ),
  "SOURCE": "hf",
  "CLEAN_STRATEGY": "answer_split",
  "HF_IDENTIFIER": "sst2",
  "HF_NAME": "default",
  "FEW_SHOT_PROMPT": (),

  "USE_PROMPT_SOURCE": False,

  "MODULE_NAME": "configs_data.complete.sst2",
  "CLASS_NAME": "SST2",
  "INPUT_PREFIX": "Q: What is the sentiment of the following sentence?\n",

#   "TRAIN": "train_r3",
    "VALID": "validation",
#   "TEST": "test_r3",

  "FEW_SHOT_COT_PROMPT": (
    '"that loves its characters and communicates something rather beautiful about human nature"',
    '"loves its characters" indicates positive sentiment.',
    'positive',

    '"hide new secretions from the parental units"',
    'If people are hiding something, it means the sentiment is on the negative side.',
    'negative',

    '"the greatest musicians"',
    'By saying someone being the "greatest", it means positive sentiment.',
    'positive',
    
    '"contains no wit , only labored gags"',
    '"contains no wit" is clearly a negative sentiment.',
    'negative',

    '"demonstrates that the director of such hollywood blockbusters as patriot games can still turn out a small , personal film with an emotional wallop ."',
    '"can still turn out a small , personal film with an emotional wallop ." indicates sentiment on the positive side.',
    'positive',

    '"that \'s far too tragic to merit such superficial treatment"',
    '"far too tragic" and "to merit such superficial treatment" both mean negative sentiments.',
    'negative',
  ),


  "FEW_SHOT_PROMPT_INDEX": (1, ),

  "FEW_SHOT_PROMPT_ORIGINAL": (1, ) # place holder

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

class SST2(CoTDataset):
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
        question = '"{}"'.format(example["sentence"])
        if example["label"] == 0:
            answer_word = "negative"
        elif example["label"] == 1:
            answer_word = "positive"
        target = answer_word
        return question, target, None
    
    def convert_original_to_examples(self, original_prompts):
        examples = [
          {'sentence': 'that loves its characters and communicates something rather beautiful about human nature', 'label': 1},
            {'sentence': 'hide new secretions from the parental units', 'label': 0},
            {'sentence': 'the greatest musicians', 'label': 1},
            {'sentence': 'contains no wit , only labored gags', 'label': 0},
            {'sentence': 'demonstrates that the director of such hollywood blockbusters as patriot games can still turn out a small , personal film with an emotional wallop .', 'label': 1},
            {'sentence': 'that \'s far too tragic to merit such superficial treatment', 'label': 0},
        ]
        return examples
