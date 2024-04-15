# use py as the config file because yaml throws a tone of errors when parsing strings
cfg = {
  "TYPE": ("commonsense", ),
  "SOURCE": "hf",
  "CLEAN_STRATEGY": "answer_split",
  "HF_IDENTIFIER": "super_glue",
  "HF_NAME": "wic",
  "FEW_SHOT_PROMPT": (),

  "USE_PROMPT_SOURCE": False,

  "MODULE_NAME": "configs_data.complete.wic",
  "CLASS_NAME": "WIC",
  
  "INPUT_PREFIX": "Sentences: ",

#   "TRAIN": "train_r3",
  "VALID": "validation",
  "TEST": "validation",

  "FEW_SHOT_COT_PROMPT": (

    'Do you want to come over to my place later?\nA political system with no place for the less prominent groups.\nQ: Is the word "place" used in the same way in the two sentences above?',
    'The first "place" means "home", the second "place" means "room".',
    'no',


    'Approach a task.\nTo approach the city.\nQ: Is the word "approach" used in the same way in the two sentences above?',
    'The first "approach" means "deal with", the second "approach" means "come near".',
    'no',


    'The general ordered the colonel to hold his position at all costs.\nHold the taxi.\nQ: Is the word "hold" used in the same way in the two sentences above?',
    'Both "hold" mean "keep" or "detain".',
    "yes",

    'We like to summer in the Mediterranean.\nWe summered in Kashmir.\nQ: Is the word "summer" used in the same way in the two sentences above?',
    'Both "summer" mean "spend the summer".',
    'yes',
      

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

class WIC(CoTDataset):
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
        question ="{}\n{}\nQ: Is the word \"{}\" used in the same way in the two sentences above?".format(example["sentence1"], example["sentence2"], example["word"])
        target = 'no' if example["label"] == 0 else 'yes'
        return question, target, None