# use py as the config file because yaml throws a tone of errors when parsing strings
cfg = {
  "TYPE": ("commonsense", ),
  "SOURCE": "hf",
  "CLEAN_STRATEGY": "answer_split",
  "HF_IDENTIFIER": "boolq",
  "HF_NAME": "default",
  "FEW_SHOT_PROMPT": (),

  "USE_PROMPT_SOURCE": False,

  "MODULE_NAME": "configs_data.complete.boolq",
  "CLASS_NAME": "BOOLQ",
  #"INPUT_PREFIX": "Read the following sentence and choose the sentence that will most likely happen next.\n",

#   "TRAIN": "train_r3",
    "VALID": "validation",
#   "TEST": "test_r3",

    "RANDOM_DROP_COT_PROMPTS_MAX_NUM": 2,

  "FEW_SHOT_COT_PROMPT": (
    "does system of a down have 2 singers?",
    "System of a Down currently consists of Serj Tankian, Daron Malakian, Shavo Odadjian and John Dolmayan. Serj and Daron do vocals, so the band does have two singers.",
    "yes",

    "do iran and afghanistan speak the same language?",
    "Iran and Afghanistan both speak the Indo-European language Persian.",
    "yes",

    "is a cello and a bass the same thing?",
    "The cello is played sitting down with the instrument between the knees, whereas the double bass is played standing or sitting on a stool.",
    "no",

    "can you use oyster card at epsom station?",
    "Epsom railway station serves the town of Epsom in Surrey and is not in the London Oyster card zone.",
    "no",

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

label = {
            0: "no",
            1: "yes",
        }
class BOOLQ(CoTDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.dataset = load_dataset(path=self.dataset_cfg.HF_IDENTIFIER, name=self.dataset_cfg.HF_NAME, split=self.split_name_str)
        new_dataset = []
        for i in range(len(self.dataset)):
            example = self.dataset[i]
            new_dataset.append(self.convert_example(example))
        self.dataset = new_dataset
        self.limit_size()
      
    def convert_example(self, datapoint):
        return datapoint["question"], label[datapoint["answer"]], None