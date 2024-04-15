# use py as the config file because yaml throws a tone of errors when parsing strings
cfg = {
  "TYPE": ("commonsense", ),
  "SOURCE": "hf",
  "CLEAN_STRATEGY": "answer_split",
  "HF_IDENTIFIER": "boolq",
  "HF_NAME": "default",
  "FEW_SHOT_PROMPT": (),

  "USE_PROMPT_SOURCE": False,

  "MODULE_NAME": "configs_data.complete.boolq_context",
  "CLASS_NAME": "BOOLQ",
  "INPUT_PREFIX": "Read the passage and answer a question. ",

  "RANDOM_DROP_COT_PROMPTS_MAX_NUM": 1,
#   "TRAIN": "train_r3",
    "VALID": "validation",
#   "TEST": "test_r3",

  "FEW_SHOT_COT_PROMPT": (
    "System of a Down, sometimes shortened to System and abbreviated as SOAD, is an Armenian-American heavy metal band from Glendale, California, formed in 1994. The band currently consists of Serj Tankian (lead vocals, keyboards), Daron Malakian (vocals, guitar), Shavo Odadjian (bass, backing vocals) and John Dolmayan (drums).\nBased on the above text, does system of a down have 2 singers?",
    "System of a Down currently consists of Serj Tankian, Daron Malakian, Shavo Odadjian and John Dolmayan. Serj and Daron do vocals, so the band does have two singers.",
    "yes",

    "Persian, also known by its endonym Farsi, is one of the Western Iranian languages within the Indo-Iranian branch of the Indo-European language family. It is primarily spoken in Iran, Afghanistan, and Tajikistan, and some other regions which historically were Persianate societies and considered part of Greater Iran.\nBased on the above text, do iran and afghanistan speak the same language?",
    "Iran and Afghanistan both speak the Indo-European language Persian.",
    "yes",

    "Both the violin and viola are played under the jaw. The viola, being the larger of the two instruments, has a playing range that reaches a perfect fifth below the violin’s. The cello is played sitting down with the instrument between the knees, and its playing range reaches an octave below the viola’s. The double bass is played standing or sitting on a stool, with a range that typically reaches a minor sixth, an octave or a ninth below the cello’s.\nBased on the above text, is a cello and a bass the same thing?",
    "The cello is played sitting down with the instrument between the knees, whereas the double bass is played standing or sitting on a stool.",
    "no",

    "Epsom railway station serves the town of Epsom in Surrey. It is located off Waterloo Road and is less than two minutes’ walk from the High Street. It is not in the London Oyster card zone unlike Epsom Downs or Tattenham Corner stations. The station building was replaced in 2012/2013 with a new building with apartments above the station.\nBased on the above text, can you use oyster card at epsom station?",
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
        return "{}\nBased on the above text, {}".format(datapoint["passage"], datapoint["question"]), label[datapoint["answer"]], None