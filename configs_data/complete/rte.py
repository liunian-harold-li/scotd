# use py as the config file because yaml throws a tone of errors when parsing strings
cfg = {
  "TYPE": ("commonsense", ),
  "SOURCE": "hf",
  "CLEAN_STRATEGY": "answer_split",
  "HF_IDENTIFIER": "glue",
  "HF_NAME": "rte",
  "FEW_SHOT_PROMPT": (),

  "USE_PROMPT_SOURCE": False,

  "MODULE_NAME": "configs_data.complete.rte",
  "CLASS_NAME": "RTE",
  "INPUT_PREFIX": "",
  "STEP_PREFIX": "\nA: ", # this is the prompt for the answer
  "ANSWER_PREFIX": " The answer is ",
  "EXAMPLE_SUFFIX": "\n\n",

#   "TRAIN": "train_r3",
    "VALID": "validation",
#   "TEST": "test_r3",

  "FEW_SHOT_COT_PROMPT": (
    'Premise:\n"No Weapons of Mass Destruction Found in Iraq Yet."\nBased on this premise, can we conclude the hypothesis "Weapons of Mass Destruction Found in Iraq." is true?',
    '"No Weapons of Mass Destruction Found" contradicts "Weapons of Mass Destruction Found".', 
    'no',

    'Premise:\n"A place of sorrow, after Pope John Paul II died, became a place of celebration, as Roman Catholic faithful gathered in downtown Chicago to mark the installation of new Pope Benedict XVI."\nBased on this premise, can we conclude the hypothesis "Pope Benedict XVI is the new leader of the Roman Catholic Church." is true?’',
    '"installation of new Pope Benedict XVI." means "Pope Benedict XVI is the new leader".',
    'yes',


    'Premise:\n"A man is due in court later charged with the murder 26 years ago of a teenager whose case was the first to be featured on BBC One’s Crimewatch. Colette Aram, 16, was walking to her boyfriend’s house in Keyworth, Nottinghamshire, on 30 October 1983 when she disappeared. Her body was later found in a field close to her home. Paul Stewart Hutchinson, 50, has been charged with murder and is due before Nottingham magistrates later."\nBased on this premise, can we conclude the hypothesis "Paul Stewart Hutchinson is accused of having stabbed a girl." is true?',
    'The premise does not say Paul Stewart Hutchinson "stabbed" this girl.', 
    'no',

    'Premise:\n"Herceptin was already approved to treat the sickest breast cancer patients, and the company said, Monday, it will discuss with federal regulators the possibility of prescribing the drug for more breast cancer patients." Based on this premise, can we conclude the hypothesis "Herceptin can be used to treat breast cancer." is true?',
    '"Herceptin was approved to treat breast cancer" implies that "Herceptin can be used to treat breast cancer".', 
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

class RTE(CoTDataset):
    def __init__(self, cfg, split, dataset_cfg):
        super().__init__(cfg, split, dataset_cfg)

        self.dataset = load_dataset(path=self.dataset_cfg.HF_IDENTIFIER, name=self.dataset_cfg.HF_NAME, split=self.split_name_str)
        new_dataset = []
        for i in range(len(self.dataset)):
            example = self.dataset[i]
            new_dataset.append(self.convert_example(example))
        self.dataset = new_dataset
        self.limit_size()
    
    def convert_example(self, example):
        # could have multiple ways 
        question = 'Premise:\n"{}"\nBased on this premise, can we conclude the hypothesis "{}" is true?'.format(example['sentence1'], example['sentence2'])
        if example["label"] == 0:
            answer_word = "yes"
        elif example["label"] == 1:
            answer_word = "no"
        target = answer_word
        return question, target, None