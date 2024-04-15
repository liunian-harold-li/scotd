# use py as the config file because yaml throws a tone of errors when parsing strings
cfg = {
  "TYPE": ("commonsense", ),
  "SOURCE": "hf",
  "CLEAN_STRATEGY": "answer_split",
  "HF_IDENTIFIER": "quoref",
  "HF_NAME": "default",
  "FEW_SHOT_PROMPT": (),

  "USE_PROMPT_SOURCE": False,

  "MODULE_NAME": "configs_data.complete.quoref",
  "CLASS_NAME": "HFDataset",
  "INPUT_PREFIX": "Read the passage and answer the question.\n",

#   "TRAIN": "train_r3",
    "VALID": "validation",
#   "TEST": "test_r3",

  "FEW_SHOT_COT_PROMPT": (
    "Question: What is the full name of the person Warren invites to the dance?\nContext: Joy Linnett and her stepsister Jackie miss a flight home to Ohio, but the attractive Joy, accustomed to getting her way with men, flirts with pilot Stevie Wilson until he agrees to personally fly the two young women.\nAt home, old beau Warren James comes calling and invites Jackie to a country club's dance. As soon as Joy emerges in a swimsuit, the smitten Warren not only neglects Jackie, he invites her sister to the dance.\nA quarrel ensues between the women's parents. Jackie's dad is outraged by the way his daughter is treated, but Joy's mom says he's just miffed that her daughter is more popular than his.\nStevie calls out of the blue, giving Jackie an idea. She emulates her sister's behavior and wardrobe, persuading Stevie to accompany her to the dance. Once there, all the men get a look at the new Jackie and line up to dance with her, as sister Joy looks on, delighted. Now it is Warren who is neglected, so much so that he gets drunk and proposes marriage to both sisters. In the end, he comes to appreciate that Jackie is the one he really loves.",
    "Warren initially comes calling and invites Jackie. But as soon as Joy emerges in a swinsuit, Warren invites Jackie's sister, who is Joy Linnett.",
    'Joy Linnett',

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

    def convert_example(self, datapoint):
        return "Question: {}\nContext: {}".format(datapoint["question"], datapoint["context"]), "\t".join(datapoint["answers"]["text"]), None