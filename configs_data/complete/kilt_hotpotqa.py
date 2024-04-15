# use py as the config file because yaml throws a tone of errors when parsing strings
cfg = {
  "TYPE": ("commonsense", ),
  "SOURCE": "hf",
  "CLEAN_STRATEGY": "answer_split",
  "HF_IDENTIFIER": "kilt_tasks",
  "HF_NAME": "hotpotqa",
  "FEW_SHOT_PROMPT": (),

  "USE_PROMPT_SOURCE": False,

  "MODULE_NAME": "configs_data.complete.kilt_hotpotqa",
  "CLASS_NAME": "KILT_HOTPOTQA",
  #"INPUT_PREFIX": "Read the following sentence and choose the sentence that will most likely happen next.\n",

#   "TRAIN": "train_r3",
    "VALID": "validation",
#   "TEST": "test_r3",

  "FEW_SHOT_COT_PROMPT": (
    "He picks up the yellow ball and throws it across to his father. When his father throws the ball to him, he catches it with the lacrosse stick. his mother (A) backs away when the ball lands on the ground. (B) screams at him over it. (C) comes to his aid to throw the ball to him. (D) gets excited and shouts out for him.",
    "He catches the ball with the lacrosse stick, which is a very cool and exciting act. Thus, his mother would get excited and shouts out for him.",
    "(D)",

    "Two men enter the racquetball room. the two men (A) start playing racquetball, running around the room trying to hit the ball. (B) attempt to move forward blocking balls. (C) play ping pong as the spectators applaud. (D) take a seat on the table.",
    "People play racquetball in the racquetball room. Thus, the two men would mostly likely start playing racquetball.",
    "(A)",

    "A man is standing up in a red row boat. A boat flips over and the camera goes into the water. two people standing on a dock (A) are then seen floating past one another. (B) are being pulled over by a jet ski. (C) are underwater looking at the camera. (D) lift their boat out of the water.",
    "The boat flips over and the man falls in the water. Thus, the two people on a dock would try to rescue the man and help lift the boat.",
    "(D)",

    "A group of hockey players are on the ice in a gym. They are using sticks to hit at the puck back and forth. the players (A) fight over the puck, trying to get it into the goal. (B) ferociously hit at each other. (C) kick the puck back and forth across the ice in such an elaborate fashion, it looks like they would overlay all of the hockey blocks to their attackers. (D) kick their legs up and down.",
    "When playing hockey, the goal is to get the puck into the goal. Thus, the players would fight over the puck.",
    "(A)",

  ),

  "FEW_SHOT_PROMPT_INDEX": (1000, 1029, 8764, 4878)
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
id2alphabet = {0: "(A)", 1: "(B)", 2: "(C)", 3: "(D)", 4: "(E)"}

class KILT_HOTPOTQA(CoTDataset):
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
        return datapoint["input"].replace("\n", " "), "\t".join([item["answer"] for item in datapoint["output"]]), None