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
  "CLASS_NAME": "QUAREL",
  "INPUT_PREFIX": "Q: ",

#   "TRAIN": "train_r3",
    "VALID": "validation",
#   "TEST": "test_r3",

  "FEW_SHOT_COT_PROMPT": (
    'Mike was snowboarding on the snow and hit a piece of ice. He went much faster on the ice because _____ is smoother. (A) snow (B) ice',
    'When something is smoother, it is easier to slide on. Thus, he could go faster on the ice because ice is smoother.',
    '(B)',

    'I could hear then boy that say close to me clear as day, however I could not hear the young lady sitting in the back of the room.  WHo am I able to hear louder (A) Boy (B) Lady',
    'When someone is close, it is easier to hear them. I also could not hear the young lady well. Thus, I am able to hear the boy louder.',
    '(A)',

    'I watched the snowflakes go from tiny specks in the sky to a nice size once they fell on my face.  When did the snowflakes seem bigger (A) in the sky (B) on my face',
    'When something is closer, it seems bigger. The snowflakes is closer when they are on my face. Thus, they seem bigger when they are on my face.',
    '(B)',
    
    'When Tammy tried to slide the glass mixing bowl down the marble counter top to her mom, it came to a dead stop when it reached the wooden cutting board. The bowl came to a stop because the wooden cutting board has (A) more resistance or (B) less resistance',
    'When something has more resistance, it is harder to slide. Thus, the bowl came to a stop because the wooden cutting board has more resistance.',
    '(A)',

    'Sarah walked through the city and saw a tourist attraction she wanted to visit. She had several blocks to go to get to it, and the attraction looked very small. As she got close it though, it towered over her. This is because when she was close to it the attraction looked (A) much bigger (B) much smaller.',
    'When something is closer, it looks bigger. Thus, the attraction looked much bigger when she was close to it.',
    '(A)',
  ),

  "FEW_SHOT_PROMPT_INDEX": (0, 788, 78, 100, 598)
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

class QUAREL(CoTDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.dataset = load_dataset(path=self.dataset_cfg.HF_IDENTIFIER, name=self.dataset_cfg.HF_NAME, split=self.split_name_str)
        new_dataset = []
        for i in range(len(self.dataset)):
            example = self.dataset[i]
            new_dataset.append(self.convert_example(example))
        self.dataset = new_dataset
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