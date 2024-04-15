# use py as the config file because yaml throws a tone of errors when parsing strings
cfg = {
  "TYPE": ("commonsense", ),
  "SOURCE": "hf",
  "CLEAN_STRATEGY": "answer_split",
  "HF_IDENTIFIER": "ai2_arc",
  "HF_NAME": "ARC-Challenge",
  "FEW_SHOT_PROMPT": (),

  "USE_PROMPT_SOURCE": False,

  "MODULE_NAME": "configs_data.complete.arc_c",
  "CLASS_NAME": "ARC_C",
  
  "INPUT_PREFIX": "Q: ",

#   "TRAIN": "train_r3",
  "VALID": "validation",
#   "TEST": "test",

  "FEW_SHOT_COT_PROMPT": (
    "George wants to warm his hands quickly by rubbing them. Which skin surface will produce the most heat? (a) dry palms. (b) wet palms. (c) palms covered with oil. (d) palms covered with lotion.",
    "Dry surfaces will more likely cause more friction via rubbing than other smoother surfaces, hence dry palms will produce the most heat.",
    "(a)",
    
    "Which factor will most likely cause a person to develop a fever? (a) a leg muscle relaxing after exercise. (b) a bacterial population in the bloodstream. (c) several viral particles on the skin. (d) carbohydrates being digested in the stomach.",
    "Option (b), bacterial population is the most likely cause for a person developing fever.",
    "(b)",
    
    "Which change in the state of water particles causes the particles to become arranged in a fixed position? (a) boiling. (b) melting. (c) freezing. (d) evaporating.",
    "When water is freezed, the particles are arranged in a fixed position; the particles are still moving for all other options.",
    "(c)",
    
    "When a switch is used in an electrical circuit, the switch can (a) cause the charge to build. (b) increase and decrease the voltage. (c) cause the current to change direction. (d) stop and start the flow of current.",
    "The function of a switch is to start and stop the flow of a current.",
    "(d)",

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

class ARC_C(CoTDataset):
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
        
        choices = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)", "(i)", "(j)"]
        choice_string = ""
        for index, i in enumerate(example["choices"]["text"]):
            choice_string += f"{choices[index]} {i} "
        question = "{} {}".format(example["question"], choice_string)
        target = "({})".format(example["answerKey"]).lower()
        return question, target, None