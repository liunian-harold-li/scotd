# use py as the config file because yaml throws a tone of errors when parsing strings
cfg = {
  "TYPE": ("numerical", "symbolic", ),
  "SOURCE": "hf",
  "CLEAN_STRATEGY": "general",
  "HF_IDENTIFIER": "bigbench",
  "HF_NAME": "strategyqa",

  "MODULE_NAME": "configs_data.complete.strategyqa",
  "CLASS_NAME": "StrategyQADataset",
  # "INPUT_PREFIX": "Answer the following question. ",
  #"STEP_PREFIX": "\nA: ", # this is the prompt for the answer
  #"ANSWER_PREFIX": "\nThe answer is: ",
  #"EXAMPLE_SUFFIX": "\n\n",

  "TRAIN": "train",
  "VALID": "validation",
  "TEST": "test",

  "FEW_SHOT_COT_PROMPT": (
    "Q: Do hamsters provide food for any animals?",
    "Hamsters are prey animals. Prey are food for predators. Thus, hamsters provide food for some animals.",
    "Yes",


    "Q: Could Brooke Shields succeed at University of Pennsylvania?",
    "Brooke Shields went to Princeton University. Princeton University is about as academically rigorous as the University of Pennsylvania. Thus, Brooke Shields could also succeed at the University of Pennsylvania.",
    "Yes",


    
    "Q: Yes or no: Hydrogen's atomic number squared exceeds number of Spice Girls?",
    
    "Hydrogen has an atomic number of 1. 1 squared is 1. There are 5 Spice Girls. Thus, Hydrogen's atomic number squared is less than 5.",
    
    "No",
    
    
    "Q: Yes or no: Is it common to see frost during some college commencements?",
    "College commencement ceremonies can happen in December, May, and June. December is in the winter, so there can be frost. Thus, there could be frost at some commencements.",
   "Yes",
   
   "Q: Yes or no: Could a llama birth twice during War in Vietnam (1945-46)?",
   "The War in Vietnam was 6 months. The gestation period for a llama is 11 months, which is more than 6 months. Thus, a llama could not give birth twice during the War in Vietnam.",
   "No",
   
   "Q: Yes or no: Would a pear sink in water?",
   "The density of a pear is about 0.6 g/cm^3, which is less than water. Objects less dense than water float. Thus, a pear would float.",
   "No"
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

class StrategyQADataset(CoTDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset = load_dataset(path=self.dataset_cfg.HF_IDENTIFIER, name=self.dataset_cfg.HF_NAME, split=self.split_name_str)

        # dataset_fields: ['question', 'answer']
        new_dataset = []
        inputs = self.dataset["inputs"] # Q: Would New Year's Eve hypothetically be Bacchus's favorite holiday?\nA:
        inputs = [i[:-3] for i in inputs]
        targets = self.dataset["targets"]

        for i in tqdm(range(len(inputs))):
            
            question = inputs[i]
            label_word = "Yes" if targets[i][:3] == "Yes" else "No"
    
            _ = (question, label_word)
            new_dataset.append(_)

        self.dataset = new_dataset
        self.limit_size()