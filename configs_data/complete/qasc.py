# use py as the config file because yaml throws a tone of errors when parsing strings
cfg = {
  "TYPE": ("commonsense", ),
  "SOURCE": "hf",
  "CLEAN_STRATEGY": "answer_split",
  "HF_IDENTIFIER": "qasc",
  "HF_NAME": "default",
  "FEW_SHOT_PROMPT": (),

  "USE_PROMPT_SOURCE": False,

  "MODULE_NAME": "configs_data.complete.qasc",
  "PROMPT_SOURCE_BAD_TEMPLATE": ("is_correct_2", "is_correct_1", "qa_with_combined_facts_1", ),
  "CLASS_NAME": "HFDataset",
  "INPUT_PREFIX": "Q: ",

#   "TRAIN": "train_r3",
    "VALID": "validation",
#   "TEST": "test_r3",

  "FEW_SHOT_COT_PROMPT": (
    'What are two ways you can save food? (A) Fully cooking the oysters (B) Burning fuel and air (C) Bread it and fry it (D) Water it down and drink it (E) converting electricity to heat (F) lowered energy cost (G) Dehydration and salting (H) Burn it and throw it away',
    'Dehydration and salting can make the food lose its water. This makes it hard to bacteria to grow and the food can be perserved.',
    '(G)',

    'Climate can be annalyzed with (A) sphygmomanometer (B) scattered light (C) seasons (D) heat or cold (E) seismometers (F) satellites (G) Water expanding (H) nanometers',
    "Climate can be analyzed from the space with satellites. Satellites can take pictures so we can see the clouds and the weather.",
    '(F)',

    'what can you use to generate power to reduce greenhouse gases? (A) bamboo (B) coal (C) fibers (D) watts (E) energy (F) fossil fuels (G) trees (H) wind',
    "We can use renewable energy to reduce greenhouse gases. Wind is a kind of renewable energy.",
    "(H)",

    "Looking at what has a negative impact on the eyes? (A) Pollution (B) Allergies (C) disease (D) the sun (E) clouds (F) sun's heat (G) the moon (H) glasses",
    "We should not look at the sun with naked eyes. It will burn our eyes.",
    "(D)",

    "What happens when microorganisms get on food? (A) It tastes better (B) Animal survival (C) Dehydration (D) hydrate their cells (E) Nothing, it's normal (F) It gets moldy (G) hyperthyroidism (H) The food is safe to eat",
    "microorganisms often include fungus and bacteria. Fungus will form mold on the food.",
    "(F)",
  ),

  #"FEW_SHOT_PROMPT_INDEX": (10, 78, 400, 1255)
"FEW_SHOT_PROMPT_INDEX": (1, 2, ), # place holder
  "FEW_SHOT_PROMPT_ORIGINAL": (1, 2, ) # place holder
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
        #pdb.set_trace()
        self.limit_size()
    
    def get_choices_and_answer_string(self, datapoint):
        answer_index = datapoint["answerKey"]
        choices_string = ""
        for i in range(len(datapoint["choices"]["label"])):
            if datapoint["choices"]["label"][i] == answer_index:
                answer_string = datapoint["choices"]["text"][i]
            choices_string += " (" + datapoint["choices"]["label"][i] + ") " + datapoint["choices"]["text"][i]
        return choices_string, "({})".format(answer_index)

    def convert_example(self, datapoint):
        choices_string, answer_string = self.get_choices_and_answer_string(datapoint)
        return datapoint["question"] + choices_string, answer_string.strip(), None
    
    def convert_original_to_examples(self, original_prompts):
        examples = [
          {'question': "What are two ways you can save food?", 'choices': {'label': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'], 'text': ['Fully cooking the oysters', 'Burning fuel and air', 'Bread it and fry it', 'Water it down and drink it', 'converting electricity to heat', 'lowered energy cost', 'Dehydration and salting', 'Burn it and throw it away']}, 'answerKey': 'G'},
            {'question': 'Climate can be annalyzed with', 'choices': {'label': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'], 'text': ['sphygmomanometer', 'scattered light', 'seasons', 'heat or cold', 'seismometers', 'satellites', "Water expanding", 'nanometers']}, 'answerKey': 'F'},
            {'question': 'what can you use to generate power to reduce greenhouse gases?', 'choices': {'label': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'], 'text': ['bamboo', 'coal', 'fibers', 'watts', 'energy', 'fossil fuels', 'trees', 'wind']}, 'answerKey': 'H'},
            {'question': 'Looking at what has a negative impact on the eyes?', 'choices': {'label': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'], 'text': ['Pollution', 'Allergies', 'disease', 'the sun', 'clouds', "sun's heat", 'the moon', 'glasses']}, 'answerKey': 'D'},
            {'question': 'What happens when microorganisms get on food?', 'choices': {'label': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'], 'text': ['It tastes better', 'Animal survival', 'Dehydration', 'hydrate their cells', 'Nothing, it\'s normal', 'It gets moldy', 'hyperthyroidism', 'The food is safe to eat']}, 'answerKey': 'F'}
        ]
        return examples