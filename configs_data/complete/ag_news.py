# use py as the config file because yaml throws a tone of errors when parsing strings
cfg = {
  "TYPE": ("commonsense", ),
  "SOURCE": "hf",
  "CLEAN_STRATEGY": "answer_split",
  "HF_IDENTIFIER": "ag_news",
  "HF_NAME": "default",
  "FEW_SHOT_PROMPT": (),

  "USE_PROMPT_SOURCE": False,

  "MODULE_NAME": "configs_data.complete.ag_news",
  "CLASS_NAME": "AGNews",
  
  "INPUT_PREFIX": "Q: ",

#   "TRAIN": "train_r3",
  "VALID": "test",
  "TEST": "test",

  "RANDOM_DROP_COT_PROMPTS_MAX_NUM": 3,

  "FEW_SHOT_COT_PROMPT": (

    'What topic best summarises the following passage?\nTopics: World politics, Sports, Business, Science and technology.\nSentence: Oil and Economy Cloud Stocks\' Outlook  NEW YORK (Reuters) - Soaring crude prices plus worries  about the economy and the outlook for earnings are expected to  hang over the stock market next week during the depth of the  summer doldrums.',
    "'Stocks', 'crude prices', 'economy', 'earnings', and 'stock market' are all related to the topic 'Business'.",
    'Business', # 10

    "What topic best summarises the following passage?\nTopics: World politics, Sports, Business, Science and technology.\nSentence: Burma to relaunch democracy talks \\Burma sets a new date for a constitutional conference seen as a first step towards democracy.",
    "The sentence is talking about the relaunch of democracy talks. 'Burma', 'constitutional conference', and 'democracy' indicate that the topic is 'World politics'.",
    'World politics', # 100009
 
    'What topic best summarises the following passage?\nTopics: World politics, Sports, Business, Science and technology.\nSentence: Ants Form Supercolony Spanning 60 Miles (AP) AP - Normally clannish and agressive Argentine ants have become so laid back since arriving in Australia decades ago that they no longer fight neighboring nests and have formed a supercolony here that spans 60 miles, scientists say.',
    "The sentence is talking about the behaviour of ants. And `scientists say` is a clue that the topic is 'Science and technology'.",
    'Science and technology', # 110

    'What topic best summarises the following passage?\nTopics: World politics, Sports, Business, Science and technology.\nSentence: Delgado Looks Ready to Fly Blue Jays Nest  TORONTO (Reuters) - The Toronto Blue Jays beat the New York  Yankees 4-2 on Saturday but are losing money, losing fans and  at the conclusion of the Major League season on Sunday could be  losing first baseman Carlos Delgado.',
    "'Toronto Blue Jays', 'New York Yankees', the '4-2' scores, 'Major League', and 'first baseman Carlos Delgado' are all related to the topic 'Sports'.",
    'Sports', # 47859

    "What topic best summarises the following passage?\nTopics: World politics, Sports, Business, Science and technology.\nSentence: Renewed battles erupt in Najaf Explosions and gunfire shook Najaf #39;s Old City on Sunday in a fierce battle between US forces and Shiite militants, as negotiations dragged on for the handover",
    "The sentence is talking about the battle between US forces and Shiite militants. 'US forces', 'Shiite militants', and 'negotiations' indicate that the topic is 'World politics'.",
    'World politics', # 8459
  ),

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

class AGNews(CoTDataset):
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
        labels = ["World politics", "Sports", "Business", "Science and technology",]
        question = "What topic best summarises the following passage?\nTopics: World politics, Sports, Business, Science and technology.\nSentence: {}".format(
            example["text"]
        )
        
        target = labels[example["label"]]
        return question, target, None
    
    def convert_original_to_examples(self, original_prompts):
        examples = [
          {'text': 'Oil and Economy Cloud Stocks\' Outlook  NEW YORK (Reuters) - Soaring crude prices plus worries  about the economy and the outlook for earnings are expected to  hang over the stock market next week during the depth of the  summer doldrums.', 'label': 2},
            {'text': 'Burma to relaunch democracy talks \\Burma sets a new date for a constitutional conference seen as a first step towards democracy.', 'label': 0},
            {'text': 'Ants Form Supercolony Spanning 60 Miles (AP) AP - Normally clannish and agressive Argentine ants have become so laid back since arriving in Australia decades ago that they no longer fight neighboring nests and have formed a supercolony here that spans 60 miles, scientists say.', 'label': 3},
            {'text': 'Delgado Looks Ready to Fly Blue Jays Nest  TORONTO (Reuters) - The Toronto Blue Jays beat the New York  Yankees 4-2 on Saturday but are losing money, losing fans and  at the conclusion of the Major League season on Sunday could be  losing first baseman Carlos Delgado.', 'label': 1},
            {'text': 'Renewed battles erupt in Najaf Explosions and gunfire shook Najaf #39;s Old City on Sunday in a fierce battle between US forces and Shiite militants, as negotiations dragged on for the handover', 'label': 0}
        ]
        return examples
