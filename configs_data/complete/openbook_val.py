# use py as the config file because yaml throws a tone of errors when parsing strings
cfg = {
  "TYPE": ("commonsense", ),
  "SOURCE": "hf",
  "CLEAN_STRATEGY": "answer_split",
  "HF_IDENTIFIER": "openbookqa",
  "HF_NAME": "main",
  "FEW_SHOT_PROMPT": (),

  "USE_PROMPT_SOURCE": False,

  "RANDOM_DROP_COT_PROMPTS_MAX_NUM": 2,

  "PROMPT_SOURCE_BAD_TEMPLATE": ("only_options",),

  "MODULE_NAME": "configs_data.complete.openbook",
  "CLASS_NAME": "OPBooKataset",
  "INPUT_PREFIX": "Q: ",

  "TRAIN": "validation",
  "VALID": "validation",
  "TEST": "test",

  "FEW_SHOT_COT_PROMPT": (
    "Poison causes harm to which of the following? (a) a Tree (b) a robot (c) a house (d) a car",
    "Poison will harm living things, only a tree is a living thing.",
    "(a)",

    "As you look deeper into a Marbel you can see (a) the future (b) minut defects (c) colors (d) the other side",
    "Marbel is not transparent, so you can not see the other side. Marbel does not necessarily have multiple colors. You will see minut defects.",
    "(b)",

    "When food is reduced in the stomach (a) the mind needs time to digest (b) take a second to digest what I said (c) nutrients are being deconstructed (d) reader's digest is a body of works",
    "The food is being deconstructed in the stomach during digestion.", 
    "(c)",

    "The sun is responsible for (a) puppies learning new tricks (b) children growing up and getting old (c) flowers wilting in a vase (d) plants sprouting, blooming and wilting",
    "The sun can affect the growing of living things, like plants.",
    "(d)"
  ),

  "FEW_SHOT_PROMPT_INDEX": (5, 12, 2, 0),

  "FEW_SHOT_PROMPT_ORIGINAL": (1, 2, ) # placeholder
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

class OPBooKataset(CoTDataset):
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
      choices = example["choices"]["text"]
      question = "{} (a) {} (b) {} (c) {} (d) {}".format(example["question_stem"], choices[0], choices[1], choices[2], choices[3])
      rationale = ""
      answer = "({})".format(example["answerKey"]).lower()
      return question, answer, rationale
    
  def convert_original_to_examples(self, original_prompts):
      # all_examples = []
      # for i in self.dataset_cfg.FEW_SHOT_PROMPT_INDEX:
      #     all_examples.append(self.hf_dataset[i])
      # print(all_examples)
      # pdb.set_trace()
      examples = [{'id': '9-782', 'question_stem': 'Poison causes harm to which of the following?', 'choices': {'text': ['a Tree', 'a robot', 'a house', 'a car'], 'label': ['A', 'B', 'C', 'D']}, 'answerKey': 'A'}, {'id': '9-572', 'question_stem': 'As you look deeper into a Marbel you can see', 'choices': {'text': ['the future', 'minut defects', 'colors', 'the other side'], 'label': ['A', 'B', 'C', 'D']}, 'answerKey': 'B'}, {'id': '7-870', 'question_stem': 'When food is reduced in the stomach', 'choices': {'text': ['the mind needs time to digest', 'take a second to digest what I said', 'nutrients are being deconstructed', "reader's digest is a body of works"], 'label': ['A', 'B', 'C', 'D']}, 'answerKey': 'C'}, {'id': '7-980', 'question_stem': 'The sun is responsible for', 'choices': {'text': ['puppies learning new tricks', 'children growing up and getting old', 'flowers wilting in a vase', 'plants sprouting, blooming and wilting'], 'label': ['A', 'B', 'C', 'D']}, 'answerKey': 'D'}]
      return examples
