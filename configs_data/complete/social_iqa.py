# use py as the config file because yaml throws a tone of errors when parsing strings
cfg = {
  "TYPE": ("commonsense", ),
  "SOURCE": "hf",
  "CLEAN_STRATEGY": "general",
  "HF_IDENTIFIER": "social_i_qa",
  "HF_NAME": "default",
  "FEW_SHOT_PROMPT": (),

  "MODULE_NAME": "configs_data.complete.social_iqa",
  "CLASS_NAME": "SocialIQADataset",
  "PROMPT_SOURCE_INDEX": 1,
  
  "INPUT_PREFIX": "Q: ",
  "STEP_PREFIX": "\nA: ", # this is the prompt for the answer
  "EXAMPLE_SUFFIX": "\n\n",

  "FEW_SHOT_COT_PROMPT": (
    "Context: Jordan's dog peed on the couch they were selling and Jordan removed the odor as soon as possible.\nQ: How would Jordan feel afterwards?\n(a) selling a couch\n(b) Disgusted\n(c) Relieved",
    "The dog peed on the couch and Jordan had to remove the ordor, which means he touched the urine. Urine is disgusting. Jordan is disgusted.",
    "(b)",

    "Context: Sydney kept a close watch on her daughter and it paid off when she almost got hit by a car but Sydney was their to save her.\nQ: How would Sydney feel afterwards?\n(a) depressed\n(b) wrong\n(c) relieved",
    "Sydney's daughter almost got hit by a car and she was saved by Sydney. Saving someone who was in danger is relieving. Sydney is relieved.",
    "(c)",

    "Context: Skylar told everyone how Alex felt, even though Alex didn't want them to know.\nQ: How would Alex feel as a result?\n(a) upset\n(b) like a gossip\n(c) trustworthy",
    "Skylar did something Alex did not want them to do. This is upsetting. Alex is upset.",
    '(a)',

    "Context: Lee had been feeling down lately so Remy decided to try to make Lee feel better.\nQ: What does Remy need to do before this?\n(a) know what Lee likes\n(b) buy a gift\n(c) improve Lee's confidence",
    "Before Remy could make Lee feel better, Lee has to know what Remy likes.",
    '(a)',
  ),

  "FEW_SHOT_PROMPT_INDEX": (1, 1, ),

  "FEW_SHOT_PROMPT_ORIGINAL": (1, 1, ), # place holder


  "TRAIN": "train",
  "VALID": "validation",
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


class SocialIQADataset(CoTDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.dataset = load_dataset(path=self.dataset_cfg.HF_IDENTIFIER, name=self.dataset_cfg.HF_NAME, split=self.split_name_str)

        # dataset_fields: ['question', 'answer']
        new_dataset = []
        answer_A = self.dataset["answerA"]
        answer_B = self.dataset["answerB"]
        answer_C = self.dataset["answerC"]
        context = self.dataset["context"]
        questions = self.dataset["question"]
        label = self.dataset["label"]
        label = [["A", "B", "C"][int(i)-1] for i in label]

        for i in tqdm(range(len(self.dataset["question"]))):
            question = "Context: {}\nQ: {}\n(a) {}\n(b) {}\n(c) {}".format(
                context[i], questions[i], answer_A[i], answer_B[i], answer_C[i])

            answer = "({})".format(label[i].lower())

            _ = (question, answer)
            new_dataset.append(_)
        #pdb.set_trace()
        self.dataset = new_dataset
        self.limit_size()
    def convert_original_to_examples(self, original_prompts):
        examples = [
          {
            "context": "Jordan's dog peed on the couch they were selling and Jordan removed the odor as soon as possible.",
            "question": "How would Jordan feel afterwards?",
            "answerA": "selling a couch",
            "answerB": "Disgusted",
            "answerC": "Relieved",
            "label": 2,
          },
          {
            "context": "Sydney kept a close watch on her daughter and it paid off when she almost got hit by a car but Sydney was their to save her.",
            "question": "How would Sydney feel afterwards?",
            "answerA": "depressed",
            "answerB": "wrong",
            "answerC": "relieved",
            "label": 3,
          },
          {
            "context": "Skylar told everyone how Alex felt, even though Alex didn't want them to know.",
            "question": "How would Alex feel as a result?",
            "answerA": "upset",
            "answerB": "like a gossip",
            "answerC": "trustworthy",
            "label": 1,
          },
          {
            "context": "Lee had been feeling down lately so Remy decided to try to make Lee feel better.",
            "question": "What does Remy need to do before this?",
            "answerA": "know what Lee likes",
            "answerB": "buy a gift",
            "answerC": "improve Lee's confidence",
            "label": 1,
          },
        ]
        return examples