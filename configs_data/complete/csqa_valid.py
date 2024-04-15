# use py as the config file because yaml throws a tone of errors when parsing strings
cfg = {
  "TYPE": ("commonsense", ),
  "SOURCE": "hf",
  "CLEAN_STRATEGY": "answer_split",
  "HF_IDENTIFIER": "commonsense_qa",
  "HF_NAME": "default",
  "FEW_SHOT_PROMPT": (),

  "USE_PROMPT_SOURCE": False,
  "PROMPT_SOURCE_BAD_TEMPLATE": ("answer_given_question_without_options",),


  "MODULE_NAME": "configs_data.complete.commonsenseqa_old",
  "CLASS_NAME": "CommonsenseQADataset",
  "INPUT_PREFIX": "Q: ",

  "TRAIN": "validation",
  "VALID": "validation",
  "TEST": "validation",

  "RANDOM_DROP_COT_PROMPTS_MAX_NUM": 5,

  "FEW_SHOT_COT_PROMPT": (
    "What do people use to absorb extra ink from a fountain pen?\nAnswer Choices:\n(a) shirt pocket\n(b) calligrapher's hand\n(c) inkwell\n(d) desk drawer\n(e) blotter",
    "The answer must be an item that can absorb ink. Of the above choices, only blotters are used to absorb ink.",
    "(e)",

    "What home entertainment equipment requires cable?\nAnswer Choices:\n(a) radio shack\n(b) substation\n(c) cabinet\n(d) television\n(e) desk",
    "The answer must require cable. Of the above choices, only television requires cable.",
    "(d)",

    "The fox walked from the city into the forest, what was it looking for?\nAnswer Choices:\n(a) pretty flowers.\n(b) hen house\n(c) natural habitat\n(d) storybook\n(e) dense forest",
    "The answer must be something in the forest. Of the above choices, only natural habitat is in the forest.",
    "(c)",

    "Sammy wanted to go to where the people were.  Where might he go?\nAnswer Choices:\n(a) race track\n(b) populated areas\n(c) the desert\n(d) apartment\n(e) roadblock",
    "The answer must be a place with a lot of people. Of the above choices, only populated areas have a lot of people.",
    "(b)",

    "Where do you put your grapes just before checking out?\nAnswer Choices:\n(a) mouth\n(b) grocery cart\n(c) super market\n(d) fruit basket\n(e) fruit market",
    "The answer should be the place where grocery items are placed before checking out. Of the above choices, grocery cart makes the most sense for holding grocery items.",
    "(b)",


    "Google Maps and other highway and street GPS services have replaced what?\nAnswer Choices:\n(a) united states\n(b) mexico\n(c) countryside\n(d) atlas\n(e) oceans",
    "The answer must be something that used to do what Google Maps and GPS services do, which is to give directions. Of the above choices, only atlases are used to give directions.",
    "(d)",

    "Before getting a divorce, what did the wife feel who was doing all the work?\nAnswer Choices:\n(a) harder\n(b) anguish\n(c) bitterness\n(d) tears\n(e) sadness",
    "The answer should be the feeling of someone getting divorced who was doing all the work. Of the above choices, the closest feeling is bitterness.",
    "(c)",
  ),

  "FEW_SHOT_PROMPT_INDEX": (8, 5, 4, 1, 10, 3, 11),

  "FEW_SHOT_PROMPT_ORIGINAL": (1, 2, ) # place holder

    # 1e1d0ce1-b0ea-4ad8-9971-b2b44948123b: !Template
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

class CommonsenseQADataset(CoTDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.dataset = load_dataset(path=self.dataset_cfg.HF_IDENTIFIER, name=self.dataset_cfg.HF_NAME, split=self.split_name_str)

        # dataset_fields: ['question', 'answer']
        new_dataset = []
        answers = self.dataset["answerKey"]
        questions = self.dataset["question"]
        choices = self.dataset["choices"]
        for i in tqdm(range(len(self.dataset["question"]))):
            # zip choices[i]["label"] and choices[i]["text"]
            choices_label_text = list(zip(choices[i]["label"], choices[i]["text"]))
            choices_label_text = "\n".join(["({}) {}".format(x[0].lower(), x[1]) for x in choices_label_text])

            question = questions[i] + "\nAnswer Choices:\n" + choices_label_text

            answer = "({})".format(answers[i].lower())

            _ = (question, answer)
            new_dataset.append(_)

        self.dataset = new_dataset
        self.limit_size()

    def convert_original_to_examples(self, original_prompts):
        examples = [
          {'question': 'What do people use to absorb extra ink from a fountain pen?', 'question_concept': 'fountain pen', 'choices': {'label': ['A', 'B', 'C', 'D', 'E'], 'text': ['shirt pocket', "calligrapher's hand", 'inkwell', 'desk drawer', 'blotter']}, 'answerKey': 'E'},
          {'question': 'What home entertainment equipment requires cable?', 'question_concept': 'cable', 'choices': {'label': ['A', 'B', 'C', 'D', 'E'], 'text': ['radio shack', 'substation', 'cabinet', 'television', 'desk']}, 'answerKey': 'D'},
          {'question': 'The fox walked from the city into the forest, what was it looking for?', 'question_concept': 'fox', 'choices': {'label': ['A', 'B', 'C', 'D', 'E'], 'text': ['pretty flowers.', 'hen house', 'natural habitat', 'storybook', 'dense forest']}, 'answerKey': 'C'},
          {'question': 'Sammy wanted to go to where the people were.  Where might he go?', 'question_concept': 'people', 'choices': {'label': ['A', 'B', 'C', 'D', 'E'], 'text': ['race track', 'populated areas', 'the desert', 'apartment', 'roadblock']}, 'answerKey': 'B'},
          {'question': 'Where do you put your grapes just before checking out?', 'question_concept': 'grape', 'choices': {'label': ['A', 'B', 'C', 'D', 'E'], 'text': ['mouth', 'grocery cart', 'super market', 'fruit basket', 'fruit market']}, 'answerKey': 'B'},
          {'question': 'Google Maps and other highway and street GPS services have replaced what?', 'question_concept': 'highway', 'choices': {'label': ['A', 'B', 'C', 'D', 'E'], 'text': ['united states', 'mexico', 'countryside', 'atlas', 'oceans']}, 'answerKey': 'D'},
          {'question': 'Before getting a divorce, what did the wife feel who was doing all the work?', 'question_concept': 'getting divorce', 'choices': {'label': ['A', 'B', 'C', 'D', 'E'], 'text': ['harder', 'anguish', 'bitterness', 'tears', 'sadness']}, 'answerKey': 'C'}
        ]
        return examples
