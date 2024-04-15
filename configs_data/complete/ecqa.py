# use py as the config file because yaml throws a tone of errors when parsing strings
cfg = {
  "TYPE": ("commonsense", ),
  "SOURCE": "hf",
  "CLEAN_STRATEGY": "general",
  "FEW_SHOT_PROMPT": (),

  "USE_PROMPT_SOURCE": False,

  "MODULE_NAME": "configs_data.complete.ecqa",
  "CLASS_NAME": "ECQADataset",
  "LOCAL_PATH": "data/ecqa/",
  "INPUT_PREFIX": "Q: ",
  "STEP_PREFIX": "\nA: ", # this is the prompt for the answer
  "EXAMPLE_SUFFIX": "\n\n",

  "USE_EXPLAINATION": True,

  "TRAIN": "cqa_data_train",
  "VALID": "cqa_data_val",
  "TEST": "cqa_data_test",

  "FEW_SHOT_COT_PROMPT": (
    "Where would you borrow coffee if you do not have any?\nAnswer Choices:\n(a) meeting\n(b) convenience store\n(c) supermarket\n(d) fast food restaurant\n(e) friend's house",
    "If you are finished with stock of coffee beans / powder and don't want to buy it, you can borrow it from friend's home because you can't borrow from meeting and other options are selling the coffee.",
    "(e)",

    "If you want to set a romantic atmosphere you might light a candle where?\nAnswer Choices:\n(a) dimly lit room\n(b) synagogue\n(c) bedroom\n(d) birthday cake\n(e) roses",
    "A romantic atmosphere can be set in bedroom and not in a synagogue. Bedroom is a place where one sleeps unlike a dimly lit room or a birthday cake. Candles can be lit in a bedroom and not in roses.",
    "(c)",

    "What is the likelihood of drowning when getting wet?\nAnswer Choices:\n(a) shrinking\n(b) feeling cold\n(c) become cold\n(d) cool off\n(e) could",
    "One could drown in too much water. So the likelihood of drowning when getting wet is they could. All other options are not likelihood.",
    "(e)",

    "What does the government have control over?\nAnswer Choices:\n(a) trouble\n(b) country\n(c) army\n(d) city\n(e) control",
    "A city is a large town over which a government has control. One cannot have control over control and a government might not have control over others",
    "(d)",

    "He had a lot on his plate opening business, this cause a lot of what?\nAnswer Choices:\n(a) headaches\n(b) making money\n(c) success\n(d) failure\n(e) stress",
    "When someone has lot on plate, they often feel stressed. A new business demands lot o fwork that can cause stress. All the other options are incorrect as they are not a result of being a lot on plate in a business.",
    "(e)",


    "If you want to set a romantic atmosphere you might light a candle where?\nAnswer Choices:\n(a) dimly lit room\n(b) synagogue\n(c) bedroom\n(d) birthday cake\n(e) roses",
    "A romantic atmosphere can be set in bedroom and not in a synagogue. Bedroom is a place where one sleeps unlike a dimly lit room or a birthday cake. Candles can be lit in a bedroom and not in roses.",
    "(c)",

    "Q: What is the likelihood of drowning when getting wet?\nAnswer Choices:\n(a) shrinking\n(b) feeling cold\n(c) become cold\n(d) cool off\n(e) could",
    "One could drown in too much water. So the likelihood of drowning when getting wet is they could. All other options are not likelihood.",
    "(e)",
  )

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


class ECQADataset(CoTDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        local_path = "{}/{}.json".format(self.dataset_cfg.LOCAL_PATH, self.split_name_str)
        with open(local_path, "r") as f:
            data = json.load(f)
        
        '''
        "@ID": "nluds-0945",
        "@Grade": "3",
        "@Source": "http://www.mathplayground.com",
        "Body": "Tommy has $79. He wants to buy a $35 camera. He also wants to buy a $59 CD player.",
        "Question": "How much more money does Tommy need?",
        "Solution-Type": "TVQ-Change",
        "Answer": "15 (dollars)",
        "Formula": "35+59-79=15"
        '''
        new_dataset = []
        questions = data["q_text"]
        q_op1 = data["q_op1"]
        q_op2 = data["q_op2"]
        q_op3 = data["q_op3"]
        q_op4 = data["q_op4"]
        q_op5 = data["q_op5"]

        q_ans = data["q_ans"]
        explaination = data["taskB"]


        for i in tqdm(range(len(questions))):
            question = "{}\nAnswer Choices:\n(a) {}\n(b) {}\n(c) {}\n(d) {}\n(e) {}".format(questions[i], q_op1[i], q_op2[i], q_op3[i], q_op4[i], q_op5[i])
            answer = q_ans[i]
            if answer == q_op1[i]:
                answer = "(a)"
            elif answer == q_op2[i]:
                answer = "(b)"
            elif answer == q_op3[i]:
                answer = "(c)"
            elif answer == q_op4[i]:
                answer = "(d)"
            elif answer == q_op5[i]:
                answer = "(e)"
            else:
                print("Error", answer, q_op1, q_op2, q_op3, q_op4, q_op5)
                assert(0)
            
            if self.dataset_cfg.USE_EXPLAINATION:
                target = "{}. So the answer is: {}".format(explaination[i], answer)
            else:
                target = answer

            new_dataset.append((question, explaination[i], answer))

        self.dataset = new_dataset
        self.limit_size()
    
    def __getitem__(self, original_index, rationale = None):
        index = self.valid_indexes[original_index]
        output_x, rationale, y = self.dataset[index]

        output_x = self.dataset_cfg.INPUT_PREFIX + output_x + self.dataset_cfg.STEP_PREFIX

        if self.cfg.DATA.PROMPT_ARGS.PREDICT_RATIONALE and self.is_train:
            response = "{}{}{}{}".format(rationale.strip("\n"), self.dataset_cfg.ANSWER_PREFIX, y, self.dataset_cfg.EXAMPLE_SUFFIX)
        else:
            response = "{}{}".format(y, self.dataset_cfg.EXAMPLE_SUFFIX)
        return index, output_x, y, response