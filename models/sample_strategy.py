from statistics import mean
from torch.utils.data import Dataset
from collections import OrderedDict
import xml.etree.ElementTree as ET
import openai # For GPT-3 API ...
import os
import multiprocessing
import json
import numpy as np
import random
import torch
import torchtext
import re
import random
import time
import datetime
import pandas as pd
import pdb
from arguments import yaml_config_to_args
from copy import deepcopy
from dataset.answer_cleaner import AnswerCleaner

def answer_cleansing(args, pred):    
    if args.method in ("few_shot", "few_shot_cot"):
        preds = pred.split(args.direct_answer_trigger_for_fewshot)
        answer_flag = True if len(preds) > 1 else False 
        pred = preds[-1]
    if args.dataset in ("aqua", "commonsensqa"):
        pred = re.findall(r'A|B|C|D|E', pred)
    elif args.dataset == "bigbench_date":
        pred = re.findall(r'A|B|C|D|E|F', pred)
    elif args.dataset in ("object_tracking"):
        pred = re.findall(r'A|B|C', pred)
    elif args.dataset in ("gsm8k", "addsub", "multiarith", "svamp", "singleeq"):
        pred = pred.replace(",", "")
        pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]
    elif args.dataset in ("strategyqa", "coin_flip"):
        pred = pred.lower()
        pred = re.sub("\"|\'|\n|\.|\s|\:|\,"," ", pred)
        pred = pred.split(" ")
        pred = [i for i in pred if i in ("yes", "no")]
    elif args.dataset == "last_letters":
        pred = re.sub("\"|\'|\n|\.|\s","", pred)
        pred = [pred]
    else:
        raise ValueError("dataset is not properly defined ...")

    # If there is no candidate in list, null is set.
    if len(pred) == 0:
        pred = ""
    else:
        if args.method in ("few_shot", "few_shot_cot"):
            if answer_flag:
                # choose the first element in list ...
                pred = pred[0]
            else:
                # choose the last element in list ...
                pred = pred[-1]
        elif args.method in ("zero_shot", "zero_shot_cot"):
            # choose the first element in list ...
            pred = pred[0]
        else:
            raise ValueError("method is not properly defined ...")
    
    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred != "":
        if pred[-1] == ".":
            pred = pred[:-1]
        
    return pred

# Sentence Generator (Decoder) for GPT-3 ...

class SampleStrategy:
    def __init__(self, cfg):
        self.cfg = cfg
        self.args = yaml_config_to_args(cfg.DATA.STEP_BY_STEP) # quick a hack
        self.answer_cleaner = AnswerCleaner(cfg)
    def __call__(self,
               model,
               prompts = None,
               answers = None,
               sample_rounds = 1,
               **kwargs):
        
        output = {
            'query/input_ids': None,
            'query/text': None,
            'query/mask': None,
            'query/answer': answers,
            'response/input_ids': None,
            'response/text': None,
            'response/mask': None,
            'response/log_prob': None,
            'response/answer': None
        }

        # Answer prediction by generating text ...
        max_length = self.args.max_length_cot if "cot" in self.args.method else self.args.max_length_direct

        z = model.sample(
            prompts = prompts, 
            answers = answers,
            max_length = max_length,
            sample_rounds = sample_rounds,
            **kwargs)

        output.update(z)

        text_z = z["response/text"]

        if self.cfg.MISC.VERBOSE:
            print("Prompt: ")
            print(prompts)
            print("\n\n")
            print("First response: ")
            print("\n".join(text_z))

        all_response = deepcopy(text_z)

        # Answer extraction for zero-shot-cot ...
        if self.args.method == "zero_shot_cot":
            z2 = []
            
            for i in range(len(text_z)):
                z2.append(prompts[i] + text_z[i] + " " + self.args.direct_answer_trigger_for_zeroshot_cot)
            pred = model.sample(
                prompts = z2, 
                max_length = self.args.max_length_direct, 
                **kwargs)

            all_response = [ text_z[i] + " " + self.args.direct_answer_trigger_for_zeroshot_cot + pred["response/text"][i] for i in range(len(text_z)) ]

            if self.cfg.MISC.VERBOSE:
                print("\n\n")
                print('Second response:')
                print("\n".join(pred["response/text"]))
            
        else:
            z2 = text_z
            pred = z
        
        text_pred = pred["response/text"]

        # Clensing of predicted answer ...
        pred_answers = [self.answer_cleaner(_) for _ in text_pred]
        pred_rationales = [self.answer_cleaner.parse_rationale(_) for _ in text_pred]
        if self.cfg.MISC.VERBOSE:
            print("\n\n")
            print("Predicted answer:")
            print("\n".join(pred_answers))

        output.update({
            'query/input_ids': None,
            'query/text': prompts,
            'query/mask': None,
            'query/answer': answers,
            'response/input_ids': None,
            'response/text': all_response,
            'response/mask': None,
            'response/log_prob': None,
            'response/answer': pred_answers,
            "response/rationale": pred_rationales
        })
        return output
