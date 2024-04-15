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


class GPT3Policy:
    def __init__(self, cfg):
        self.cfg = cfg
    
    def sample(self,
               prompts = None,
               answers = None,
               max_length = None,
               sample_rounds = 1,
               **kwargs):

        # GPT-3 API allows each users execute the API within 60 times in a minute ...
        # time.sleep(1)
        time.sleep(self.cfg.SLEEP_TIME)
        
        # https://beta.openai.com/account/api-keys
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        # Specify engine ...
        # Instruct GPT3
        if self.cfg.NAME == "gpt3":
            engine = "text-ada-001"
        elif self.cfg.NAME == "gpt3-medium":
            engine = "text-babbage-001"
        elif self.cfg.NAME == "gpt3-large":
            engine = "text-curie-001"
        elif self.cfg.NAME == "gpt3-xl":
            engine = "text-davinci-002"
        elif self.cfg.NAME == "gpt3-xl-code":
            engine = "code-davinci-002"
        else:
            raise Exception("Unknown model name: {}".format(self.cfg.NAME))
        success = False
        #pdb.set_trace()
        while not success:
            try:
                response = openai.Completion.create(
                engine=engine, prompt=prompts, max_tokens=max_length, temperature=self.cfg.TEMPERATURE, n=sample_rounds, stop=self.cfg.STOP_TOKEN_GPT, logprobs=1)
                success = True
            except Exception as e:
                print("Exception: {}".format(e))
                time.sleep(45)

        return {
            'response/text': [i['text'] + self.cfg.ADDITIONL_SUFFIX_STRING for i in response.choices],
            'response/token_logprobs': [i["logprobs"]["token_logprobs"] for i in response.choices],
            'response/tokens': [i["logprobs"]["tokens"] for i in response.choices],
        }
    # [i['text'] for i in response.choices]

    def forward_pass(self, **kwargs):
        assert(0)
