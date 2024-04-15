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


class EmptyPolicy:
    def __init__(self, cfg):
        self.cfg = cfg
    
    def sample(self,
               prompts = None,
               answers = None,
               max_length = None,
               **kwargs):
        return {
            'response/text': [i + self.cfg.ADDITIONL_SUFFIX_STRING for i in answers],
        }
    def forward_pass(self, **kwargs):
        pass