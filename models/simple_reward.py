import json
import os

import math
import pandas as pd
from tqdm import tqdm
import logging
from transformers import pipeline
from typing import List, Iterable, Dict, Any

from utils.utils import batchify, load_jsonl

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger(__name__)
import numpy as np

class SimpleReward:
    def __init__(self,):
        pass
    def get_reward(self, prompts: List[str], responses: List[str], answers) -> List[float]:
        responses = [response.lower() for response in responses]
        answers = [answer.lower() for answer in answers]
        rewards = []
        for i in range(len(answers)):
            correct = (np.array([answers[i]]) == np.array([responses[i]])).sum().item()
            rewards.append(correct)
        return rewards