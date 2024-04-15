from torch import nn
from torch.utils.data import DataLoader, Dataset
from arguments import yaml_config_to_args
import os
import numpy as np
import pdb
import re
import pdb
from configs_data._dataset_config import _C as DATASET_CFG

class AnswerCleaner():
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset_cfg = DATASET_CFG.clone()
        self.dataset_cfg.merge_from_file(self.cfg.DATA.CONFIG.split(",")[0])
        if self.cfg.DATA.OVERRIDE_CONFIG is not None:
            self.dataset_cfg.merge_from_file(self.cfg.DATA.OVERRIDE_CONFIG.split(",")[0])
    
    def __call__(self, prediction):
        if self.dataset_cfg.CLEAN_STRATEGY == "general":
            prediction = prediction.split(". ")[-1]
            prediction = prediction.strip("\n")
            prediction = prediction.split(' ')[-1].strip('.')
            return prediction
        if self.dataset_cfg.CLEAN_STRATEGY == "answer_split":
            original_prediction = prediction
            # remove leading "\n" but not other "\n"
            while prediction.startswith("\n"):
                prediction = prediction[1:]
            
            prediction = prediction.split(self.dataset_cfg.EXAMPLE_SUFFIX)[0]
            prediction = prediction.split(self.dataset_cfg.ANSWER_PREFIX)[-1]
            prediction = prediction.strip("\n")
            if len(prediction) != 0 and prediction[0] == " ":
                prediction = prediction[1:]

            return prediction
        if "numerical" in self.dataset_cfg.CLEAN_STRATEGY:
            prediction = prediction.split(". ")[-1]
            prediction = prediction.strip("\n")
            p = '[\d]+[.,\d]+|[\d]*[.][\d]+|[\d]+'
            newest = ""
            if re.search(p, prediction) is not None:
                for catch in re.finditer(p, prediction):
                    newest = catch[0].replace(",", "")
                    if newest[-1] == ".":
                        newest = newest[:-1]
                    if "float" in self.dataset_cfg.CLEAN_STRATEGY:
                        try:
                            newest = str(float(newest))
                        except:
                            newest = "None"
                    else:
                        try:
                            newest = str(int(float(newest)))
                        except:
                            newest = "None"
            return newest
    
    def parse_rationale(self, prediction):
        prediction = prediction.split(self.dataset_cfg.EXAMPLE_SUFFIX)[0]
        rationale = self.dataset_cfg.ANSWER_PREFIX.join(prediction.split(self.dataset_cfg.ANSWER_PREFIX)[:-1])
        return rationale