import torch
import torch.nn.functional as F
import numpy as np
from typing import Union, List, Dict
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from utils.utils import logits_to_entropy, mask_pad
from .gpt3_api import GPT3Policy
from .simple_reward import SimpleReward
from .automodel import AutoModelPolicy
from .empty_policy import EmptyPolicy
from arguments import yaml_config_to_args

def build_policy(model_cfg):
    if model_cfg.TYPE == "auto":
        policy = AutoModelPolicy(model_cfg)
    if model_cfg.TYPE == "api":
        policy = GPT3Policy(model_cfg)
    if model_cfg.TYPE == "empty_baseline":
        policy = EmptyPolicy(model_cfg)
    return policy

def build_policys(cfg):

    ref_policy = build_policy(cfg.MODEL)
    policy = build_policy(cfg.POLICY_MODEL)
    reward = SimpleReward()
    
    return ref_policy, policy, reward