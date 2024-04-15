from statistics import mean
from torch.utils.data import Dataset
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
import torch.nn.functional as F
from utils.utils import ensure_dir, ceil_div, reduce_mean, reduce_sum, distinctness

from arguments import yaml_config_to_args
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, AutoModelForPreTraining
from transformers import BertConfig, BertTokenizer, BertModel, BertForMaskedLM, \
                         RobertaConfig, RobertaTokenizer, RobertaModel, RobertaForMaskedLM, \
                         AlbertTokenizer, AlbertConfig, AlbertModel, AlbertForMaskedLM, \
                         T5Config, T5Tokenizer, T5ForConditionalGeneration, \
                         OpenAIGPTTokenizer, OpenAIGPTLMHeadModel, OpenAIGPTConfig, \
                         GPT2Config, GPT2Tokenizer, GPT2LMHeadModel, GPT2Tokenizer, OPTForCausalLM
from collections import namedtuple
from yacs.config import CfgNode
from huggingface_hub import snapshot_download
import os
from accelerate import init_empty_weights, dispatch_model, infer_auto_device_map, load_checkpoint_and_dispatch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
import gc
from transformers.generation_stopping_criteria import (
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from utils.utils import logits_to_entropy, mask_pad
from torch import nn
from pprint import pprint
def decide_model_class(model_cfg):
    if "t5" in model_cfg.NAME:
        assert(model_cfg.LOAD_WITH_FP16 is False)
        return AutoModelForPreTraining
    elif 'T0' in model_cfg.NAME:
        return AutoModelForSeq2SeqLM
    elif "opt" or "gpt" in model_cfg.NAME:
        return AutoModelForCausalLM
    

def load_with_accelerate(model_cfg):
    model_class = decide_model_class(model_cfg)
    checkpoint = model_cfg.NAME
    config = AutoConfig.from_pretrained(checkpoint)
    with init_empty_weights():
        model = model_class.from_config(config)
    # # Initialize the model under the previous context manager breaks the tied weights.
    model.tie_weights()

    # # infer device map
    device_map = infer_auto_device_map(model, no_split_module_classes=["OPTDecoderLayer"], 
        dtype='float16' if model_cfg.LOAD_WITH_FP16 else 'float32')

    del model
    gc.collect()
    
    # Manually override the device map to use the CPU.
    if len(model_cfg.MANUAL_DEVICE_MAP) != 0:
        device_map.update({k:v for k,v in model_cfg.MANUAL_DEVICE_MAP})
    print(device_map)

    if "30b" or "13b" in model_cfg.NAME:
        model = model_class.from_pretrained(
            checkpoint,
            device_map=device_map, 
            offload_folder='offload_folder', 
            torch_dtype=torch.float16 if model_cfg.LOAD_WITH_FP16 else torch.float32,
            offload_state_dict=True,
            max_memory = {k:v for k,v in model_cfg.ACCELERATE_MAX_MEMORY}
        )
    else:
        model = model_class.from_pretrained(
            checkpoint,
            #device_map=device_map, 
            #offload_folder='offload_folder', 
            torch_dtype=torch.float16 if model_cfg.LOAD_WITH_FP16 else torch.float32,
            #offload_state_dict=True,
            #max_memory = {k:v for k,v in model_cfg.ACCELERATE_MAX_MEMORY}
        )
    
    model.tie_weights()

    return model

class NewLineCriteria(StoppingCriteria):
    def __init__(self, new_line_str = "\n", tokenizer = None):
        self.new_line_str = new_line_str
        self.tokenizer = tokenizer
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        decoded_strings = self.tokenizer.batch_decode(input_ids)
        result = [i[-1] == self.new_line_str and i[-2] == self.new_line_str for i in decoded_strings]
        assert(len(result) == 1) 
        return result[0]

class NewLineCriteriaBatched(StoppingCriteria):
    def __init__(self, new_line_str = "\n", tokenizer = None, batch_size = None):
        self.new_line_str = new_line_str
        self.tokenizer = tokenizer
        self.stopped_record = [False] * batch_size
        self.stopped_position = [-1] * batch_size

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        decoded_strings = self.tokenizer.batch_decode(input_ids, clean_up_tokenization_spaces=False)
        for i in range(len(self.stopped_record)):
            if self.stopped_record[i]:
                continue
            self.stopped_record[i] = decoded_strings[i][-1] == self.new_line_str and decoded_strings[i][-2] == self.new_line_str
            if self.stopped_record[i]:
                self.stopped_position[i] = len(input_ids[i])
        return all(self.stopped_record)

class EosCriteriaBatched(StoppingCriteria):
    def __init__(self, tokenizer = None, batch_size = None):
        self.eos_token_id = tokenizer.eos_token_id
        self.tokenizer = tokenizer
        self.stopped_record = [False] * batch_size
        self.stopped_position = [-1] * batch_size

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for i in range(len(self.stopped_record)):
            if self.stopped_record[i]:
                continue
            self.stopped_record[i] = input_ids[i][-1].item() == self.eos_token_id
            if self.stopped_record[i]:
                self.stopped_position[i] = len(input_ids[i])
        return all(self.stopped_record)

class AutoModelPolicy(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        if "gpt2" in model_cfg.NAME:
            model_class = GPT2LMHeadModel
            tokenizer_class = GPT2Tokenizer
            self.encoder_decoder_style = False
        elif "t5" in model_cfg.NAME:
            model_class = T5ForConditionalGeneration
            tokenizer_class = T5Tokenizer
            self.encoder_decoder_style = True
        elif "opt" in model_cfg.NAME:
            self.encoder_decoder_style = False
            model_class = OPTForCausalLM 
            tokenizer_class = AutoTokenizer
        elif "T0" in model_cfg.NAME:
            self.encoder_decoder_style = True
            tokenizer_class = AutoTokenizer
    
        if model_cfg.LOAD_WITH_ACCELERATE:
            self.model = load_with_accelerate(model_cfg)
        else:
            self.model = model_class.from_pretrained(model_cfg.NAME)
            
        self.tokenizer = tokenizer_class.from_pretrained(model_cfg.NAME, use_fast=False)
        
        if "gpt" in model_cfg.NAME:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"
            self.model.config.pad_token_id = self.model.config.eos_token_id

        if self.model_cfg.USE_QUARK:
            tree_tokens = [' _QUARK_CONTROL_TOKEN_{}'.format(str(idx).zfill(5)) for idx in range(model_cfg.REWARD_CATEGORY_NUM)]

            self.tokenizer.add_tokens(tree_tokens, special_tokens=True)

            # get the index of the control token
            self.control_token_idx = self.tokenizer.convert_tokens_to_ids(tree_tokens)

            weights = self.model.get_input_embeddings().weight.detach().numpy()
            mean_weights, std_weights = np.mean(weights, axis=0), np.std(weights, axis=0)
            new_inits = np.vstack([np.random.normal(loc=mean_weights, scale=std_weights) for _ in tree_tokens])

            self.model.resize_token_embeddings(len(self.tokenizer))
            with torch.no_grad():
                new_inits = torch.tensor(new_inits)
                self.model.get_input_embeddings().weight[-len(tree_tokens):, :] = new_inits
        
        if self.model_cfg.NO_OP_TOKEN_NUM != 0:
            self.special_noop_token = "NOOP"
            self.tokenizer.add_tokens([self.special_noop_token], special_tokens=True)
            self.special_noop_token_idx = self.tokenizer.convert_tokens_to_ids([self.special_noop_token])[0]
            weights = self.model.get_input_embeddings().weight.detach().numpy()
            mean_weights, std_weights = np.mean(weights, axis=0), np.std(weights, axis=0)
            new_inits = np.vstack([np.random.normal(loc=mean_weights, scale=std_weights) for _ in [self.special_noop_token]])
            self.model.resize_token_embeddings(len(self.tokenizer))
            with torch.no_grad():
                new_inits = torch.tensor(new_inits)
                self.model.get_input_embeddings().weight[-1:, :] = new_inits

    def sample(self,
               prompts = None,
               answers = None,
               max_length = None,
               sample_rounds = 1,
               **kwargs):
        device = kwargs.pop("device", None)
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
        if hasattr(self.model, "module"):
            model = self.model.module
        else:
            model = self.model
        
        if self.model_cfg.STOP_AT_NEW_LINE_BATCHED:
            stopping_criteria = StoppingCriteriaList([NewLineCriteriaBatched(
                "\n", self.tokenizer, batch_size = inputs.input_ids.size(0) * sample_rounds)])
        elif self.model_cfg.STOP_AT_EOS_BATCHED:
            stopping_criteria = StoppingCriteriaList([EosCriteriaBatched(
                self.tokenizer, batch_size = inputs.input_ids.size(0) * sample_rounds)])
        elif self.model_cfg.STOP_AT_NEW_LINE:
            stopping_criteria = StoppingCriteriaList([NewLineCriteria("\n", self.tokenizer)])
        else:
            stopping_criteria = StoppingCriteriaList()
        
        input_ids = inputs.input_ids.to(device) if device else inputs.input_ids
        if self.model_cfg.USE_QUARK:
            input_ids = self.append_control_tokens(input_ids)
        
        if self.model_cfg.NO_OP_TOKEN_NUM != 0:
            input_ids = self.append_noop_token(input_ids)
        
        generate_dicts = model.generate(
            input_ids, 
            attention_mask = inputs.attention_mask.to(device) if device else inputs.attention_mask,
            pad_token_id = self.tokenizer.pad_token_id,
            max_new_tokens=max_length, 
            stopping_criteria=stopping_criteria, 
            do_sample = self.model_cfg.DO_SAMPLE, 
            temperature = self.model_cfg.TEMPERATURE,
            num_return_sequences = sample_rounds,
            num_beams = self.model_cfg.NUM_BEAMS,
            return_dict_in_generate=True, output_scores=True
            )
        
        generate_ids = generate_dicts["sequences"]
        generate_ids = generate_ids.to("cpu").numpy().tolist()
        if self.model_cfg.STOP_AT_NEW_LINE_BATCHED or self.model_cfg.STOP_AT_EOS_BATCHED:
            generate_ids = [generate_ids[i][:stopping_criteria[0].stopped_position[i]] for i in range(len(generate_ids))]
        
        if not self.encoder_decoder_style:
            generate_ids = [i[input_ids.shape[-1]:] for i in generate_ids]
        response = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True,  clean_up_tokenization_spaces=False)

        # calculate the scores for each decoded sequence
        scores = generate_dicts["scores"]
        sentence_probs = []
        for i in range(len(generate_ids)):
            score_sentence_i = []
            for k in range(len(generate_ids[i])):
                log_softmaxed_score = torch.log_softmax(scores[k][i], dim=-1)
                score = log_softmaxed_score[generate_ids[i][k]]
                score_sentence_i.append(score.item())
            score_sentence_i = sum(score_sentence_i)
            sentence_probs.append(score_sentence_i)
        normalized_sentence_probs = [p / len(generate_ids[index]) for index, p in enumerate(sentence_probs)]
        return {
            'response/text': response,
            'response/normalized_sentence_probs': normalized_sentence_probs,
            'response/sentence_probs': sentence_probs,
        }
    
    def handle_special_tokens(self, query_ids, query_mask, response_ids, response_mask):
        if "gpt2" in self.model_cfg.NAME or "opt" in self.model_cfg.NAME:
            return query_ids, query_mask, response_ids[:, 1:], response_mask[:, 1:]
        else:
            return query_ids, query_mask, response_ids, response_mask
        

    def append_control_tokens(self, query_ids, query_mask = None, scores = None):
        if scores is None:
            scores = [1] * len(query_ids)
        control_token_ids = []
        for i in range(len(scores)): 
            control_token_ids.append(self.control_token_idx[int(scores[i])])

        control_token_ids = torch.tensor(control_token_ids).to(query_ids.device).unsqueeze(1)
        # append control token
        query_ids = torch.cat([control_token_ids, query_ids], dim=1)

        if query_mask is not None:
            query_mask = torch.cat([torch.ones_like(control_token_ids), query_mask], dim=1)
            return query_ids, query_mask
        else:
            return query_ids
    
    def append_noop_token(self, query_ids, query_mask = None):
        noop_token_ids = torch.ones([len(query_ids), self.model_cfg.NO_OP_TOKEN_NUM], dtype=torch.long) * self.special_noop_token_idx
        noop_token_ids = noop_token_ids.to(query_ids.device)
        query_ids = torch.cat([query_ids, noop_token_ids], dim=1)
        if query_mask is not None:
            query_mask = torch.cat([query_mask, torch.ones_like(noop_token_ids)], dim=1)
            return query_ids, query_mask
        else:
            return query_ids

    def forward_pass(self,
                    query = None,
                    response = None,
                    scores = None,
                    **kwargs):
        device = kwargs.pop("device", None)
        

        # create query 
        # Note: tokenizer will try to auto add special tokens, which we will need to remove and take care of 
        if self.model_cfg.MAX_INPUT_LENGTH == -1:
            query_dict = self.tokenizer(
                query, return_tensors="pt", padding=True,)
        else:
            query_dict = self.tokenizer(
                query, return_tensors="pt", padding=True,
                truncation=True, max_length=self.model_cfg.MAX_INPUT_LENGTH)
        query_ids = query_dict.input_ids.to(device)
        query_mask = query_dict.attention_mask.to(device)

        # append quark tokens;
        if self.model_cfg.USE_QUARK:
            query_ids, query_mask = self.append_control_tokens(query_ids, query_mask)
        
        if self.model_cfg.NO_OP_TOKEN_NUM != 0:
            query_ids, query_mask = self.append_noop_token(query_ids, query_mask)
        
        # create response
        response_dict = self.tokenizer(response, return_tensors="pt", padding=True)
        response_ids = response_dict.input_ids.to(device)
        response_mask = response_dict.attention_mask.to(device)
        # handle special tokens
        query_ids, query_mask, response_ids, response_mask = self.handle_special_tokens(
            query_ids, query_mask, response_ids, response_mask)


        if hasattr(self.model, "module"):
            model = self.model.module
        else:
            model = self.model
        if self.encoder_decoder_style:
            response_ids[response_ids == self.tokenizer.pad_token_id] = -100
            outputs = self.model(
                input_ids=query_ids,
                attention_mask=query_mask,
                labels = response_ids,)
            
            return {
                'response/log_prob': None,
                'response/lm_loss': outputs.loss,
                'response/entropy': None,
                'response/logits': None,
                'response/masks': None,
            }
        else:
            batch_size, query_seq_len = query_ids.shape
            input_ids = torch.cat([query_ids, response_ids], dim=-1)

            model_inputs = model.prepare_inputs_for_generation(
                input_ids, 
                attention_mask = torch.cat([query_mask, response_mask], dim=-1)) # this is every important; will auto take care of the padding

            # forward pass to get next token
            outputs = self.model(
                **model_inputs,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            )
            
            # basically to make sure we could calculate the loss for the first RESPONSE token
            # get the first logit
            query_logits = outputs.logits[:, :query_seq_len, :]
            last_non_masked_idx = torch.sum(query_mask, dim=1) - 1
            first_logits = query_logits[range(batch_size), last_non_masked_idx, :]
            # get the second to last logit
            response_logits = outputs.logits[:, query_seq_len:-1, :]
            logits = torch.cat([first_logits[:, None], response_logits], dim=1)

            log_prob = F.log_softmax(logits, dim=-1)
            output_logprob = torch.gather(log_prob, 2, response_ids[:, :, None]).squeeze(2)
            output_entropy = logits_to_entropy(logits)
            lm_loss = -1. * output_logprob
            if self.model_cfg.VANILLA_POLICY_GRADIENT:
                scores = [1 if i == 1 else self.model_cfg.VANILLA_POLICY_GRADIENT_WEIGHT for i in scores]
                scores = torch.tensor(scores).to(device)
                lm_loss = lm_loss * scores.unsqueeze(1)
            lm_loss = reduce_mean(lm_loss, response_mask)
            return {
                'response/log_prob': mask_pad(output_logprob, response_mask),
                'response/lm_loss': lm_loss,
                'response/entropy': mask_pad(output_entropy, response_mask),
                'response/logits': logits,
                'response/masks': response_mask,
            }

