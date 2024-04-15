import os
import torch
import json
import time
import logging
import random
import argparse
import numpy as np
from typing import List
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup
from arguments import get_args
from data_pool import DataPool, DataPoolWithIndex
from utils.utils import ensure_dir, ceil_div, reduce_mean, reduce_sum, distinctness
from arguments import get_cfg, yaml_config_to_args
from models import build_policys
from models.sample_strategy import SampleStrategy
from accelerate import Accelerator
import wandb
import torch.nn.functional as F
from data_pool import MultipleDataPools, DataPool, DataPoolWithIndex, DataPoolWithIndexV2
from transformers import AutoTokenizer
from models.stats_collector import StatsCollector
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger(__name__)
import torch.distributed as dist
import pickle
from collections import Counter
def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def all_gather(data, flattern = True):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        if flattern:
            return [i for i in data]
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.LongTensor([tensor.numel()]).to("cuda")
    size_list = [torch.LongTensor([0]).to("cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to("cuda"))
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to("cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    if flattern:
        return [item for sublist in data_list for item in sublist]

    return data_list

def mute(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

class SequenceCollator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, sequences):
        queries = [sequence['query'] for sequence in sequences]
        responses = [sequence['response'] for sequence in sequences]
        cat_ids = [self.tokenizer.convert_tokens_to_ids(sequence['cat_tokens']) for sequence in sequences]

        query_encodings_dict = self.tokenizer(queries, return_tensors="pt", padding=True)
        query_input_ids = query_encodings_dict['input_ids']
        query_mask = query_encodings_dict['attention_mask']

        query_input_ids = torch.cat([query_input_ids.new(cat_ids)[:, None], query_input_ids], dim=1)
        query_mask = torch.cat([query_mask.new([1] * len(query_mask))[:, None], query_mask], dim=1)

        response_encodings_dict = self.tokenizer(responses, return_tensors="pt", padding=True)
        response_input_ids = response_encodings_dict['input_ids']
        response_mask = response_encodings_dict['attention_mask']

        return query_input_ids, query_mask, response_input_ids, response_mask

class ConditionTrainer:
    def __init__(self,
                 params: argparse.Namespace,
                 policy: object,
                 ref_policy: object,
                 score_model: object,
                 tree_tokens: List[str],
                 train_dataloader: DataLoader,
                 train_dataloader_as_val: DataLoader,
                 val_dataloader: DataLoader,
                 optimizer: Optimizer,
                 scheduler: LambdaLR,
                 cfg: dict,
                 accelerator: Accelerator = None
                 ):     
        self.params = params
        self.policy = policy
        self.ref_policy = ref_policy
        self.score_model = score_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.train_dataloader_as_val = train_dataloader_as_val
        self.writer = SummaryWriter(comment=f"target:{self.params.target_sentiment}")
        self.best_previsous_eval_acc = -1
        self.best_step = -1
        self.tree_tokens = tree_tokens
        self.best_cat = self.tree_tokens[0]
        self.best_cat_id = self.policy.tokenizer.convert_tokens_to_ids(self.best_cat)

        self.sample_dataloader, self.sampler = None, None
        self.seq_collator = SequenceCollator(tokenizer=policy.tokenizer)
        self.sample_strategy = SampleStrategy(cfg)
        self.cfg = cfg
        self.accelerator = accelerator

        if not self.cfg.DATA.PROMPT_ARGS.USE_DATASET:
            datasets_keys_mapped_to_pool = self.train_dataloader.dataset.datasets
            self.data_pool = MultipleDataPools(
                self.cfg.TRAIN.DATAPOOL_PATH,
                datasets = datasets_keys_mapped_to_pool,
                with_index=self.cfg.DATA.DATAPOOL_USE_INDEX,
                version=self.cfg.DATA.DATAPOOL_VERSION,)

        self.step_num = 0

    def add_control_code(self, input_ids, attention_mask):
        input_ids = torch.cat([input_ids.new([self.best_cat_id] * len(input_ids))[:, None], input_ids], dim=1)
        attention_mask = torch.cat([attention_mask.new([1] * len(attention_mask))[:, None], attention_mask], dim=1)
        return input_ids, attention_mask

    def decode(self, query_input_ids, response_input_ids=None):
        query = [self.policy.tokenizer.decode(p, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                 for p in query_input_ids]

        if response_input_ids is None:
            return query

        response = [self.policy.tokenizer.decode(r, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    for r in response_input_ids]
        return query, response


    def sample(self, 
            update_pool = True, 
            dataloader = None, 
            datapool_name = None, 
            sample_ref_pro=1.0,
            limit_size = -1,
            sample_rounds = 1,
            verbose=False,
            save_sample_eval_path=None,
        ):

        if dataloader is None:
            dataloader = self.train_dataloader
        
        prompts, responses_text, responses_answer, answers, rationales, sentence_probs = [], [], [], [], [], []
        token_probs, tokens = [], []
        indexes = []
        running_accuracy = []
        running_accuracy_multiround = []
        running_accuracy_majority = []
        running_accuracy_best_sampled = []
        ppl_running = []
        probability_of_correct_one_best = []
        probability_of_correct_one_majority = []

        stats_collector = StatsCollector(self.score_model)
        print("Sampling from policy with sample_ref_pro {} for pool {}".format(sample_ref_pro, datapool_name))
        for batch_i, batch in enumerate(tqdm(dataloader, total=len(dataloader),)):
            if batch_i == limit_size:
                break
            pro = random.random()
            if pro < sample_ref_pro:
                policy = self.ref_policy
            else:
                policy = self.policy
            
            rollouts = self.sample_strategy(
                policy, 
                **batch, 
                sample_rounds=sample_rounds,
                device = self.accelerator.device if self.accelerator else None)
            
            stats_collector.add_batch(rollouts, sample_rounds)

            # makes copies of the query
            current_batch_prompts = []
            current_batch_answers = []
            current_batch_indexes = []
            for i in range(len(rollouts['query/text'])):
                current_batch_prompts.extend([rollouts['query/text'][i]] * sample_rounds)
                current_batch_answers.extend([rollouts['query/answer'][i]] * sample_rounds)
                current_batch_indexes.extend([batch["indexes"][i]] * sample_rounds)
            current_batch_responses_text = rollouts['response/text']
            current_batch_responses_answer = rollouts['response/answer']
            prompts.extend(current_batch_prompts)
            responses_answer.extend(current_batch_responses_answer)
            answers.extend(current_batch_answers)
            responses_text.extend(current_batch_responses_text)
            rationales.extend(rollouts['response/rationale'])

            if "response/token_logprobs" in rollouts:
                token_probs.extend(rollouts['response/token_logprobs'])
                tokens.extend(rollouts['response/tokens'])

            # batch total scores 
            batch_size = len(current_batch_prompts) // sample_rounds
            batch_scores = self.score_model.get_reward(current_batch_prompts, current_batch_responses_answer, answers = current_batch_answers)

            # majority vote scores
            current_batch_responses_answer_majority_vote = [current_batch_responses_answer[i * sample_rounds: (i + 1) * sample_rounds] for i in range(batch_size)]
            # take the most common answer per response
            current_batch_responses_answer_majority_vote = [Counter(i).most_common(1)[0][0] for i in current_batch_responses_answer_majority_vote]
            majority_vote_scores = self.score_model.get_reward(None, current_batch_responses_answer_majority_vote, answers = rollouts['query/answer'])
            
            # multi round scores
            multiround_batch_scores = torch.LongTensor(batch_scores).view(batch_size, sample_rounds)
            multiround_batch_scores = multiround_batch_scores.sum(-1) > 0

            # filter out and print examples where the majority vote is not the same as the batch total score
            current_batch_responses_text_groupped = [current_batch_responses_text[i * sample_rounds: (i + 1) * sample_rounds] for i in range(batch_size)]
            
            if "response/normalized_sentence_probs" in rollouts:
                probs = rollouts['response/normalized_sentence_probs']
                # pick the one with the highest probability; prob is a list
                max_sentence = torch.argmax(torch.Tensor(probs)).item()
                if batch_scores[max_sentence] == 0 and majority_vote_scores[0] == 1:
                    print("\n\n\n\n\n")
                    print("Best", np.exp(-probs[max_sentence]), current_batch_responses_text_groupped[0][max_sentence])
                    for index_sentence, sentence in enumerate(current_batch_responses_text_groupped[0]):
                        print(np.exp(-probs[index_sentence]), sentence)
                    print(current_batch_prompts[0]) # questions
                    print(current_batch_answers[0]) # gold answers
                    probability_of_correct_one_majority.append(np.mean(batch_scores))
                if batch_scores[max_sentence] == 1:
                    probability_of_correct_one_best.append(np.mean(batch_scores))
                ppl_running.extend(probs)
                running_accuracy_best_sampled.append(batch_scores[max_sentence])

                sentence_probs.append(probs)
            else:
                ppl_running.append(0)
                probability_of_correct_one_best.append(0)
                probability_of_correct_one_majority.append(0)
                running_accuracy_best_sampled.append(0)
            
            running_accuracy_multiround.append(multiround_batch_scores.int().float().mean())
            running_accuracy.append(np.mean(batch_scores))
            running_accuracy_majority.append(np.mean(majority_vote_scores))
            indexes.extend(current_batch_indexes)
            
            if verbose:
                interval = 10
            else:
                interval = 50
            
            if batch_i % interval == 0:
                print("running accuracy: ", np.mean(running_accuracy))
                print("running accuracy multiround: ", np.mean(running_accuracy_multiround))
                print("running accuracy majority: ", np.mean(running_accuracy_majority))
                print("running accuracy best sampled: ", np.mean(running_accuracy_best_sampled))
                print("running ppl: ", np.exp(-np.mean(ppl_running)))
                print("prob one best: ", np.mean(probability_of_correct_one_best))
                print("prob one majority: ", np.mean(probability_of_correct_one_majority))
                print("\n\n")
                i = 0
                print(batch_i, i)
                print(current_batch_prompts[i]) # questions
                for j in current_batch_responses_text_groupped[i]:
                    print("    ", [j])
                print(current_batch_answers[i]) # gold answers
                print("\n\n")

        prompts = all_gather(tuple(prompts))
        responses_text = all_gather(tuple(responses_text))
        responses_answer = all_gather(tuple(responses_answer))
        answers = all_gather(tuple(answers))
        indexes = all_gather(tuple(indexes))
        rationales = all_gather(tuple(rationales))
        
        if save_sample_eval_path is not None:
            stats_collector.dump_to_file(save_sample_eval_path)

        # random show 3 examples 
        if self.accelerator.is_main_process:
            random_idx = np.random.choice(len(prompts), 3, replace=False)
            for i in random_idx:
                print("Prompt: ", prompts[i][-1000:], "\n\n") # only print the last 1000 characters
                print("Response: ", responses_text[i], "\n\n")
                print("Response answer: ", responses_answer[i], "\n\n")
                print("Gold answer: ", answers[i], "\n\n")

        scores = self.score_model.get_reward(prompts, responses_answer, answers = answers)
        accuracy = np.mean(scores)
        if self.accelerator.is_main_process:
            print("\n\nMean Reward : {} ".format(accuracy))
            print("Mean Reward* : {} ".format(np.mean(running_accuracy)))
            print("Mean Reward multiround : {}".format(np.mean(running_accuracy_multiround)))
            print("Mean Reward majority : {} ".format(np.mean(running_accuracy_majority)))
            print("Mean Reward best sampled : {} ".format(np.mean(running_accuracy_best_sampled)))
            print("Mean PPL : {} \n\n\n\n\n".format(np.exp(-np.mean(ppl_running))))

        if update_pool:
            if self.cfg.DATA.PROMPT_ARGS.USE_DATASET:
                from dataset.hf_datasets import SimpleDataPoolV3
                one_pool = SimpleDataPoolV3(self.cfg.TRAIN.DATAPOOL_PATH)
                one_pool.add(prompts=prompts, 
                    responses=responses_text, 
                    scores=scores, 
                    answers=answers, 
                    indexes = indexes,
                    rationales = rationales,
                    token_probs = token_probs,
                    tokens = tokens)
                one_pool.dump_to_file()
                exit()

            if self.cfg.DATA.DATAPOOL_VERSION == "v2":
                one_pool = DataPoolWithIndexV2()
                one_pool.add(
                    prompts=prompts, 
                    responses=responses_text, 
                    scores=scores, 
                    answers=answers, 
                    indexes = indexes,
                    rationales = rationales)
            elif self.cfg.DATA.DATAPOOL_USE_INDEX:
                one_pool = DataPoolWithIndex(self.tree_tokens, self.cfg.PPO.N_EXTRA_TOKENS, )
                one_pool.add(prompts=prompts, responses=responses_text, scores=scores, answers=answers, indexes = indexes)
            else:
                one_pool = DataPool(self.tree_tokens, self.cfg.PPO.N_EXTRA_TOKENS, )
                one_pool.add(prompts=prompts, responses=responses_text, scores=scores, answers=answers,)
            # def add_onepool(self, model_name, dataset_name, run_name, additional, pool):
            if self.cfg.TRAIN.PURGE_EVERYTIME:
                self.data_pool.purge()
            self.data_pool.add_onepool(
                model_name=self.cfg.MODEL.NAME,
                dataset_name=self.cfg.DATA.CONFIG,
                run_name=self.cfg.RUN_UID,
                additional=str(datapool_name),
                pool=one_pool
            )
            self.data_pool.dump_to_file()
        
        return accuracy, np.mean(running_accuracy_multiround), np.mean(running_accuracy_majority)

    def update_sampler(self):
        if self.cfg.DATA.PROMPT_ARGS.USE_DATASET:
            self.sample_dataloader = self.train_dataloader
        else:
            self.sample_dataset = self.data_pool.form_dataset(
                theshold=self.cfg.POLICY_MODEL.POOL_THRESHOLD, 
                limit_size = self.cfg.DATA.LIMIT_DATASET_SIZE,
                filter_strategy=self.cfg.DATA.FILTER_STRATEGY,)
            # print a few examples
            random_index = np.random.choice(len(self.sample_dataset), 3, replace=False)
            for i in random_index:
                print(self.sample_dataset[i], "\n")
        
            self.sample_dataloader = DataLoader(
                self.sample_dataset, batch_size=self.cfg.DATA.BATCH_SIZE,
                shuffle=True, drop_last=True, collate_fn=self.sample_dataset.collate_fn) # oops, we forgot the patch the data sampler for distributed training...
        self.sampler = iter(self.sample_dataloader)

    def step(self,):
        step_num = self.step_num
        step_started_at = time.time()
        
        if self.cfg.TRAIN.SAMPLE_INTERVAL == -1:
            if step_num == 0:
                self.update_sampler()
        elif step_num % self.cfg.TRAIN.SAMPLE_INTERVAL == 0:
            self.sample(sample_rounds = self.cfg.TRAIN.SAMPLE_ROUNDS)
        try:
            batch = next(self.sampler)
            #assert len(batch[0]) == self.params.batch_size, 'insufficient batch'
        except (StopIteration, AssertionError):
            self.sampler = iter(self.sample_dataloader)
            batch = next(self.sampler)
        
        #with accelerator.accumulate(model):
        with self.accelerator.autocast():
            ppo_loss, stats = self.loss(step_num, **batch)

        self.accelerator.backward(ppo_loss)
        if self.cfg.TRAIN.CLIP_GRAD:
            self.accelerator.clip_grad_norm_(self.policy.model.parameters(), self.params.max_grad_norm)
        
        if (step_num+1) % self.cfg.TRAIN.GRAD_ACCUMULATE_STEPS == 0:
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
       
        step_time = time.time() - step_started_at
        eps_per_second = float(self.params.batch_size) / step_time
        ppo_loss = self.accelerator.gather(ppo_loss).mean()
        wandb_log = {
                'loss': ppo_loss.item(),
            }
        
        if (step_num + 1) % self.cfg.TRAIN.REPORT_INTERVAL == 0:
            print(f"[step {step_num}] step_time={step_time:.2f}s, eps/s={eps_per_second:.2f}, name={self.cfg.TRAIN.CUSTOM_NAME}")
        
        if (step_num + 1) % (self.cfg.TRAIN.SAVE_INTERVAL * self.cfg.TRAIN.GRAD_ACCUMULATE_STEPS) == 0:
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                self.save(step="latest")

        if (step_num + 1) % (self.cfg.TRAIN.EVAL_INTERVAL * self.cfg.TRAIN.GRAD_ACCUMULATE_STEPS) == 0 or (step_num + 1) == 1000: # do a eager evaluation
            print("\n\nEvaluating the model")

            eval_accuracy, *_ = self.sample(
                update_pool=False, 
                dataloader=self.val_dataloader, sample_ref_pro=0.0, 
                limit_size=self.cfg.TRAIN.EVAL_LIMIT_SIZE)

            print("Evaluation accuracy: ", eval_accuracy)
            wandb_log['eval_accuracy'] = eval_accuracy

            if self.best_previsous_eval_acc < eval_accuracy:
                self.best_previsous_eval_acc = eval_accuracy
                self.best_step = step_num
                if self.accelerator.is_main_process:
                    self.save(step='best')
        
        if self.accelerator.is_main_process:
            if self.params.use_wandb:
                if (step_num + 1) % self.cfg.TRAIN.GRAD_ACCUMULATE_STEPS == 0:
                    wandb.log(wandb_log)
            else:
                print(wandb_log)
        

        self.step_num += 1

    def loss(self, step, prompts, responses, **kwargs):
        device = self.accelerator.device if self.accelerator else None
        outputs = self.policy.forward_pass(
            prompts,
            responses,
            device = device)
        lm_loss, logprobs, entropy, logits = outputs['response/lm_loss'], outputs['response/log_prob'], \
                                             outputs['response/entropy'], outputs['response/logits']
        return lm_loss, {}

    def record_step_stats(self, data):
        masks = data['masks']
        kl = torch.sum(self.kl_loss(F.log_softmax(data['ref_logits'], dim=-1), F.softmax(data['logits'], dim=-1)), dim=-1)
        mean_kl = torch.mean(reduce_sum(kl, masks, axis=1))
        mean_entropy = torch.mean(reduce_sum(-data['logprobs'], masks, axis=1))
        stats = {
            'objective/kl': mean_kl.item(),
            'objective/entropy': mean_entropy.item(),
        }
        stats.update({
            'loss/total': data['total_loss'].item(),
            'loss/kl': data['kl_loss'].item(),
            'loss/lm': data['lm_loss'].item(),
            'loss/entropy': data['entropy'].item(),
        })

        return stats

    def print_samples(self, queries, responses, lm_loss, logprobs, ref_logprobs, masks, step):
        if step % self.params.log_interval != 0:
            return
            # Log samples
        for i in range(min(3, len(queries))):
            sample_kl = torch.sum((logprobs[i] - ref_logprobs[i]) * masks[i]).item()
            print(queries[i] + responses[i])
            print(f"  lm_loss = {lm_loss[i].item():+.2f}")
            print(f"  kl = {sample_kl:+.2f}")
            print(f"  total = {lm_loss[i].item() + self.params.kl_coef * sample_kl:+.2f}")

    def save_only_model(self, step):
        unwrapped_model = self.accelerator.unwrap_model(self.policy.model)
        self.accelerator.save(unwrapped_model.state_dict(), f'{self.cfg.TRAIN.SAVE_DIR}/model_{step}.pth')
        log.info(f"[step {step}] model checkpoint saved to {self.cfg.TRAIN.SAVE_DIR}/model_{step}.pth")
    

    def load_only_model(self, path):
        checkpoint = torch.load(path)
        unwrapped_model = self.accelerator.unwrap_model(self.policy.model)
        unwrapped_model.load_state_dict(checkpoint)
        log.info(f"model loaded from {path}")
    
    def save(self, step):
        print("Saving model...")
        unwrapped_model = self.accelerator.unwrap_model(self.policy.model)
        torch.save(unwrapped_model.state_dict(), f'{self.cfg.TRAIN.SAVE_DIR}/model_{step}.pth')
        print(f"[step {step}] model checkpoint saved to {self.cfg.TRAIN.SAVE_DIR}/model_{step}.pth")
        torch.save(self.scheduler.state_dict(), f'{self.cfg.TRAIN.SAVE_DIR}/scheduler_{step}.pth')
        print(f"[step {step}] scheduler checkpoint saved to {self.cfg.TRAIN.SAVE_DIR}/scheduler_{step}.pth")

    def load(self, save_dir, step):
        unwrapped_model = self.accelerator.unwrap_model(self.policy.model)
        self.accelerator.load(unwrapped_model.state_dict(), f'{save_dir}/model_{step}.pth')

        self.accelerator.load(self.optimizer.state_dict(), f'{save_dir}/optimizer_{step}.pth')
        self.accelerator.load(self.scheduler.state_dict(), f'{save_dir}/scheduler_{step}.pth')
        self.step_num = step
        

    def eval(self, step):
        if step % self.params.eval_interval != 0:
            return
        log.info(f"[step {step}] evaluating ...")

        generations, perplexities, scores = [], [], []
        for i, (input_ids, attention_mask) in enumerate(tqdm(self.val_dataloader)):
            with torch.no_grad():
                input_ids, attention_mask = self.add_control_code(input_ids, attention_mask)
                rollouts = self.policy.sample(input_ids=input_ids, attention_mask=attention_mask, top_p=self.params.top_p)
                forward_inputs = {'query_input_ids': rollouts['query/input_ids'][:, 1:],
                                  'query_mask': rollouts['query/mask'][:, 1:],
                                  'response_input_ids': rollouts['response/input_ids'],
                                  'response_mask': rollouts['response/mask']}
                ref_logprobs = self.ref_policy.forward_pass(**forward_inputs)['response/log_prob']
                perplexity = -1. * reduce_sum(ref_logprobs, rollouts['response/mask'].float(), axis=1)
                perplexities.extend(perplexity.cpu().detach().numpy().tolist())

                prompt = self.decode(rollouts['query/input_ids'][:, 1:])
                response = rollouts['response/text']
                score = self.score_model.get_reward(prompt, response, f'step{step}_eval{i}')
                scores.extend(score)

                generations.extend(rollouts['response/text'])

        ppl_score, sentiment_score = np.mean(perplexities), np.mean(scores)
        dist_1, dist_2, dist_3 = distinctness(generations)
        print(f"  perplexity = {ppl_score:+.2f}")
        print(f"  {self.params.target_sentiment} sentiment = {sentiment_score:+.2f}")
        print(f'dist-1={dist_1:.3f}, dist-2={dist_2:.3f}, dist-3={dist_3:.3f}')
        self.writer.add_scalar('Evaluation/perplexity', ppl_score, step)
        self.writer.add_scalar(f'Evaluation/{self.params.target_sentiment}', sentiment_score, step)
        self.writer.add_scalar('Evaluation/Dist-1', dist_1, step)
        self.writer.add_scalar('Evaluation/Dist-2', dist_2, step)
        self.writer.add_scalar('Evaluation/Dist-3', dist_3, step)

def main():
    args = get_args()
    cfg = get_cfg(args)
    
    args = yaml_config_to_args(cfg.PPO, args) # override

    accelerator = Accelerator(
        fp16=cfg.POLICY_MODEL.ACCELERATE_FP16,
        mixed_precision=cfg.POLICY_MODEL.ACCELERATE_MIX_PREDICTION,)

    if args.use_wandb and accelerator.is_main_process:
        run = wandb.init(
            project = 'cot',
            job_type = 'train_model',
            name = cfg.TRAIN.CUSTOM_NAME
        )
        cfg.defrost()
        cfg.RUN_UID = run.id
        

        args.save_dir = f'{args.output_dir}/{cfg.RUN_UID}'
        ensure_dir(args.save_dir)

        cfg.TRAIN.SAVE_DIR = args.save_dir
        cfg.freeze()
        # save cfg
        with open(f'{args.save_dir}/cfg.yaml', 'w') as f:
            f.write(cfg.dump())
            print(cfg.dump())
        
        wandb.save(f'{args.save_dir}/cfg.yaml')

        with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
    else:
        # save cfg and args
        cfg.defrost()
        args.save_dir = f'{args.output_dir}/{cfg.RUN_UID}'
        ensure_dir(args.save_dir)
        cfg.TRAIN.SAVE_DIR = args.save_dir
        cfg.freeze()

        with open(f'{args.save_dir}/cfg.yaml', 'w') as f:
            f.write(cfg.dump())
            print(cfg.dump())
        
        with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)


    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    num_gpus = torch.cuda.device_count()
    log.info(f'Detect {num_gpus} GPUS')

    tree_tokens = [' _TREE_TOKEN_{}'.format(str(idx).zfill(5)) for idx in range(args.n_extra_tokens)] + \
                  [' _TREE_TOKEN_ZERO_COMMENTS']

    mute(accelerator.is_main_process)

    # Build Dataset
    from dataset import build_dataloader
    train_dataloader = build_dataloader(cfg, cfg.DATA.TRAIN, is_train=True)
    val_dataloader = build_dataloader(cfg, cfg.DATA.TEST, is_train=False)
    train_dataloader_as_val = build_dataloader(cfg, cfg.DATA.TRAIN, is_train=False)
    
    # Building Model
    log.info(f'Initializing models ...')
    ref_policy, policy, reward = build_policys(cfg)
    log.info(f'Initialization done!')

    if cfg.MODEL.LOAD_WITH_ACCELERATE:
        ref_policy = accelerator.prepare(ref_policy)

    # set up optimizer and scheduler
    if cfg.POLICY_MODEL.TYPE == "api":
        optimizer = None
        optimizer_total_steps = 1000
        scheduler = 0
    else:
        optimizer = Adam(policy.model.parameters(), lr=cfg.TRAIN.LR, eps=1e-5)
        optimizer_total_steps = ceil_div(cfg.TRAIN.TOTAL_STEPS, cfg.TRAIN.GRAD_ACCUMULATE_STEPS)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=optimizer_total_steps)
    
    if cfg.POLICY_MODEL.LOAD_WITH_ACCELERATE and "30b" not in cfg.POLICY_MODEL.NAME and "13b" not in cfg.POLICY_MODEL.NAME:
        policy.model, optimizer, scheduler, train_dataloader, val_dataloader, train_dataloader_as_val = accelerator.prepare(policy.model, optimizer, scheduler, train_dataloader, val_dataloader, train_dataloader_as_val)

    # print dataset size
    print(f'Train dataset size: {len(train_dataloader)}')
    print(f'Validation dataset size: {len(val_dataloader)}')
    print(f'Train dataset size as validation dataset: {len(train_dataloader_as_val)}')

    # report the sequence length
    tokenizer = AutoTokenizer.from_pretrained(cfg.POLICY_MODEL.NAME, use_fast=False)

    for dataset in train_dataloader.dataset.datasets:
        print("\nCalculating sequence length")
        length = []
        size = 2000
        random_indexes = np.random.randint(0, len(train_dataloader.dataset), size=size)
        counter = 0
        for i in random_indexes:
            index, x, y, response, *_ = train_dataloader.dataset[i]   
            input_x = tokenizer(
                    x, return_tensors="pt", padding=True,)
            length.append(input_x.input_ids.size()[1])
            if cfg.POLICY_MODEL.MAX_INPUT_LENGTH != -1 and input_x.input_ids.size()[1] > cfg.POLICY_MODEL.MAX_INPUT_LENGTH:
                counter += 1

        print("  Dataset size {}, HF name: {}".format(len(dataset), dataset.dataset_cfg.HF_IDENTIFIER))
        print(f'  Average sequence length: {np.mean(length)}')
        print(f'  Max sequence length: {np.max(length)}')
        print("  Truncate rate (sequence length > MAX_INPUT_LENGTH): {}".format(counter / size))

    trainer = ConditionTrainer(params=args, policy=policy, ref_policy=ref_policy,
                               score_model=reward, tree_tokens=tree_tokens,
                               train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                               train_dataloader_as_val = train_dataloader_as_val,
                               optimizer=optimizer, scheduler=scheduler, cfg=cfg,
                               accelerator=accelerator)
    
    if cfg.POLICY_MODEL.WEIGHT is not None:
        trainer.load_only_model(path=cfg.POLICY_MODEL.WEIGHT)
        print(f'Loaded model from {cfg.POLICY_MODEL.WEIGHT}')
    
    if cfg.TRAIN.RESUME_FROM is not None:
        trainer.load(path=cfg.TRAIN.RESUME_FROM, step = cfg.TRAIN.RESUME_FROM_STEP)
        print(f'Loaded model from {cfg.TRAIN.RESUME_FROM} at step {cfg.TRAIN.RESUME_FROM_STEP}')

    if args.do_sample_eval:
        cfg.defrost()
        cfg.TRAIN.EVAL_LIMIT_SIZE = -1
        eval_accuracy, eval_accuracy_upperbound, eval_accuracy_majority = trainer.sample(
            update_pool=args.save_eval_results, 
            dataloader=trainer.val_dataloader, sample_ref_pro=0.0, limit_size=cfg.TRAIN.EVAL_LIMIT_SIZE,
            sample_rounds = cfg.TRAIN.SAMPLE_ROUNDS,
            verbose=False,
            save_sample_eval_path=args.save_sample_eval_path)
        print(f'Eval accuracy: {eval_accuracy}, upperbound: {eval_accuracy_upperbound}, majority: {eval_accuracy_majority}')

        dir_eval = os.environ['DIR_EVAL']
        model_epoch = os.environ['MODEL_EPOCH']
        key_appendix = "acc_{}".format(cfg.DATA.CONFIG.split('/')[-1].split('.')[0])
        if cfg.TRAIN.SAMPLE_ROUNDS == 1:
            info_to_save = {
            "{}_average".format(key_appendix): float(eval_accuracy),
            }
        else:
            info_to_save = {
                "{}_average".format(key_appendix): float(eval_accuracy),
                "{}_majority".format(key_appendix): float(eval_accuracy_majority),
            }
        eval_file = os.path.join(dir_eval, "eval.json")
        if os.path.exists(eval_file):
            with open(eval_file, "r") as f:
                eval_info = json.load(f)
        else:
            eval_info = {}


        if model_epoch not in eval_info:
            eval_info[model_epoch] = {}
        eval_info[model_epoch].update(info_to_save)
    
        with open(eval_file, "w") as f:
            json.dump(eval_info, f)
        
        if args.update_wandb:
            run_name = cfg.TRAIN.CUSTOM_NAME+"_eval"

            # add beam temp info
            if cfg.POLICY_MODEL.NUM_BEAMS != 1:
                run_name += "_beam{}".format(cfg.POLICY_MODEL.NUM_BEAMS)
            if cfg.TRAIN.SAMPLE_ROUNDS != 1:
                run_name += "_round{}".format(cfg.TRAIN.SAMPLE_ROUNDS)
            run_name += "_temp{}".format(cfg.POLICY_MODEL.TEMPERATURE)
            if cfg.TRAIN.EVAL_LIMIT_SIZE != -1:
                run_name += "_limit{}".format(cfg.TRAIN.EVAL_LIMIT_SIZE)
            #run_name += "_{}".format(cfg.DATA.CONFIG.split('/')[-1].split('.')[0])

            api = wandb.Api()
            runs = api.runs('haroldli/cot_eval')
            for run in runs:
                if run.name == run_name:
                    run.delete()
            
            run = wandb.init(
                project = 'cot_eval',
                job_type = 'evaluate',
                name = run_name,
            )
            for step, info in eval_info.items():
                try:
                    step = int(step)
                except:
                    step = 0
                run.log(info, step=step)

        exit(0)
    
    for step_num in tqdm(range(0, cfg.TRAIN.TOTAL_STEPS)):
        trainer.step()
    
    if accelerator.is_main_process:
        print("Training finished")
        trainer.load_only_model(f'{trainer.cfg.TRAIN.SAVE_DIR}/model_best.pth')
        eval_accuracy, *_ = trainer.sample(
                    update_pool=False, 
                    dataloader=trainer.val_dataloader, sample_ref_pro=0.0,)
        print(f'Final Eval accuracy: {eval_accuracy}')
        wandb.log({'full_eval_best_accuracy': eval_accuracy})
        
if __name__ == "__main__":
    main()