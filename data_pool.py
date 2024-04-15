from typing import List
from copy import deepcopy
import pdb
from collections import defaultdict, namedtuple
import numpy
from torch.utils.data import Dataset, DataLoader
import json
import os
from pprint import pprint
from dataset.answer_cleaner import AnswerCleaner
import time, random
import shutil

class DataPool:
    def __init__(self, 
        tree_tokens, n_extra_tokens, min_score=0.0, max_score=1.0,
        ):
        self.tree_tokens = tree_tokens
        self.n_extra_tokens = n_extra_tokens
        self.prompt_pool, self.response_pool, self.score_pool, self.cat_tokens = [], [], [], []
        self.min_score = min_score
        self.max_score = max_score

    def add(self, prompts: List[str], responses: List[str], scores: List[float], **kwargs):
        self.prompt_pool.extend(prompts)
        self.response_pool.extend(responses)
        self.score_pool.extend(scores)

        sorted_data = sorted(zip(self.prompt_pool, self.response_pool, self.score_pool),
                             key=lambda x: x[-1], reverse=True)
        self.prompt_pool, self.response_pool, self.score_pool = [list(x) for x in list(zip(*sorted_data))]
        self.valid_indexes = list(range(len(self.prompt_pool)))

        # get the min and max from the score pool and create bins according to the score
        bin_interval = (self.max_score - self.min_score) / self.n_extra_tokens
        bins = [self.min_score + bin_interval * (i + 1) for i in range(self.n_extra_tokens)] # e.g., interval = 0.2 and n_extra_tokens = 5, bins = [0.2, 0.4, 0.6, 0.8, 1.0] 
        score_binned = numpy.digitize(self.score_pool, bins)

        self.cat_tokens = [self.tree_tokens[i] for i in score_binned]

    def filter(self, lower_threshold):
        self.valid_indexes = []
        for i in range(len(self.score_pool)):
            if self.score_pool[i] >= lower_threshold:
                self.valid_indexes.append(i)
    
    def __getitem__(self, index):
        valid_index = self.valid_indexes[index]
        result =  {
            'query': self.prompt_pool[valid_index],
            'response': self.response_pool[valid_index],
            'cat_tokens': self.cat_tokens[valid_index],
            'score': self.score_pool[valid_index]
        }
        return result
    
    def __len__(self):
        return len(self.valid_indexes)

    @classmethod
    def from_dict(cls, data_dict: dict):
        tree_tokens = data_dict['tree_tokens']
        n_extra_tokens = data_dict['n_extra_tokens']
        min_score = data_dict['min_score']
        max_score = data_dict['max_score']
        data_pool = cls(tree_tokens, n_extra_tokens, min_score, max_score)
        data_pool.add(data_dict['prompts'], data_dict['responses'], data_dict['scores'])
        return data_pool
    
    def to_dict(self):
        return {
            'tree_tokens': self.tree_tokens,
            'n_extra_tokens': self.n_extra_tokens,
            'min_score': self.min_score,
            'max_score': self.max_score,
            'prompts': self.prompt_pool,
            'responses': self.response_pool,
            'scores': self.score_pool,
            'cat_tokens': self.cat_tokens
        }

    def get_rationale_and_answer(self, index):
        prompt = self.prompt_pool[index]
        response = self.response_pool[index]
        cat_tokens = self.cat_tokens[index]
        score = self.score_pool[index]
        return prompt, response, cat_tokens, score




class DataPoolWithIndex:
    def __init__(self, 
            tree_tokens, 
            n_extra_tokens, 
            min_score=0.0, 
            max_score=1.0,
            use_index = False,
        ):
        self.tree_tokens = tree_tokens
        self.n_extra_tokens = n_extra_tokens
        self.prompt_pool, self.response_pool, self.score_pool, self.cat_tokens, self.index_pool = [], [], [], [], []
        self.min_score = min_score
        self.max_score = max_score

    def add(self, prompts: List[str], responses: List[str], scores: List[float], indexes: List[int], **kwargs):
        self.prompt_pool.extend(prompts)
        self.response_pool.extend(responses)
        self.score_pool.extend(scores)
        self.index_pool.extend(indexes)

        sorted_data = sorted(zip(self.prompt_pool, self.response_pool, self.score_pool, self.index_pool),
                             key=lambda x: x[-1], reverse=True)
        self.prompt_pool, self.response_pool, self.score_pool, self.index_pool = [list(x) for x in list(zip(*sorted_data))]

        self.valid_indexes = [i for i in range(len(self.score_pool))]

        # get the min and max from the score pool and create bins according to the score
        bin_interval = (self.max_score - self.min_score) / self.n_extra_tokens
        bins = [self.min_score + bin_interval * (i + 1) for i in range(self.n_extra_tokens)] # e.g., interval = 0.2 and n_extra_tokens = 5, bins = [0.2, 0.4, 0.6, 0.8, 1.0] 
        score_binned = numpy.digitize(self.score_pool, bins)

        # this assumes uniform distribution?
        cat_pos = [[i] * (len(sorted_data) // self.n_extra_tokens) for i in range(self.n_extra_tokens)]
        cat_pos = [y for x in cat_pos for y in x]
        cat_pos = cat_pos + [self.n_extra_tokens - 1] * (len(sorted_data) - len(cat_pos))

        self.cat_tokens = [self.tree_tokens[i] for i in score_binned]

    def advanced_filter(self, lower_threshold, strategy_name):
        # strategy 1: one correct path per example
        if strategy_name == 'one_correct_path':
            self.valid_indexes = []
            index_filled = set()

            for i in range(len(self.score_pool)):
                original_dataset_index = self.index_pool[i]
                if original_dataset_index in index_filled:
                    continue
                if self.score_pool[i] >= lower_threshold:
                    self.valid_indexes.append(i)
                    index_filled.add(original_dataset_index)

        # stratey 2: all correct paths per example
        elif strategy_name == 'all_correct_paths':
            self.valid_indexes = []
            for i in range(len(self.score_pool)):
                if self.score_pool[i] >= lower_threshold:
                    self.valid_indexes.append(i)
    
    def __getitem__(self, index):
        valid_index = self.valid_indexes[index]
        true_dataset_index = self.index_pool[valid_index] # this is the index of the original dataset
        _, x, y = self.dataset[true_dataset_index]
        result =  {
            'query': x,
            'response': self.response_pool[valid_index],
            'cat_tokens': self.cat_tokens[valid_index],
            'score': self.score_pool[valid_index],
            "indexes": self.index_pool[valid_index]
        }
        return result
    
    def __len__(self):
        return len(self.valid_indexes)

    def get_indexes_mapping(self):
        index_groups = defaultdict(list) # original_dataset_index -> xxx
        for index in range(len(self.index_pool)):
            index_groups[self.index_pool[index]].append(index)
        return index_groups
    
    def get_data_by_original_dataset_index(self, index_groups, original_index, cap_token = 500):
        return_results = []
        for inspect_index in index_groups[original_index]:
            result =  {
                'query': self.prompt_pool[inspect_index][-cap_token:],
                'response': self.response_pool[inspect_index],
                'score': self.score_pool[inspect_index]
            }
            return_results.append(result)
        return return_results

    @classmethod
    def from_dict(cls, data_dict: dict):
        tree_tokens = data_dict['tree_tokens']
        n_extra_tokens = data_dict['n_extra_tokens']
        min_score = data_dict['min_score']
        max_score = data_dict['max_score']
        data_pool = cls(tree_tokens, n_extra_tokens, min_score, max_score)
        data_pool.add(data_dict['prompts'], data_dict['responses'], data_dict['scores'], data_dict['indexes'])
        return data_pool
    
    def to_dict(self):
        return {
            'tree_tokens': self.tree_tokens,
            'n_extra_tokens': self.n_extra_tokens,
            'min_score': self.min_score,
            'max_score': self.max_score,
            'prompts': self.prompt_pool,
            'responses': self.response_pool,
            'scores': self.score_pool,
            'cat_tokens': self.cat_tokens,
            'indexes': self.index_pool
        }
    
    def parse_to_rationale(self):
        pass


class DataPoolWithIndexV2:
    def __init__(self, 
        ):
        self.examples = []

    def add(self, prompts: List[str], responses: List[str], scores: List[float], indexes: List[int], rationales, **kwargs):
        for i in range(len(prompts)):
            self.examples.append({
                'prompt': prompts[i],
                'response': responses[i],
                'score': scores[i],
                'index': indexes[i],
                'rationale': rationales[i]
            })
        self.valid_indexes = [i for i in range(len(self.examples))]

    def advanced_filter(self, lower_threshold, strategy_name):
        self.strategy_name = strategy_name
        # strategy 1: one correct path per example
        if strategy_name == 'one_correct_path' or strategy_name == "one_correct_path_no_rationel" or strategy_name == "all_correct_paths_balanced":
            self.valid_indexes = []
            self.index_filled = set()

            for i in range(len(self.examples)):
                original_dataset_index = self.examples[i]["index"]
                if original_dataset_index in self.index_filled:
                    continue
                if self.examples[i]["score"] >= lower_threshold:
                    self.valid_indexes.append(i)
                    self.index_filled.add(original_dataset_index)

        # stratey 2: all correct paths per example
        elif strategy_name == 'all_correct_paths':
            self.valid_indexes = []
            for i in range(len(self.examples)):
                if self.examples[i]["score"] >= lower_threshold:
                    self.valid_indexes.append(i)
        self.get_indexes_mapping()

    def get_indexes_mapping(self):
        self.index_groups = defaultdict(list) # original_dataset_index -> current dataset index
        for index in range(len(self.examples)):
            self.index_groups[self.examples[index]["index"]].append(index)
    
    def __getitem__(self, index):
        valid_index = self.valid_indexes[index]
        example = self.examples[valid_index]

        true_dataset_index = example['index'] # this is the index of the original dataset
        _, x, y = self.dataset[true_dataset_index]

        if self.strategy_name == 'one_correct_path_no_rationel':
            response = y + "\n\n" # a bit of a hack to mark the end of generation
        elif self.strategy_name == 'all_correct_paths_balanced':
            good_indexes = self.index_groups[true_dataset_index]
            good_indexes = [i for i in good_indexes if self.examples[i]["score"] >= 0.5]
            assert(valid_index in good_indexes)
            # reset random seed with time
            random.seed(time.time())
            good_index = random.choice(good_indexes)
            response = self.examples[good_index]["response"]
        else:
            response = example["response"]
        
        # if we used prompt source; we need to override the response
        if self.dataset.dataset_cfg.USE_PROMPT_SOURCE:
            rationale = example["rationale"]
            response = rationale + self.dataset.dataset_cfg.ANSWER_PREFIX + y

        result =  {
            'query': x,
            'response': response,
            'score': example["score"],
            "index": example["index"],
            "rationale": example["rationale"]
        }
        return result
    
    def __len__(self):
        return len(self.valid_indexes)

    @classmethod
    def from_dict(cls, data_dict: dict):
        data_pool = cls()
        data_pool.examples = data_dict
        return data_pool
    
    def to_dict(self):
        return self.examples


DataPoolID = namedtuple("DataPoolID", ['model_name', 'dataset_name', 'run_name', 'additional'])
class MultipleDataPools():
    def __init__(self, file_name = None, datasets = None, with_index = False, version = "v1"):
        # we will maintain this pool across runs
        if file_name is None:
            print("Not using data pools")
            return
        self.pools = {}
        self.file_name = file_name
        self.with_index = with_index
        self.version = version
        self.load_from_file(datasets)

    def load_from_file(self, datasets=None):
        if not os.path.exists(self.file_name):
            return
        with open(self.file_name, 'r') as f:
            pools = json.load(f)
        keys = list(pools.keys())
        # sort
        keys.sort()
        for index, key in enumerate(keys):
            if self.version == "v2":
                self.pools[key] = DataPoolWithIndexV2.from_dict(pools[key])
                self.pools[key].dataset = datasets[index]
            elif self.with_index:
                self.pools[key] = DataPoolWithIndex.from_dict(pools[key])
                self.pools[key].dataset = datasets[index]
            else:
                self.pools[key] = DataPool.from_dict(pools[key])
    
    def form_dataset(self, theshold, limit_size = -1, filter_strategy = "all_correct_paths", **kwargs):
        pools = []
        for pool_id in self.pools:
            pool_id_tuple = self.id_str_to_tuple(pool_id)
            if all([getattr(pool_id_tuple, key) == kwargs[key] for key in kwargs]):
                pools.append(self.pools[pool_id])
                print("Using pool {} from {}".format(pool_id, self.file_name))

        self.total_length = 0
        self.lengths_record = []
        for pool in pools:
            pool.advanced_filter(theshold, filter_strategy)
            self.total_length += len(pool)
            self.lengths_record.append(self.total_length)
        self.data_pools_in_use = pools
        if limit_size > 0:
            self.total_length = limit_size
        print("\nDatapool final total length for training:\n   {}".format(self.total_length))
        return self

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        # determine which pool the idx belongs to
        for i, length in enumerate(self.lengths_record):
            if idx < length:
                pool = self.data_pools_in_use[i]
                correct_index = idx - (self.lengths_record[i-1] if i != 0 else 0)
                break
        # get the data from the pool
        return pool[correct_index]
    @staticmethod
    def collate_fn(batch):
        queries = [seq['query'] for seq in batch]
        responses = [seq['response'] for seq in batch]
        # cat_tokens = [seq['cat_tokens'] for seq in batch]
        scores = [seq['score'] for seq in batch]
        return queries, responses, scores

    @staticmethod
    def id_str_to_tuple(id_str):
        model_name, dataset_name, run_name, additional = id_str.split("*")
        return DataPoolID(model_name, dataset_name, run_name, additional)
    
    @staticmethod
    def tuple_to_id_str(pool_id):
        return "*".join([pool_id.model_name, pool_id.dataset_name, pool_id.run_name, pool_id.additional])

    def add_onepool(self, model_name, dataset_name, run_name, additional, pool):
        pool_id = "*".join([model_name, dataset_name, str(run_name), additional])
        self.pools[pool_id] = pool
    
    def check_pool_exists(self, model_name, dataset_name, run_name, additional):
        pool_id = "*".join([model_name, dataset_name, str(run_name), additional])
        return pool_id in self.pools
    
    def purge(self, **kwargs):
        assert("test" in self.file_name) # only purge test data pools
        pool_ids = list(self.pools.keys())
        for pool_id in pool_ids:
            pool_id_tuple = self.id_str_to_tuple(pool_id)
            if all([getattr(pool_id_tuple, key) == kwargs[key] for key in kwargs]): # get rid of the pool
                del self.pools[pool_id]

    def dump_to_file(self):
        # turn every pool into dict
        if os.path.exists(self.file_name):
            # make a copy just incase 
            shutil.copy(self.file_name, self.file_name + ".bak")
        pools = {}
        for key in self.pools:
            pools[key] = self.pools[key].to_dict()
        with open(self.file_name, 'w') as f:
            json.dump(pools, f)
