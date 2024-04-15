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
from typing import List
from copy import deepcopy
import pdb
from collections import defaultdict, namedtuple
import numpy
from torch.utils.data import Dataset, DataLoader
import json
import os
import time
from pprint import pprint
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    all_ = []
    for i in range(0, len(lst), n):
        data_index = lst[i:i + n]
        all_.append(data_index)
    counter = 0
    for i in all_:
        counter += len(i)
    assert(counter == len(lst))

    return all_

class SimpleDataPoolV3:
    def __init__(self, 
        file_name,
        ):
        self.file_name = file_name
        self.examples = []
        if os.path.exists(file_name):
            self.load_from_file()

    def add(self, prompts: List[str], responses: List[str], scores: List[float], indexes: List[int], rationales, **kwargs):
        for i in range(len(prompts)):
            one_example = {'prompt': prompts[i],
                'response': responses[i],
                'score': scores[i],
                'index': indexes[i],
                'rationale': rationales[i],}
            if "token_probs" in kwargs:
                one_example["token_probs"] = kwargs["token_probs"][i]
            if "tokens" in kwargs:
                one_example["tokens"] = kwargs["tokens"][i]
            self.examples.append(one_example)
        self.valid_indexes = [i for i in range(len(self.examples))]

    def get_rationale_data(self):
        rationales_by_index = defaultdict(list)
        for example in self.examples:
            rationales_by_index[example["index"]].append(
                    example
            )
        return rationales_by_index

    def __len__(self):
        return len(self.valid_indexes)
    
    def dump_to_file(self):
        with open(self.file_name, 'w') as f:
            json.dump(self.examples, f)
    
    def load_from_file(self):
        with open(self.file_name, 'r') as f:
            self.examples = json.load(f)

class CoTDataset(Dataset):
    def __init__(self, 
        cfg, split, dataset_cfg, datapool_path = None, 
        is_train = False):
        self.cfg = cfg
        self.dataset_cfg = dataset_cfg

        # find the corresponding split
        self.split_name_str = getattr(self.dataset_cfg, split, split) # if not specifed in dataset_cfg, we use split as the name
        self.is_train = is_train
        if self.is_train and self.cfg.DATA.PROMPT_ARGS.LOAD_RATIONALES:
            self.rationale_datapool = SimpleDataPoolV3(datapool_path)
            self.rationales = self.rationale_datapool.get_rationale_data()

        if self.dataset_cfg.USE_PROMPT_SOURCE:
            self.prompt_templates = DatasetTemplates(f"{self.dataset_cfg.HF_IDENTIFIER}/{self.dataset_cfg.HF_NAME}").templates
            if len(self.prompt_templates) == 0:
                self.prompt_templates = DatasetTemplates(f"{self.dataset_cfg.HF_IDENTIFIER}").templates
            
            if len(self.prompt_templates) == 0:
                print("No prompt templates found")
                assert(0)
            # get all templates and filter
            self.used_templates = []
            for key in self.prompt_templates:
                template = self.prompt_templates[key]
                if template.name in self.dataset_cfg.PROMPT_SOURCE_BAD_TEMPLATE:
                    print("Template {} is not used.".format(template.name))
                    continue
                self.used_templates.append(template)
            print("Using {} templates".format(len(self.used_templates)))
            print("Use templates: {}".format([template.name for template in self.used_templates]))
            self.hf_dataset = load_dataset(path=self.dataset_cfg.HF_IDENTIFIER, name=self.dataset_cfg.HF_NAME, split=self.split_name_str)


    def limit_size(self):
        if self.is_train and self.cfg.DATA.PROMPT_ARGS.LOAD_RATIONALES: # only limit for training set
            self.valid_indexes = self.advanced_filter(
                lower_threshold=self.cfg.POLICY_MODEL.POOL_THRESHOLD, 
                strategy_name=self.cfg.DATA.FILTER_STRATEGY,)
        else:
            self.valid_indexes = [i for i in range(len(self.dataset))]

        if self.is_train and self.cfg.DATA.LIMIT_WITH_JSON_FILE is not None:
            # if the file does not exist, we will create it
            if os.path.exists(self.cfg.DATA.LIMIT_WITH_JSON_FILE):
                with open(self.cfg.DATA.LIMIT_WITH_JSON_FILE, 'r') as f:
                    limit_indexes = json.load(f)
                self.valid_indexes = [i for i in self.valid_indexes if i in limit_indexes]
                print("Limit dataset size to {}".format(len(self.valid_indexes)))
            else:
                print("The limit json file does not exist. We will create it after the dataset is loaded.")
                all_indexes = [i for i in range(len(self.dataset))]
                random.shuffle(all_indexes)
                # hack
                if "0.1" in self.cfg.DATA.LIMIT_WITH_JSON_FILE:
                    limit_indexes = all_indexes[:int(len(all_indexes) * 0.1)]
                elif "0.2" in self.cfg.DATA.LIMIT_WITH_JSON_FILE:
                    limit_indexes = all_indexes[:int(len(all_indexes) * 0.2)]
                elif "0.3" in self.cfg.DATA.LIMIT_WITH_JSON_FILE:
                    limit_indexes = all_indexes[:int(len(all_indexes) * 0.3)]
                elif "0.4" in self.cfg.DATA.LIMIT_WITH_JSON_FILE:
                    limit_indexes = all_indexes[:int(len(all_indexes) * 0.4)]
                elif "0.5" in self.cfg.DATA.LIMIT_WITH_JSON_FILE:
                    limit_indexes = all_indexes[:int(len(all_indexes) * 0.5)]
                elif "0.6" in self.cfg.DATA.LIMIT_WITH_JSON_FILE:
                    limit_indexes = all_indexes[:int(len(all_indexes) * 0.6)]
                elif "0.7" in self.cfg.DATA.LIMIT_WITH_JSON_FILE:
                    limit_indexes = all_indexes[:int(len(all_indexes) * 0.7)]
                elif "0.8" in self.cfg.DATA.LIMIT_WITH_JSON_FILE:
                    limit_indexes = all_indexes[:int(len(all_indexes) * 0.8)]
                elif "0.9" in self.cfg.DATA.LIMIT_WITH_JSON_FILE:
                    limit_indexes = all_indexes[:int(len(all_indexes) * 0.9)]
                else:
                    assert(0)
                with open(self.cfg.DATA.LIMIT_WITH_JSON_FILE, 'w') as f:
                    json.dump(limit_indexes, f)
                exit(0)

        if self.cfg.DATA.LIMIT_DATASET_SIZE != -1:
            if self.cfg.DATA.SHUFFLE_BEFORE_LIMIT:
                random.seed(0)
                random.shuffle(self.valid_indexes)
            self.valid_indexes = self.valid_indexes[self.cfg.DATA.START_INDEX : self.cfg.DATA.START_INDEX + self.cfg.DATA.LIMIT_DATASET_SIZE]
            print("Limit dataset size to {}".format(len(self.valid_indexes)))
            print("Start index is {}".format(self.cfg.DATA.START_INDEX))

        if self.cfg.DATA.SHUFFLE_AFTER_LIMIT:
            random.seed(0)
            random.shuffle(self.valid_indexes)
        
        self.start_index = self.cfg.DATA.START_INDEX

        self.print_meta_info()


        # try to get the prompt index
        if self.dataset_cfg.USE_PROMPT_SOURCE and len(self.dataset_cfg.FEW_SHOT_PROMPT_INDEX) == 0:
            prompts = chunks(self.dataset_cfg.FEW_SHOT_COT_PROMPT, 3)
            few_shot_prompt_index = []
            for i, (x_prompt, steps_prompt, y_prompt) in enumerate(prompts):
                index = self.check_index(x_prompt)
                if index is not None:
                    few_shot_prompt_index.append(index)
                else:
                    print("Cannot find the prompt index for {} {}.".format(i, x_prompt))
                    #assert(0)
            self.dataset_cfg.FEW_SHOT_PROMPT_INDEX = tuple(few_shot_prompt_index)
            print("Few shot prompt index is {}".format(self.dataset_cfg.FEW_SHOT_PROMPT_INDEX))
            assert(0)

    def advanced_filter(self, lower_threshold, strategy_name):
        self.strategy_name = strategy_name
        self.valid_indexes = []
        all_indexes = list(self.rationales.keys())
        # sort
        all_indexes.sort()
        if strategy_name == 'one_correct_path' or strategy_name == "one_correct_path_no_rationel" or strategy_name == "all_correct_paths_balanced":
            for i in all_indexes:
                valid_ones_i = [j for j in self.rationales[i] if j['score'] >= lower_threshold]
                if len(valid_ones_i) > 0:
                    self.valid_indexes.append(i)
                self.rationales[i] = valid_ones_i
        elif strategy_name == 'no_rationale_keep_all':
            for i in all_indexes: # keep all the rationales that are sampled
                self.valid_indexes.append(i)
        elif strategy_name == "all_correct_paths":
            # modify the rationales
            counter = 0
            new_rationales = {}
            new_dataset = []
            for i in all_indexes:
                valid_ones_i = [j for j in self.rationales[i] if j['score'] >= lower_threshold]
                for j in valid_ones_i:
                    self.valid_indexes.append(counter)
                    new_rationales[counter] = [j]
                    new_dataset.append(self.dataset[i])
                    counter += 1
            self.dataset = new_dataset
            self.rationales = new_rationales
        elif strategy_name == "wrong_ones":
            # modify the rationales
            counter = 0
            new_rationales = {}
            new_dataset = []
            for i in all_indexes:
                valid_ones_i = [j for j in self.rationales[i] if j['score'] <= 0.0]
                for j in valid_ones_i:
                    self.valid_indexes.append(counter)
                    new_rationales[counter] = [j]
                    new_dataset.append(self.dataset[i])
                    counter += 1
            self.dataset = new_dataset
            self.rationales = new_rationales
        return self.valid_indexes

    def __len__(self):
        return len(self.valid_indexes)

    def get_raw_data(self, index, template = None):
        if self.dataset_cfg.USE_PROMPT_SOURCE:
            example = self.hf_dataset[index]
            # random select a template
            assert(template is not None)
            example = template.apply(example)
            x = example[0]
            y = example[1]
            rationale = ''
        else:
            x, y, *_ = self.dataset[index]
        if self.cfg.DATA.PROMPT_ARGS.PREDICT_RATIONALE and self.is_train:
            if self.cfg.DATA.FILTER_STRATEGY == "one_correct_path":
                rationale_selected = self.rationales[index][0]
            elif self.cfg.DATA.FILTER_STRATEGY == "all_correct_paths_balanced":
                random.seed(time.time())
                rationale_selected = random.choice(self.rationales[index])
            elif self.cfg.DATA.FILTER_STRATEGY == "all_correct_paths" or self.cfg.DATA.FILTER_STRATEGY == "wrong_ones":
                assert(len(self.rationales[index]) == 1)
                rationale_selected = self.rationales[index][0]
            assert(rationale_selected['score'] >= self.cfg.POLICY_MODEL.POOL_THRESHOLD)
            rationale = rationale_selected['rationale']
        else:
            rationale = ""
        return x, rationale.strip("\n"), y
    
    
    def check_index(self, example_question):
        for i in range(len(self.dataset)):
            x, y, *_ = self.dataset[i]
            if example_question in x:
                return i
        print("On first try, cannot find the example question {} ".format(example_question))

        for i in range(len(self.dataset)):
            x, y, *_ = self.dataset[i]
            if example_question[:30] == x[:30]:
                print("Match", i, x)
            if "Conceptually cream skimming has two basic dimensions" in x:
                print("Match", i, x)

        pdb.set_trace()
        return None

    def __getprediction__(self, prediction):
        while prediction.startswith("\n"):
            prediction = prediction[1:]
            
        prediction = prediction.split(self.dataset_cfg.EXAMPLE_SUFFIX)[0]
        prediction = prediction.split(self.dataset_cfg.ANSWER_PREFIX)[-1]
        prediction = prediction.strip("\n")
        if len(prediction) != 0 and prediction[0] == " ":
            prediction = prediction[1:]
        return prediction
    
    def __getitem__(self, original_index, rationale = None):
        index = self.valid_indexes[original_index]
        # choose a template
        if self.dataset_cfg.USE_PROMPT_SOURCE:
            if self.dataset_cfg.PROMPT_SOURCE_INDEX is None:
                template = random.choice(self.used_templates)
            else:
                template = self.used_templates[self.dataset_cfg.PROMPT_SOURCE_INDEX]
        else:
            template = None
        x, rationale, y = self.get_raw_data(index, template)
        
        output_x = ""
        if self.cfg.DATA.FEW_SHOT_COT_PROMPT:
            prompts = chunks(self.dataset_cfg.FEW_SHOT_COT_PROMPT, 3)
            # replace the prompts with index if we are using prompt source
            if self.dataset_cfg.USE_PROMPT_SOURCE:
                prompt_rationales = [i[1] for i in prompts]
                if len(self.dataset_cfg.FEW_SHOT_PROMPT_ORIGINAL) == 0:
                    # should not be used anymore
                    new_prompts = []
                    for promt_i, prompt in enumerate(self.dataset_cfg.FEW_SHOT_PROMPT_INDEX):
                        x_i, rationale_i, y_i = self.get_raw_data(prompt, template)
                        new_prompts.append((x_i, prompt_rationales[promt_i], y_i))
                    prompts = new_prompts
                else:
                    incontext_examples = self.convert_original_to_examples(self.dataset_cfg.FEW_SHOT_PROMPT_ORIGINAL)
                    new_prompts = []
                    for promt_i, example in enumerate(incontext_examples):
                        example = template.apply(example)
                        new_prompts.append((example[0], prompt_rationales[promt_i], example[1]))
                    prompts = new_prompts
                
            if self.cfg.DATA.SHUFFLE_COT_PROMPTS:
                random.shuffle(prompts)
            if self.cfg.DATA.RANDOM_DROP_COT_PROMPTS:
                # drop some prompts
                if self.cfg.DATA.RANDOM_DROP_COT_PROMPTS_MIN_NUM < len(prompts):
                    num_prompt = np.random.randint(self.cfg.DATA.RANDOM_DROP_COT_PROMPTS_MIN_NUM, len(prompts) + 1)
                    prompts = prompts[:num_prompt]
            if self.cfg.DATA.PROMPT_ARGS.LIMIT_PROMPT_NUM:
                prompts = prompts[:self.dataset_cfg.RANDOM_DROP_COT_PROMPTS_MAX_NUM] # this is dataset specific

            if self.cfg.DATA.PROMPT_ARGS.USE_RANDOM_TRAIN_PROMPT and self.is_train:
                new_prompts = []
                for i in range(len(prompts)):
                    new_random_index = random.choice(self.valid_indexes)
                    new_prompts.append(self.get_raw_data(index=new_random_index, template=template))
                prompts = new_prompts

            for x_prompt, steps_prompt, y_prompt in prompts:
                if self.cfg.DATA.PROMPT_ARGS.PROMPT_RATIONALE:
                    output_x += self.dataset_cfg.INPUT_PREFIX + x_prompt \
                    + self.dataset_cfg.STEP_PREFIX + steps_prompt + self.dataset_cfg.ANSWER_PREFIX + y_prompt + self.dataset_cfg.EXAMPLE_SUFFIX
                else:
                    output_x += self.dataset_cfg.INPUT_PREFIX + x_prompt \
                    + self.dataset_cfg.STEP_PREFIX + y_prompt + self.dataset_cfg.EXAMPLE_SUFFIX
            output_x += self.dataset_cfg.INPUT_PREFIX + x + self.dataset_cfg.STEP_PREFIX # each X is 
        else:
            output_x = self.dataset_cfg.INPUT_PREFIX + x + self.dataset_cfg.STEP_PREFIX

        if self.cfg.DATA.PROMPT_ARGS.PREDICT_RATIONALE and self.is_train:
            if self.cfg.DATA.FILTER_STRATEGY == "one_correct_path":
                rationale_selected = self.rationales[index][0]
            elif self.cfg.DATA.FILTER_STRATEGY == "all_correct_paths_balanced":
                #random.seed(time.time())
                rationale_selected = random.choice(self.rationales[index])
            elif self.cfg.DATA.FILTER_STRATEGY == "all_correct_paths" or self.cfg.DATA.FILTER_STRATEGY == "wrong_ones":
                assert(len(self.rationales[index]) == 1)
                rationale_selected = self.rationales[index][0]
            assert(rationale_selected['score'] >= self.cfg.POLICY_MODEL.POOL_THRESHOLD)
            rationale = rationale_selected['rationale']
            prediction = self.__getprediction__(rationale_selected["response"])[:10]
            response = "{}{}{}{}".format(rationale.strip("\n"), self.dataset_cfg.ANSWER_PREFIX, prediction, self.dataset_cfg.EXAMPLE_SUFFIX)
            if self.cfg.DATA.PROMPT_ARGS.DROP_RATIONALE:
                response = "{}{}".format(prediction, self.dataset_cfg.EXAMPLE_SUFFIX)
        else:
            response = "{}{}".format(y, self.dataset_cfg.EXAMPLE_SUFFIX)
        # will return both response and answer for this new version
        return index, output_x, y, response

    def print_meta_info(self):
        # print the split
        print("\nSplit : ", self.split_name_str)
        # print the dataset name
        print("Dataset name : ", self.dataset_cfg.HF_IDENTIFIER)
        # print the dataset length
        print("Dataset length : {}; Original Length : {}".format(len(self), len(self.dataset)))
        # print a few examples
        if len(self) > 0:
            if self.is_train:
                number_example = 3
            else:
                number_example = 1
            for i in range(number_example):
                random_index = random.randint(0, len(self)-1)
                index, x, y, answer = self[random_index]
                print("{}-th example : ".format(index))
                print("Input : ", x[-500:])
                print("Answer : ", y)
                print("Response : ", answer)
                print("\n")


class GSM8KHFDataset(CoTDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset = load_dataset(path=self.dataset_cfg.HF_IDENTIFIER, name=self.dataset_cfg.HF_NAME, split="train" if self.split_name_str == "train" else "test")
        print("Warning using test for GSM8KHFDataset; need to change later")

        # dataset_fields: ['question', 'answer']
        new_dataset = []
        answers = self.dataset["answer"]
        questions = self.dataset["question"]
        for i in tqdm(range(len(self.dataset["question"]))):
            # extract the numerial number 
            answer = answers[i].split("#### ")[-1]

            _ = (questions[i], answer)
            new_dataset.append(_)
        #pdb.set_trace()
        self.dataset = new_dataset
        self.limit_size()    

class AquaDataset(CoTDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset = load_dataset(path=self.dataset_cfg.HF_IDENTIFIER, name=self.dataset_cfg.HF_NAME, split=self.split_name_str)

        # dataset_fields: ['question', 'answer']
        new_dataset = []
        answers = self.dataset["correct"]
        questions = self.dataset["question"]
        choices = self.dataset["options"]
        for i in tqdm(range(len(self.dataset["question"]))):
            choices_i = []
            for j in choices[i]:
                j = "(" + j
                j = j.replace(")", ") ").lower()
                choices_i.append(j)
            choices_label_text = "\n".join(choices_i).lower()

            question = questions[i] + "\nAnswer Choices:\n" + choices_label_text

            answer = "({})".format(answers[i].lower())

            _ = (question, answer)
            new_dataset.append(_)

        self.dataset = new_dataset

        # print one example
        print(self.dataset[0])
        self.limit_size()
    

class SVAMPDataset(CoTDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        local_path = "{}/{}.json".format(self.dataset_cfg.LOCAL_PATH, self.split_name_str)
        with open(local_path, "r") as f:
            data = json.load(f)

        '''
        "ID": "chal-953",
        "Body": "After eating a hearty meal they went to see the Buckingham palace. There, Rachel learned that 132 visitors came to the Buckingham palace that day. If 406 people visited the Buckingham palace within the past 327 days",
        "Question": "How many visitors visited the Buckingham palace on the previous day?",
        "Equation": "( 406.0 - 132.0 )",
        "Answer": 274.0,
        "Type": "Subtraction"
        '''
        
        new_dataset = []
        for i in tqdm(range(len(data))):
            question = data[i]["Body"] + ". " + data[i]["Question"]
            new_dataset.append((question, str(data[i]["Answer"])))

        self.dataset = new_dataset

        # print one example
        print(self.dataset[0])

        self.limit_size()   
    
    def __len__(self):
        return len(self.dataset)
    

class ASDIVDataset(CoTDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.dataset_cfg = dataset_cfg
        #self.dataset_cfg.merge_from_file(self.cfg.DATA.CONFIG)

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
        for i in tqdm(range(len(data))):
            question = data[i]["Body"] + " " + data[i]["Question"]
            new_dataset.append((question, data[i]["Answer"].split(" ")[0]))

        self.dataset = new_dataset

        # print one example
        print(self.dataset[0])

        if self.cfg.DATA.LIMIT_DATASET_SIZE != 0:
            self.dataset = self.dataset[:self.cfg.DATA.LIMIT_DATASET_SIZE]
    
    def __len__(self):
        return len(self.dataset)
        
class DropDataset(CoTDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("Warning: drop dataset not complete")
        self.dataset = load_dataset(path=self.dataset_cfg.HF_IDENTIFIER, name=self.dataset_cfg.HF_NAME, split=self.split_name_str)

        # dataset_fields: ['question', 'answer']
        new_dataset = []
        passages = self.dataset["passage"]
        questions  = self.dataset["question"]
        answers_spans = self.dataset["answers_spans"]

        for i in tqdm(range(len(passages))):
            question = "{} {}".format(passages[i], questions[i])
            _ = (question, answers_spans[i]['spans'][0]) #" and ".join(answers_spans[i]['spans']))
            new_dataset.append(_)

        self.dataset = new_dataset
        self.limit_size()

class TableFactDataset(CoTDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset = load_dataset(path=self.dataset_cfg.HF_IDENTIFIER, name=self.dataset_cfg.HF_NAME, split=self.split_name_str)

        # dataset_fields: ['question', 'answer']
        new_dataset = []
        tabel_texts = self.dataset["table_text"]
        tabel_captions = self.dataset["table_caption"]
        statements = self.dataset["statement"]
        labels = self.dataset["label"]

        for i in tqdm(range(len(tabel_texts))):
            table = "Table caption: {}\nTable Contents:\n {}".format(tabel_captions[i], tabel_texts[i].replace("#", "\t"))
            question = "{}Is the following statement correct ? {}".format(table, statements[i])
            label_word = "Yes" if labels[i] == 1 else "No"
    
            _ = (question, label_word)
            new_dataset.append(_)

        self.dataset = new_dataset
        self.limit_size()

class PIQADataset(CoTDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset = load_dataset(path=self.dataset_cfg.HF_IDENTIFIER, name=self.dataset_cfg.HF_NAME, split=self.split_name_str)

        # dataset_fields: ['question', 'answer']
        new_dataset = []
        
        goals = self.dataset["goal"]
        solution_1 = self.dataset["sol1"]
        solution_2 = self.dataset["sol2"]
        labels = self.dataset["label"]

        for i in tqdm(range(len(goals))):
            
            question = "Goal : {}\nSolution (a): {}\nSolution (b): {}".format(goals[i], solution_1[i], solution_2[i])
            label_word = "(a)" if labels[i] == 0 else "(b)"
    
            _ = (question, label_word)
            new_dataset.append(_)

        self.dataset = new_dataset
        self.limit_size()

class StrategyQADataset(CoTDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset = load_dataset(path=self.dataset_cfg.HF_IDENTIFIER, name=self.dataset_cfg.HF_NAME, split=self.split_name_str)

        # dataset_fields: ['question', 'answer']
        new_dataset = []
        inputs = self.dataset["inputs"] # Q: Would New Year's Eve hypothetically be Bacchus's favorite holiday?\nA:
        inputs = [i[:-3] for i in inputs]
        targets = self.dataset["targets"]

        for i in tqdm(range(len(inputs))):
            
            question = inputs[i]
            label_word = "Yes" if targets[i][:3] == "Yes" else "No"
    
            _ = (question, label_word)
            new_dataset.append(_)

        self.dataset = new_dataset
        self.limit_size()

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

            new_dataset.append((question, target))

        self.dataset = new_dataset
        pdb.set_trace()
        self.limit_size()

class DreamDataset(CoTDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset = load_dataset(path=self.dataset_cfg.HF_IDENTIFIER, name=self.dataset_cfg.HF_NAME, split=self.split_name_str)

        # dataset_fields: ['question', 'answer']
        new_dataset = []
        
        dialogues = self.dataset["dialogue"]
        questions = self.dataset["question"]
        choices = self.dataset["choice"]
        answers = self.dataset["answer"]

        for i in tqdm(range(len(dialogues))):
            
            question = "Dialogue: {}\nQuestion: {}\nChoices:\n{}".format(" ".join(dialogues[i]), questions[i], "\n".join(choices[i]))
            label_word = answers[i]
    
            _ = (question, label_word)
            new_dataset.append(_)

        self.dataset = new_dataset
        self.limit_size()
        #pdb.set_trace()


class HotPotQA(CoTDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset = load_dataset(path=self.dataset_cfg.HF_IDENTIFIER, name=self.dataset_cfg.HF_NAME, split=self.split_name_str)

        # dataset_fields: ['question', 'answer']
        new_dataset = []
        pdb.set_trace()
        dialogues = self.dataset["dialogue"]
        questions = self.dataset["question"]
        choices = self.dataset["choice"]
        answers = self.dataset["answer"]

        for i in tqdm(range(len(dialogues))):
            
            question = "Dialogue: {}\nQuestion: {}\nChoices:\n{}".format(" ".join(dialogues[i]), questions[i], "\n".join(choices[i]))
            label_word = answers[i]
    
            _ = (question, label_word)
            new_dataset.append(_)

        self.dataset = new_dataset
        self.limit_size()

class HFBigBenchDataset(CoTDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.dataset = load_dataset(path=self.dataset_cfg.HF_IDENTIFIER, name=self.dataset_cfg.HF_NAME, split=self.split_name_str)

        # dataset_fields: ['idx', 'inputs', 'targets', 'multiple_choice_targets', 'multiple_choice_scores']
        new_dataset = []
        for i in range(len(self.dataset["idx"])):
            _ = (self.dataset["inputs"][i], self.dataset["targets"][i], self.dataset["multiple_choice_targets"][i], self.dataset["multiple_choice_scores"][i], self.dataset["idx"][i])
            new_dataset.append(_)
        self.dataset = new_dataset

        self.limit_size()



class BBArithemeticDataset(CoTDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.dataset = load_dataset(path=self.dataset_cfg.HF_IDENTIFIER, name=self.dataset_cfg.HF_NAME, split=self.split_name_str)

        # dataset_fields: ['idx', 'inputs', 'targets', 'multiple_choice_targets', 'multiple_choice_scores']
        new_dataset = []
        for i in range(len(self.dataset["idx"])):
            question = self.dataset[i]["inputs"]
            answer = self.dataset[i]["targets"]
            new_dataset.append((question, answer))
        self.dataset = new_dataset
        self.limit_size()
        pdb.set_trace()

class ANLIDataset(CoTDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.dataset = load_dataset(path=self.dataset_cfg.HF_IDENTIFIER, name=self.dataset_cfg.HF_NAME, split=self.split_name_str)

        # dataset_fields: ['idx', 'inputs', 'targets', 'multiple_choice_targets', 'multiple_choice_scores']
        new_dataset = []
        pdb.set_trace()
        for i in range(len(self.dataset["idx"])):
            question = self.dataset[i]["inputs"]
            answer = self.dataset[i]["targets"][0]
            new_dataset.append((question, answer))
        self.dataset = new_dataset
        self.limit_size()
        pdb.set_trace()

class ESNLIDataset(CoTDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.dataset = load_dataset(path=self.dataset_cfg.HF_IDENTIFIER, name=self.dataset_cfg.HF_NAME, split=self.split_name_str)

        # dataset_fields: ['idx', 'inputs', 'targets', 'multiple_choice_targets', 'multiple_choice_scores']
        new_dataset = []
        
        for i in range(len(self.dataset)):
            example = self.dataset[i]
            question = 'Premise:\n"{}"\nBased on this premise, can we conclude the hypothesis "{}" is true?\nOptions:\n-yes\n-no\n-it is not possible to tell'.format(example["premise"], example["hypothesis"])
            if example["label"] == 0:
                answer_word = "yes."
            elif example["label"] == 1:
                answer_word = "no."
            else:
                answer_word = "it is not possible to tell."
            if self.dataset_cfg.USE_EXPLAINATION:
                target = "{}. So the answer is: {}".format(example["explanation_1"], answer_word)
            else:
                target = answer_word
            new_dataset.append((question, target))
        self.dataset = new_dataset
        self.limit_size()

