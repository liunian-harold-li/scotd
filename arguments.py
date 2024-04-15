import torch
import argparse
import os
import random
import time

def get_args():
    parser = argparse.ArgumentParser(description='RL')

    # dataset
    parser.add_argument(
        '--target-sentiment', type=str, default='positive')
    parser.add_argument(
        '--output-dir', type=str, default='OUTPUTS/')
    parser.add_argument(
        '--dataset_type', type=str, default=None)
    parser.add_argument(
        '--dataset-train', type=str, default='data/sentiment/train.jsonl',
        help='JSONL file containing train prompts. Each row must contain a prompt at `row["prompt"]["text"]`.')

    # reward
    parser.add_argument(
        '--n_extra_tokens', type=int, default=5, help='number of reward categorization')
    parser.add_argument(
        '--sample-interval', type=int, default=500, help='step interval to sample from current policy')
    parser.add_argument(
        '--horizon', type=float, default=2500, help='horizon value in adaptive controller')
    # KL term
    parser.add_argument(
        '--kl_coef', type=float, default=0.05, help='coefficient for KL term in reward')
    parser.add_argument(
        '--adaptive_kl', action='store_true', default=False, help='whether to use adaptive KL controller')
    parser.add_argument(
        '--target_kl', type=float, default=3, help='target value in adaptive KL controller')
    # entropy term
    parser.add_argument(
        '--entropy_coef', type=float, default=0.06, help='coefficient for entropy term in reward')
    parser.add_argument(
        '--adaptive_entropy', action='store_true', default=False, help='whether to use adaptive entropy controller')
    parser.add_argument(
        '--target_entropy', type=float, default=40, help='target value in adaptive entropy controller')

    # policy
    parser.add_argument(
        '--init-model', type=str, default='gpt2-large', help='language model used for policy.')
    parser.add_argument(
        '--ref-model', type=str, default='gpt2-large', help='language model used for reference policy.')
    parser.add_argument(
        '--response-length', type=int, default=20, help='number of tokens to generate for each prompt.')
    parser.add_argument(
        '--temperature', type=float, default=1.0, help='temperature for sampling policy.')

    # training˚
    parser.add_argument(
        '--total-episodes', type=int, default=3000000, help='total number of episodes')
    parser.add_argument(
        '--batch_size', type=int, default=128, help='batch size')
    parser.add_argument(
        '--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument(
        '--num_warmup_steps', type=int, default=500, help='number of warmup steps in lr scheduler')
    parser.add_argument(
        '--clip_grad', action='store_true', default=False, help='whether to clip gradient')
    parser.add_argument(
        '--max-grad-norm', type=float, default=0.5, help='maximum norm of gradients ')

    # generation
    parser.add_argument(
        '--num-samples', type=int, default=25, help='number of samples to generate for each prompt.')
    parser.add_argument(
        '--top-p', type=float, default=1.0, help='hyperparameter for nucleus sampling')

    # other
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--log-interval', type=int, default=100, help='step interval to print out logs')
    parser.add_argument(
        '--save-interval', type=int, default=1000, help='step interval to save model checkpoints')
    parser.add_argument(
        '--eval-interval', type=int, default=500, help='step interval to do evaluation')
    parser.add_argument(
        '--cuda-deterministic', action='store_false', default=True,
        help="sets flags for determinism when using CUDA (potentially slow!)")
    
    # Custom Stuff
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--config_file", default=None, type=str, help="Specify yaml config file"
    )

    parser.add_argument("--do_sample_eval", action="store_true", default=False, help="whether to do sample eval")
    parser.add_argument("--save_sample_eval_path", type=str, default=None, help="path to save sample eval")
    parser.add_argument('--use_wandb', action='store_true', default=False, help='whether to use wandb')
    parser.add_argument('--update_wandb', action='store_true', default=False, help='whether to update wandb')

    parser.add_argument('--save_eval_results', action='store_true', default=False, help='whether to save eval results')

    parser.add_argument('--inspect_datapool', action='store_true', default=False, help='whether to inspect datapool')


    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    return args


def try_to_find(file, return_dir=False, search_path=['./DATASET', './OUTPUT', './data', './MODEL']):
    if not file:
        return file

    DATASET_PATH = ['./']
    if 'DATASET' in os.environ:
        DATASET_PATH.append(os.environ['DATASET'])
    DATASET_PATH += search_path

    for path in DATASET_PATH:
        if os.path.exists(os.path.join(path, file)):
            if return_dir:
                return path
            else:
                return os.path.join(path, file)

    print('Cannot find {} in {}'.format(file, DATASET_PATH))
    exit(1)


import os
from yacs.config import CfgNode as CN
_C = CN()

_C.RUN_UID = None

_C.MODEL = CN()
_C.DATA = CN()
_C.MISC = CN()

_C.MISC.VERBOSE = False

_C.DATA.TRAIN = "train"
_C.DATA.TEST = "val"

_C.DATA.TYPE = "cot"
_C.DATA.NAME = None
_C.DATA.BATCH_SIZE = 32
_C.DATA.EVAL_BATCH_SIZE = 1
_C.DATA.CONFIG = None
_C.DATA.OVERRIDE_CONFIG = None
_C.DATA.COPIES = ()

_C.DATA.LIMIT_DATASET_SIZE = 0
_C.DATA.LIMIT_WITH_JSON_FILE = None
_C.DATA.START_INDEX = 0 # for sharded sampling of the dataset
_C.DATA.SHUFFLE_BEFORE_LIMIT = False
_C.DATA.SHUFFLE_AFTER_LIMIT = False
_C.DATA.FEW_SHOT_PROMPT = False
_C.DATA.FEW_SHOT_COT_PROMPT = False
_C.DATA.SHUFFLE_TRAIN = True
_C.DATA.NUM_WORKERS = 1
_C.DATA.DATAPOOL_USE_INDEX = False
_C.DATA.DATAPOOL_VERSION = "v1"
_C.DATA.DATASETS_KEYS_MAPPED_TO_POOL = ()


_C.DATA.FILTER_STRATEGY = "all_correct_paths" # one_correct_path
_C.DATA.POOL_GET_ITEM_FROM_DATASET = False

# Prompt control
_C.DATA.SHUFFLE_COT_PROMPTS = False
_C.DATA.RANDOM_DROP_COT_PROMPTS = False
_C.DATA.RANDOM_DROP_COT_PROMPTS_MIN_NUM = 3


_C.DATA.PROMPT_ARGS = CN()
_C.DATA.PROMPT_ARGS.USE_DATASET = False
_C.DATA.PROMPT_ARGS.PREDICT_RATIONALE = False
_C.DATA.PROMPT_ARGS.PROMPT_RATIONALE = False
_C.DATA.PROMPT_ARGS.LOAD_RATIONALES = False
_C.DATA.PROMPT_ARGS.RATIONALE_PATH = None
_C.DATA.PROMPT_ARGS.RANDOM_DROP_COT_PROMPTS_MAX_NUM = 1000
_C.DATA.PROMPT_ARGS.USE_RANDOM_TRAIN_PROMPT = False
_C.DATA.PROMPT_ARGS.LIMIT_PROMPT_NUM = False
_C.DATA.PROMPT_ARGS.DROP_RATIONALE = False

# will be deprecated
_C.DATA.STEP_BY_STEP = CN()
_C.DATA.STEP_BY_STEP.DATASET = "aqua"
_C.DATA.STEP_BY_STEP.MAX_NUM_WORKER = 3
_C.DATA.STEP_BY_STEP.MODEL = "gpt3"
_C.DATA.STEP_BY_STEP.METHOD = "zero_shot_cot" # choices=["zero_shot", "zero_shot_cot", "few_shot", "few_shot_cot"],
_C.DATA.STEP_BY_STEP.COT_TRIGGER_NO = 1
_C.DATA.STEP_BY_STEP.MAX_LENGTH_COT = 128 # IMPORTANT, this is max NEW tokens
_C.DATA.STEP_BY_STEP.MAX_LENGTH_DIRECT = 32 # IMPORTANT, this is max NEW tokens
_C.DATA.STEP_BY_STEP.LIMIT_DATASET_SIZE = 10
_C.DATA.STEP_BY_STEP.API_TIME_INTERVAL = 1.0
_C.DATA.STEP_BY_STEP.LOG_DIR = "./log/"

_C.DATA.STEP_BY_STEP.DATASET_PATH = None
_C.DATA.STEP_BY_STEP.DIRECT_ANSWER_TRIGGER = None
_C.DATA.STEP_BY_STEP.PLAUSIBLE_ANSWER_TRIGGER = None
_C.DATA.STEP_BY_STEP.DIRECT_ANSWER_TRIGGER_FOR_ZEROSHOT = None
_C.DATA.STEP_BY_STEP.DIRECT_ANSWER_TRIGGER_FOR_ZEROSHOT_COT = None
_C.DATA.STEP_BY_STEP.DIRECT_ANSWER_TRIGGER_FOR_FEWSHOT = None
_C.DATA.STEP_BY_STEP.COT_TRIGGER = None 


_C.MODEL = CN() # due to legacy issue, _C.MODEL means the reference policy model
_C.MODEL.TYPE = "gpt3"
_C.MODEL.NAME = "gpt3"
_C.MODEL.LOAD_WITH_ACCELERATE = False
_C.MODEL.ACCELERATE_MAX_MEMORY = ()
_C.MODEL.MANUAL_DEVICE_MAP = ()
_C.MODEL.LOAD_WITH_FP16 = False
_C.MODEL.ACCELERATE_MIX_PREDICTION = "no"
_C.MODEL.ACCELERATE_FP16 = False

_C.MODEL.STOP_AT_NEW_LINE = False
_C.MODEL.STOP_AT_NEW_LINE_BATCHED = True
_C.MODEL.ADDITIONL_SUFFIX_STRING = ""
_C.MODEL.WEIGHT = None

# Sample
_C.MODEL.TEMPERATURE = 0.0
_C.MODEL.SAMPLING_TEMP = 0.0 # this argument is redundant; just keep it here for legacy reason
_C.MODEL.DO_SAMPLE = False

# Quark related
_C.MODEL.USE_QUARK = False
_C.MODEL.REWARD_CATEGORY_NUM = 2
_C.MODEL.POOL_THRESHOLD = 0.5

# PPO related
_C.MODEL.VANILLA_POLICY_GRADIENT = False
_C.MODEL.VANILLA_POLICY_GRADIENT_WEIGHT = -0.1

# this is for GPT-3
_C.MODEL.TEMPERATURE = 0.0
_C.MODEL.SAMPLING_NUM = 1
_C.MODEL.STOP_TOKEN = None
_C.MODEL.SLEEP_TIME = 1
_C.MODEL.NO_OP_TOKEN_NUM = 0

_C.MODEL.STOP_AT_EOS_BATCHED = False
_C.MODEL.NUM_BEAMS = 1
_C.MODEL.STOP_TOKEN_GPT = None
_C.MODEL.MAX_INPUT_LENGTH = -1
# copy the config of _C.MODEL
_C.POLICY_MODEL = _C.MODEL.clone()



_C.TRAIN = CN()
_C.TRAIN.REPORT_INTERVAL = 100
_C.TRAIN.DATAPOOL_PATH = None
_C.TRAIN.EVAL_INTERVAL = 1000
_C.TRAIN.SAVE_INTERVAL = 1000
_C.TRAIN.SAMPLE_INTERVAL = 500
_C.TRAIN.EVAL_LIMIT_SIZE = -1
_C.TRAIN.RESUME_FROM = None
_C.TRAIN.RESUME_FROM_STEP = 0
_C.TRAIN.CUSTOM_NAME = "Placeholder"

_C.TRAIN.LR = 1e-5
_C.TRAIN.CLIP_GRAD = False
_C.TRAIN.TOTAL_STEPS = 3000000
_C.TRAIN.SAVE_DIR = None

_C.TRAIN.SAMPLE_ROUNDS = 1
_C.TRAIN.PURGE_EVERYTIME = False

_C.TRAIN.GRAD_ACCUMULATE_STEPS = 1

_C.PPO = CN()
_C.PPO.TARGET_SENTIMENT = "positive"
_C.PPO.OUTPUT_DIR = "outputs"
_C.PPO.DATASET_TYPE = None
_C.PPO.DATASET_TRAIN = "data/sentiment/train.jsonl"

_C.PPO.N_EXTRA_TOKENS = 5
_C.PPO.HORIZON = 2500

_C.PPO.KL_COEF = 0.05
_C.PPO.ADAPTIVE_KL = False
_C.PPO.TARGET_KL = 3
_C.PPO.ENTROPY_COEF = 0.06
_C.PPO.ADAPTIVE_ENTROPY = False
_C.PPO.TARGET_ENTROPY = 40

_C.PPO.INIT_MODEL = "gpt2-large"
_C.PPO.REF_MODEL = "gpt2-large"
_C.PPO.RESPONSE_LENGTH = 20
_C.PPO.TEMPERATURE = 1.0

# _C.PPO.TOTAL_EPISODES = 3000000
# _C.PPO.BATCH_SIZE = 128
# _C.PPO.NUM_WARMUP_STEPS = 500
_C.PPO.MAX_GRAD_NORM = 0.5

_C.PPO.NUM_SAMPLES = 25
_C.PPO.TOP_P = 1.0
_C.PPO.SEED = 1
_C.PPO.LOG_INTERVAL = 100
_C.PPO.SAVE_INTERVAL = 1000
_C.PPO.CUDA_DETERMINISTIC = True
_C.PPO.CUDA = True


def post_process_yacs_config(cfg):
    cfg.defrost()

    sub_node = cfg.DATA.STEP_BY_STEP
    if sub_node.DATASET == "aqua":
        sub_node.DATASET_PATH = "./data/cot/AQuA/test.json"
        sub_node.DIRECT_ANSWER_TRIGGER = "\nTherefore, among A through E, the answer is"
    elif sub_node.DATASET == "gsm8k":
        sub_node.DATASET_PATH = "./data/cot/grade-school-math/test.jsonl"
        sub_node.DIRECT_ANSWER_TRIGGER = "\nTherefore, the answer (arabic numerals) is"
    elif sub_node.DATASET == "commonsensqa":
        sub_node.DATASET_PATH = "./data/cot/CommonsenseQA/dev_rand_split.jsonl"
        sub_node.DIRECT_ANSWER_TRIGGER = "\nTherefore, among A through E, the answer is"
        sub_node.PLAUSIBLE_ANSWER_TRIGGER = "Choose the most plausible answer from among choices A through E."
    elif sub_node.DATASET == "addsub":
        sub_node.DATASET_PATH = "./data/cot/AddSub/AddSub.json"
        sub_node.DIRECT_ANSWER_TRIGGER = "\nTherefore, the answer (arabic numerals) is"
    elif sub_node.DATASET == "multiarith":
        sub_node.DATASET_PATH = "./data/cot/MultiArith/MultiArith.json"
        sub_node.DIRECT_ANSWER_TRIGGER = "\nTherefore, the answer (arabic numerals) is"
    elif sub_node.DATASET == "strategyqa":
        sub_node.DATASET_PATH = "./data/cot/StrategyQA/task.json"
        sub_node.DIRECT_ANSWER_TRIGGER = "\nTherefore, the answer (Yes or No) is"
    elif sub_node.DATASET == "svamp":
        sub_node.DATASET_PATH = "./data/cot/SVAMP/SVAMP.json"
        sub_node.DIRECT_ANSWER_TRIGGER = "\nTherefore, the answer (arabic numerals) is"
    elif sub_node.DATASET == "singleeq":
        sub_node.DATASET_PATH = "./data/cot/SingleEq/questions.json"
        sub_node.DIRECT_ANSWER_TRIGGER = "\nTherefore, the answer (arabic numerals) is"
    elif sub_node.DATASET == "bigbench_date":
        sub_node.DATASET_PATH = "./data/cot/Bigbench_Date/task.json"
        sub_node.DIRECT_ANSWER_TRIGGER = "\nTherefore, among A through F, the answer is"
    elif sub_node.DATASET == "object_tracking":
        sub_node.DATASET_PATH = "./data/cot/Bigbench_object_tracking/task.json"
        sub_node.DIRECT_ANSWER_TRIGGER = "\nTherefore, among A through C, the answer is"
    elif sub_node.DATASET == "coin_flip":
        sub_node.DATASET_PATH = "./data/cot/coin_flip/coin_flip.json"
        sub_node.DIRECT_ANSWER_TRIGGER = "\nTherefore, the answer (Yes or No) is"
    elif sub_node.DATASET == "last_letters":
        sub_node.DATASET_PATH = "./data/cot/last_letters/last_letters.json"
        sub_node.DIRECT_ANSWER_TRIGGER = "\nTherefore, the answer is"
    else:
        raise ValueError("dataset is not properly defined ...")

    trigger = sub_node.DIRECT_ANSWER_TRIGGER.replace("\nTherefore, ", "")
    sub_node.DIRECT_ANSWER_TRIGGER_FOR_ZEROSHOT = trigger[0].upper() + trigger[1:]
    sub_node.DIRECT_ANSWER_TRIGGER_FOR_ZEROSHOT_COT = sub_node.DIRECT_ANSWER_TRIGGER
    sub_node.DIRECT_ANSWER_TRIGGER_FOR_FEWSHOT = "The answer is"

    if sub_node.COT_TRIGGER_NO == 1:
        sub_node.COT_TRIGGER = "Let's think step by step."
    elif sub_node.COT_TRIGGER_NO == 2:
        sub_node.COT_TRIGGER = "We should think about this step by step."
    elif sub_node.COT_TRIGGER_NO == 3:
        sub_node.COT_TRIGGER = "First,"
    elif sub_node.COT_TRIGGER_NO == 4:
        sub_node.COT_TRIGGER = "Before we dive into the answer,"
    elif sub_node.COT_TRIGGER_NO == 5:
        sub_node.COT_TRIGGER = "Proof followed by the answer."
    elif sub_node.COT_TRIGGER_NO == 6:
        sub_node.COT_TRIGGER = "Let's think step by step in a realistic way."
    elif sub_node.COT_TRIGGER_NO == 7:
        sub_node.COT_TRIGGER = "Let's think step by step using common sense and knowledge."
    elif sub_node.COT_TRIGGER_NO == 8:
        sub_node.COT_TRIGGER = "Let's think like a detective step by step."
    elif sub_node.COT_TRIGGER_NO == 9:
        sub_node.COT_TRIGGER = "Let's think about this logically."
    elif sub_node.COT_TRIGGER_NO == 10:
        sub_node.COT_TRIGGER = "Let's think step by step. First,"
    elif sub_node.COT_TRIGGER_NO == 11:
        sub_node.COT_TRIGGER = "Let's think"
    elif sub_node.COT_TRIGGER_NO == 12:
        sub_node.COT_TRIGGER = "Let's solve this problem by splitting it into steps."
    elif sub_node.COT_TRIGGER_NO == 13:
        sub_node.COT_TRIGGER = "The answer is after the proof."
    elif sub_node.COT_TRIGGER_NO == 14:
        sub_node.COT_TRIGGER = "Let's be realistic and think step by step."
    else:
        raise ValueError("cot_trigger_no is not properly defined ...")

    cfg.freeze()
    return cfg


def yaml_config_to_args(yaml_config, original_args = None):
    args = argparse.Namespace()
    for k, v in yaml_config.items():
        setattr(args, k.lower(), v)
    if original_args is not None: # original_args is Namespace
        for k, v in original_args.__dict__.items():
            setattr(args, k, v)
    return args

def get_cfg(args):
    cfg = _C.clone()

    cfg.merge_from_file(args.config_file)

    cfg.merge_from_list(args.opts)

    post_process_yacs_config(cfg)
    cfg.freeze()

    if "__test__" in cfg.TRAIN.CUSTOM_NAME.lower():
        args.use_wandb = False

    return cfg