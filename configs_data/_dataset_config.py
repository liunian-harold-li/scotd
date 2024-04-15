import os
from yacs.config import CfgNode as CN
_C = CN()

_C.TYPE = ()
_C.SOURCE = None
_C.LOCAL_PATH = None
_C.HF_IDENTIFIER = None
_C.HF_NAME = None
_C.FEW_SHOT_PROMPT = ()
_C.FEW_SHOT_PROMPT_INDEX = ()
_C.FEW_SHOT_PROMPT_ORIGINAL = ()
_C.FEW_SHOT_COT_PROMPT = ()
_C.CLEAN_STRATEGY = None
_C.INPUT_PREFIX = ""
_C.ANSWER_PREFIX = " So the answer is: "
_C.STEP_PREFIX = ""
_C.EXAMPLE_SUFFIX = ""
_C.MODULE_NAME = "dataset.cot_original"
_C.CLASS_NAME = "CoTOriginalDataset"

_C.PROMPT_SOURCE_INDEX = None

_C.APPEND_RATIONAL_ININPUT = False
_C.PERTURB_RATIONAL = False


_C.USE_PROMPT_SOURCE = False
_C.PROMPT_SOURCE_BAD_TEMPLATE = ()
_C.PROMPT_SOURCE_APPENDIX = None
_C.PROMPT_SOURCE_USE_RATIONALE = False

_C.USE_EXPLAINATION = False
_C.RANDOM_DROP_COT_PROMPTS_MAX_NUM = 10000

_C.TRAIN = "train"
_C.VALID = "val"
_C.TEST = "test"