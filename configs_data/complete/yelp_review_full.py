# use py as the config file because yaml throws a tone of errors when parsing strings
cfg = {
  "TYPE": ("commonsense", ),
  "SOURCE": "hf",
  "CLEAN_STRATEGY": "answer_split",
  "HF_IDENTIFIER": "yelp_review_full",
  "HF_NAME": "yelp_review_full",
  "FEW_SHOT_PROMPT": (),

  "USE_PROMPT_SOURCE": False,

  "MODULE_NAME": "configs_data.complete.yelp_review_full",
  "CLASS_NAME": "YelpReviewFull",
  "INPUT_PREFIX": "Q: How many stars (1-5) would you give to the restaurant based on the review?\n",

#   "TRAIN": "train_r3",
    "VALID": "test",
#   "TEST": "test_r3",

   "RANDOM_DROP_COT_PROMPTS_MAX_NUM": 2,

  "FEW_SHOT_COT_PROMPT": (
    "Eat at your own risk. The service is terrible, the staff seem to be generally clueless, the management is inclined to blame the staff for their own mistakes, and there's no sense of FAST in their fast food. When we came, half of the menu board was still on breakfast, and it was 4:30p. The only thing they have going for them is that the food is hot and tastes just like McDonald's should.   Then again, the franchise is owned by Rice, and I've come to take terrible service is their MO.",
    "'Eat at your own risk', 'terrible', 'clueless' all indicate that the reviewer is not happy with the service.",
    '1 star',

    "I love this place, we actually went twice this weekend!    They have some veggie options on the menu which is nice, and they are happy to change anything around to make it vegetarian if you'd like.  The service is friendly and the prices are great.    The meat eaters at my table said that the meats were fantastic and we all left very happy!",
    "'love this place', 'went twice this weekend', 'nice', 'friendly', 'great', 'fantastic' all indicate that the reviewer is happy. With so many positive words, the reviewer likely gives a very high rating.",
    '5 star',

    'Steak was not cooked properly. After the second attempt still wrong .. gave up.    Good sweet potato though!  Not my most favorite place for steak.',
    "'not cooked properly', 'still wrong' all indicate that the reviewer is not happy with the steak. But the reviewer likes the sweet potato. Thus, the reviewer did not give 1 star.",
    '2 star',
    
    "Been here a few times.  Once with a drinking buddy after a night out and for a couple of birthdays.  Generally very nice staff, comfortable, reasonable speed and very yummy as I expect from Dennys.",
    "'generally very nice', 'comfortable', 'reasonable speed', 'yummy' all indicate that the reviewer is happy. But the reviewer is not very enthusiastic. Thus, the reviewer did not give 5 star.",
    '4 star',

  ),

  #"FEW_SHOT_PROMPT_INDEX": (99, 18775, 19875, 37456)
  "FEW_SHOT_PROMPT_INDEX": (1, 2, ), # place holder
  "FEW_SHOT_PROMPT_ORIGINAL": (1, 2, ) # place holder
}

from dataset.hf_datasets import CoTDataset, chunks
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
from promptsource.templates import DatasetTemplates

class YelpReviewFull(CoTDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.dataset = load_dataset(path=self.dataset_cfg.HF_IDENTIFIER, name=self.dataset_cfg.HF_NAME, split=self.split_name_str)
        new_dataset = []
        for i in range(len(self.dataset)):
            example = self.dataset[i]
            new_dataset.append(self.convert_example(example))
        self.dataset = new_dataset
        self.limit_size()
        #pdb.set_trace()

    def convert_example(self, datapoint):
        return datapoint["text"].replace("\\n", " "),"{} star".format(datapoint["label"] + 1), None
    
    def convert_original_to_examples(self, original_prompts):
        examples = [
          {'text': "Eat at your own risk. The service is terrible, the staff seem to be generally clueless, the management is inclined to blame the staff for their own mistakes, and there's no sense of FAST in their fast food. When we came, half of the menu board was still on breakfast, and it was 4:30p. The only thing they have going for them is that the food is hot and tastes just like McDonald's should.   Then again, the franchise is owned by Rice, and I've come to take terrible service is their MO.", 'label': 0},
            {'text': "I love this place, we actually went twice this weekend!    They have some veggie options on the menu which is nice, and they are happy to change anything around to make it vegetarian if you'd like.  The service is friendly and the prices are great.    The meat eaters at my table said that the meats were fantastic and we all left very happy!", 'label': 4},
            {'text': 'Steak was not cooked properly. After the second attempt still wrong .. gave up.    Good sweet potato though!  Not my most favorite place for steak.', 'label': 1},
            {'text': "Been here a few times.  Once with a drinking buddy after a night out and for a couple of birthdays.  Generally very nice staff, comfortable, reasonable speed and very yummy as I expect from Dennys.", 'label': 3},
        ]
        return examples