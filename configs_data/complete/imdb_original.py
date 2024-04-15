# use py as the config file because yaml throws a tone of errors when parsing strings
cfg = {
  "TYPE": ("commonsense", ),
  "SOURCE": "hf",
  "CLEAN_STRATEGY": "answer_split",
  "HF_IDENTIFIER": "imdb",
  "HF_NAME": "plain_text",
  "FEW_SHOT_PROMPT": (),

  "USE_PROMPT_SOURCE": False,

  "MODULE_NAME": "configs_data.complete.imdb_original",
  "CLASS_NAME": "HFDataset",
  "INPUT_PREFIX": "Q: What is the sentiment of the following review?\n",

#   "TRAIN": "train_r3",
    "VALID": "test",
    "RANDOM_DROP_COT_PROMPTS_MAX_NUM": 0,
#   "TEST": "test_r3",

  "FEW_SHOT_COT_PROMPT": (

    "I watched Cabin by the Lake this afternoon on USA. Considering this movie was made for TV is was interesting enough to watch the sequel. So, I tune in for the airing this evening and was extremely disappointed. I knew I wouldn't like the movie, but I was not expecting to be perplexed by the use of DV (digital video). The movie would have been tolerable if it wasn't for these juxtaposed digital shots that seemed to come from nowhere. I expected the plot line to be tied in with these shots, but there seemed to be no logical explanation. (WARNING: THE FOLLOWING MAYBE A SPOILER!!!!) The open ending in Cabin by the Lake was acceptable, but the open ending on the sequel is ridiculous. I can only foresee Return of Return to The Cabin by the Lake being watch able is if the movie was shown up against nothing, but infomercials at 4 o'clock in the morning.",
    "The author at first was assuming that the movie was interesting enough. But then they were extremely disappointed. They said that the movie would have been tolerable if it wasn't for these juxtaposed digital shots, which means that the movie is just untolerable.",
    "negative",

    'Ghost of Dragstrip Hollow is a typical 1950\'s teens in turmoil movie. It is not a horror or science fiction movie. Plot concerns a group of teens who are about to get kicked out of their "hot rod" club because they cannot meet the rent. Once kicked out, they decide to try an old Haunted House. The only saving grace for the film is that the "ghost" (Paul Blaisdell in the She Creature suit) turns out to be an out of work movie monster played by Blaisdell.',
    "The author siad that 'the only saving grace for the film', implying that the movie is terrible and there is only one good thing about it.",
    "negative",


    'Reading the various external reviews of Roger Ebert and other well-known film critics makes me hesitate to admit how much I love this movie, even though I only have a dubbed-into-Japanese video version of it.<br /><br />Apparently, many critics seem to take it as a minus that the story premise has been used before, most famously in the very great "It\'s a Wonderful Life." To me, a great premise is a great premise no matter how many times it\'s used, and I\'d be happy to see more movies using this particular premise--the discontented man who gets a chance to gain a fresh perspective on his own life through a bit of "divine magic."<br /><br />I suppose folks and critics of a more intellectual bent are not as pulled into the story, but I don\'t go to movies to critique them with notebook in hand--I go to movies to throw myself into them and let them take me where they will.',
    'The author said they love this movie. They mention many critics but their point is to say that they are happy to see more movies like this.',
    'positive',
  ),

  "FEW_SHOT_PROMPT_INDEX": (20098, 208)
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
label = {
            0: "negative",
            1: "positive",
        }
class HFDataset(CoTDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.dataset = load_dataset(path=self.dataset_cfg.HF_IDENTIFIER, name=self.dataset_cfg.HF_NAME, split=self.split_name_str)
        
        # new dataset
        path = "../contrast-sets/IMDb/data/test_original.tsv"
        new_dataset = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip().split("\t")
                new_dataset.append((line[1], line[0].lower(), None))
        self.dataset = new_dataset
        #pdb.set_trace()
        self.limit_size()
    