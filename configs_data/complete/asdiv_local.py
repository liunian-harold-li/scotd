# use py as the config file because yaml throws a tone of errors when parsing strings
cfg = {
  "TYPE": ("numerical", ),
  "SOURCE": "hf",
  "CLEAN_STRATEGY": "numerical",
  "FEW_SHOT_PROMPT": (),

  "MODULE_NAME": "dataset.hf_datasets",
  "CLASS_NAME": "ASDIVDataset",
  "LOCAL_PATH": "data/asdiv-a/",

  "INPUT_PREFIX": "Q: ",
  "STEP_PREFIX": "\nA: ", # this is the prompt for the answer
  "ANSWER_PREFIX": " So the answer is ",
  "EXAMPLE_SUFFIX": "\n\n",

  "FEW_SHOT_COT_PROMPT": (
    "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",

    "We start with 15 trees. Later we have 21 trees. The difference must be the number of trees they planted. So, they must have planted 21 - 15 = 6 trees.",

    "6",

    "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",

    "There are 3 cars in the parking lot already. 2 more arrive. Now there are 3 + 2 = 5 cars.",

    "5",

    "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",

    "Leah had 32 chocolates and Leah's sister had 42. That means there were originally 32 + 42 = 74 chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 chocolates.",

    "39",

    "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",

    "Jason had 20 lollipops. Since he only has 12 now, he must have given the rest to Denny. The number of lollipops he has given to Denny must have been 20 - 12 = 8 lollipops.",

    "8",

    "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",

    "He has 5 toys. He got 2 from mom, so after that he has 5 + 2 = 7 toys. Then he got 2 more from dad, so in total he has 7 + 2 = 9 toys.",

    "9",

    "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",

    "There are 4 days from monday to thursday. 5 computers were added each day. That means in total 4 * 5 = 20 computers were added. There were 9 computers in the beginning, so now there are 9 + 20 = 29 computers.",

    "29",

    "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",

    "Michael initially had 58 balls. He lost 23 on Tuesday, so after that he has 58 - 23 = 35 balls. On Wednesday he lost 2 more so now he has 35 - 2 = 33 balls.",

    "33",

    "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",

    "She bought 5 bagels for $3 each. This means she spent 5 * $3 = $15 on the bagels. She had $23 in beginning, so now she has $23 - $15 = $8.",

    "8",
  ),

  "TRAIN": "trainset",
  "VALID": "validset",
  "TEST": "testset",
}