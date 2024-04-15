# use py as the config file because yaml throws a tone of errors when parsing strings
cfg = {
  "TYPE": ("numerical", "symbolic", ),
  "SOURCE": "hf",
  "CLEAN_STRATEGY": "answer_split",
  "HF_IDENTIFIER": "bigbench",
  "HF_NAME": "auto_debugging",
  "INPUT_PREFIX": "\n\nProgram:",
  "ANSWER_PREFIX": "\nAnswer: ",


  "INPUT_PREFIX": "Program: ",
  "STEP_PREFIX": "\nA: ", # this is the prompt for the answer
  "ANSWER_PREFIX": " So the answer is : ",
  "EXAMPLE_SUFFIX": "\n\n",

  "MODULE_NAME": "dataset.hf_datasets",
  "CLASS_NAME": "HFBigBenchDataset",

  "FEW_SHOT_PROMPT": (
    "```\nif x < 5:\n\tpass\n```\nWhat error does this program surface?", 
    "NameError: name 'x' is not defined",

    "```\nd = {1: 'hello', 2: 'bye', 3: 'good morning'}\nstr_ = ''\nfor i in range(len(d) + 1):\n\tif i % 2 == 1:\n\t\tstr_ += d[i]\n```\nWhat is the value of str_ after this program executes?", 'hellogood morning',

    "```\nclass MyClass():\n\tdef __init__(self):\n\t\tprint(param)\nx = MyClass(5)\n```\nWhat is the value of x.param at the end of this program?", 'None',

    "```\nsum = 0\nfor i in range(100):\n\tsum += i\n```\nWhat is the value of sum immediately after the 10th time line 3 is executed?", '45', ),

  "FEW_SHOT_COT_PROMPT": (
    "```\nif x < 5:\n\tpass\n```\nWhat error does this program surface?", 
    "\nLet's think step by step. At line 1, the code uses the variable x, which has not been defined.", 
    "NameError: name 'x' is not defined",

    "```\nd = {1: 'hello', 2: 'bye', 3: 'good morning'}\nstr_ = ''\nfor i in range(len(d) + 1):\n\tif i % 2 == 1:\n\t\tstr_ += d[i]\n```\nWhat is the value of str_ after this program executes?", 
    "Let's think step by step. At line 3, the code iterates over i within the range of len(d) + 1, which is 4. When i is 0, i % 2 != 1, thus str_ is not changed. When i is 1, i % 2 == 1, thus so we add d[i] to str_; d[i] is d[1], which is 'hello'; thus str_ is '' + 'hello' = 'hello'. When i is 2, i % 2 != 1, str_ is not changed. When i is 3, i % 2 == 1, thus so we add d[i] to str_; d[i] is d[3], which is 'good morning'; thus str_ is 'hello' + 'good morning' = 'hellogood morning'.",
    'hellogood morning',
    
    "```\nclass MyClass():\n\tdef __init__(self):\n\t\tprint(param)\nx = MyClass(5)\n```\nWhat is the value of x.param at the end of this program?",
    "\nLet's think step by step. param is not defined. Thus, the value of x.param is None.",
    'None',

    "```\nsum = 0\nfor i in range(100):\n\tsum += i\n```\nWhat is the value of sum immediately after the 10th time line 3 is executed?", 
    "\nLet's think step by step. The 1st time line 3 is excuted, i = 0, sum = 0 + i = 0 + 0 = 0. The 2nd time line 3 is excuted, i = 1, sum = 0 + i = 0 + 1 = 1. The 3rd time line 3 is excuted, i = 2, sum = 1 + i = 1 + 2 = 3. The 4th time line 3 is excuted, i = 3, sum = 3 + i = 3 + 3 = 6. The 5th time line 3 is excuted, i = 4, sum = 6 + i = 6 + 4 = 10. The 6th time line 3 is excuted, i = 5, sum = 10 + i = 10 + 5 = 15. The 7th time line 3 is excuted, i = 6, sum = 15 + i = 15 + 6 = 21. The 8th time line 3 is excuted, i = 7, sum = 21 + i = 21 + 7 = 28. The 9th time line 3 is excuted, i = 8, sum = 28 + i = 28 + 8 = 36. The 10th time line 3 is excuted, i = 9, sum = 36 + i = 36 + 9 = 45.",
    '45',
  )
}