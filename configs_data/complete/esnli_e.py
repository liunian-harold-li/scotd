# use py as the config file because yaml throws a tone of errors when parsing strings
cfg = {
  "TYPE": ("commonsense", ),
  "SOURCE": "hf",
  "CLEAN_STRATEGY": "answer_split",
  "HF_IDENTIFIER": "esnli",
  "HF_NAME": "plain_text",
  "FEW_SHOT_PROMPT": (),

  "USE_PROMPT_SOURCE": False,

  "MODULE_NAME": "dataset.hf_datasets",
  "CLASS_NAME": "ESNLIDataset",
  "INPUT_PREFIX": "",
  "STEP_PREFIX": "\nA: ", # this is the prompt for the answer
  "ANSWER_PREFIX": "So the answer is:",
  "EXAMPLE_SUFFIX": "\n\n",

  "TRAIN": "train",
  "VALID": "validation",
  "TEST": "test",

  "USE_EXPLAINATION": True,

  "FEW_SHOT_COT_PROMPT": (

  )

    # 1e1d0ce1-b0ea-4ad8-9971-b2b44948123b: !Template
}