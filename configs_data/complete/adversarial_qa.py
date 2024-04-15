# use py as the config file because yaml throws a tone of errors when parsing strings
cfg = {
  "TYPE": ("commonsense", ),
  "SOURCE": "hf",
  "CLEAN_STRATEGY": "answer_split",
  "HF_IDENTIFIER": "kilt_tasks",
  "HF_NAME": "hotpotqa",
  "FEW_SHOT_PROMPT": (),

  "USE_PROMPT_SOURCE": False,

  "MODULE_NAME": "dataset.hf_datasets",
  "CLASS_NAME": "HotPotQA",
  "INPUT_PREFIX": "",
  "STEP_PREFIX": "\nA: ", # this is the prompt for the answer
  "ANSWER_PREFIX": " So the answer is ",
  "EXAMPLE_SUFFIX": "\n\n",

  "TRAIN": "train",
  "VALID": "validation",
  "TEST": "test",

  "FEW_SHOT_COT_PROMPT": (
    "Motor systems are areas of the brain that are directly or indirectly involved in producing body movements, that is, in activating muscles. Except for the muscles that control the eye, which are driven by nuclei in the midbrain, all the voluntary muscles in the body are directly innervated by motor neurons in the spinal cord and hindbrain. Spinal motor neurons are controlled both by neural circuits intrinsic to the spinal cord, and by inputs that descend from the brain. The intrinsic spinal circuits implement many reflex responses, and contain pattern generators for rhythmic movements such as walking or swimming. The descending connections from the brain allow for more sophisticated control."

  )

    # 1e1d0ce1-b0ea-4ad8-9971-b2b44948123b: !Template
}