# use py as the config file because yaml throws a tone of errors when parsing strings
cfg = {
  "TYPE": ("numerical", "symbolic", ),
  "SOURCE": "hf",
  "CLEAN_STRATEGY": "answer_split",
  "HF_IDENTIFIER": "bigbench",
  "HF_NAME": "formal_fallacies_syllogisms_negation",
  "INPUT_PREFIX": "\n\nProgram:",
  "ANSWER_PREFIX": "\nAnswer: ",


  "INPUT_PREFIX": "",
  "STEP_PREFIX": " ", # this is the prompt for the answer
  "ANSWER_PREFIX": " So the answer is ",
  "EXAMPLE_SUFFIX": "\n\n",

  "MODULE_NAME": "dataset.hf_datasets",
  "CLASS_NAME": "BBFormalDataset",

  "FEW_SHOT_COT_PROMPT": (
    'Q: "It is not always easy to see who is related to whom -- and in which ways. The following argument pertains to this question: First, whoever is not both a workmate of Derrick and a grandson of Calvin is a schoolmate of Paul. Second, Harlan is a grandson of Calvin. All this entails that Harlan is not a schoolmate of Paul."\n Is the argument, given the explicitly stated premises, deductively valid or invalid?\nA:',
    "Harlan is a grandson of Calvin. But we do not know if Harlan is a workmate of Paul. So we cannot know if Harlan is not both a workmate of Derrick and a grandson of Calvin or not. The premise of the first statement is unknown. So we cannot know if the conclusion of the first argument (Harlan is a schoolmate of Paul) is true or not.",
    "invalid",

    'Q: "It is not always easy to see which chemicals are contained in our consumer products. The following argument pertains to this question: To start with, everything that is an ingredient of YSL Variation Blush is an ingredient of Concealer Stick, too. Hence, not being an ingredient of Concealer Stick is sufficient for not being an ingredient of YSL Variation Blush."\n Is the argument, given the explicitly stated premises, deductively valid or invalid?\nA:',
    "Everything that is an ingredient of YSL Variation Blush is an ingredient of Concealer Stick, too. Hence, if something is not an ingredient of Concealer Stick, then it is not an ingredient of YSL Variation Blush.",
    "valid",


    'Q: "It is not always easy to see who is related to whom -- and in which ways. The following argument pertains to this question: To start with, every daughter of Esther is a workmate of Tonda. Now, it is false that Carolina is a workmate of Tonda. In consequence, it is false that Carolina is a daughter of Esther."\n Is the argument, given the explicitly stated premises, deductively valid or invalid?\nA:',

    "Carolina is not a workmate of Tonda. Thus, Carolina is not a daughter of Esther because every daughter of Esther is a workmate of Tonda.",
    "invlaid",

  )
}