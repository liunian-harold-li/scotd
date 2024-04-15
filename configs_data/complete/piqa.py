# use py as the config file because yaml throws a tone of errors when parsing strings
cfg = {
  "TYPE": ("commonsense", ),
  "SOURCE": "hf",
  "CLEAN_STRATEGY": "general",
  "HF_IDENTIFIER": "piqa",
  "HF_NAME": "plain_text",
  "FEW_SHOT_PROMPT": (),

  "MODULE_NAME": "dataset.hf_datasets",
  "CLASS_NAME": "PIQADataset",
  "INPUT_PREFIX": "",
  "STEP_PREFIX": "\nA: ", #"\nA: Let's think step by step. ", # this is the prompt for the answer
  #"ANSWER_PREFIX": " So the answer is ",
  "EXAMPLE_SUFFIX": "\n\n",

  "FEW_SHOT_COT_PROMPT": (
    "Goal : What is the best way to remove painters tape after you have finished frosting a window?\nSolution (a): Remove the painter's tape carefully from the glass after the frosting has dried. Peel slowly to avoid accidentally removing the frosting\nSolution (b): Remove the painter's tape quickly from the glass before the frosting can dry. Peel swiftly to avoid accidentally removing the frosting.",
    
    "The goal is to remove the painters tape. We need to wait till the frosting has dried and peel slowly. Otherwise, the frosting could come off as we peel. Thus, solution (a) is correct.",
    "(a)",

    "Goal : To create a makeshift ice pack,\nSolution (a): take a sponge and soak it in oil. Put the sponge in a refrigerator and let it freeze. Once frozen, take it out and put it in a ziploc bag. You can now use it as an ice pack.\nSolution (b): take a sponge and soak it in water. Put the sponge in a refrigerator and let it freeze. Once frozen, take it out and put it in a ziploc bag. You can now use it as an ice pack.",

    "The goal is to create an ice pack. Only water can freeze and become an ice pack. Thus, solution (b) is correct.",

    "(b)",


    "Goal : How can I save my milk if I am going to be leaving town for a bit?\nSolution (a): Freeze the milk, but make sure the jug is not more than 5/4 full at most. Then just let the milk thaw when you get back.\nSolution (b): Freeze the milk, but make sure the jug is not more than 3/4 full at most. Then just let the milk thaw when you get back.",

    "The goal is to save my milk if I am going to be leaving town for a bit. When we freeze the milk, the milk will emit gas and the jug could explode if it is too full. Thus, we should make sure the jug is not more than 3/4 full at most. Thus, solution (b) is correct.",

    "(b)",


  ),

  "TRAIN": "train",
  "VALID": "validation",
}