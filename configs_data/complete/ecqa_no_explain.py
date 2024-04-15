# use py as the config file because yaml throws a tone of errors when parsing strings
cfg = {
  "TYPE": ("commonsense", ),
  "SOURCE": "hf",
  "CLEAN_STRATEGY": "general",
  "FEW_SHOT_PROMPT": (),

  "USE_PROMPT_SOURCE": False,

  "MODULE_NAME": "dataset.hf_datasets",
  "CLASS_NAME": "ECQADataset",
  "LOCAL_PATH": "data/ecqa/",
  "INPUT_PREFIX": "Q: ",
  "STEP_PREFIX": "\nA: ", # this is the prompt for the answer
  "ANSWER_PREFIX": " So the answer is ",
  "EXAMPLE_SUFFIX": "\n\n",

  "USE_EXPLAINATION": False,

  "TRAIN": "cqa_data_train",
  "VALID": "cqa_data_val",
  "TEST": "cqa_data_test",

  "FEW_SHOT_COT_PROMPT": (
    "Where would you borrow coffee if you do not have any?\nAnswer Choices:\n(a) meeting\n(b) convenience store\n(c) supermarket\n(d) fast food restaurant\n(e) friend's house",
    "If you are finished with stock of coffee beans / powder and don't want to buy it, you can borrow it from friend's home because you can't borrow from meeting and other options are selling the coffee.",
    "(e)",

    "If you want to set a romantic atmosphere you might light a candle where?\nAnswer Choices:\n(a) dimly lit room\n(b) synagogue\n(c) bedroom\n(d) birthday cake\n(e) roses",
    "A romantic atmosphere can be set in bedroom and not in a synagogue. Bedroom is a place where one sleeps unlike a dimly lit room or a birthday cake. Candles can be lit in a bedroom and not in roses.",
    "(c)",

    "What is the likelihood of drowning when getting wet?\nAnswer Choices:\n(a) shrinking\n(b) feeling cold\n(c) become cold\n(d) cool off\n(e) could",
    "One could drown in too much water. So the likelihood of drowning when getting wet is they could. All other options are not likelihood.",
    "(e)",

    "What does the government have control over?\nAnswer Choices:\n(a) trouble\n(b) country\n(c) army\n(d) city\n(e) control",
    "A city is a large town over which a government has control. One cannot have control over control and a government might not have control over others",
    "(d)",

    "He had a lot on his plate opening business, this cause a lot of what?\nAnswer Choices:\n(a) headaches\n(b) making money\n(c) success\n(d) failure\n(e) stress",
    "When someone has lot on plate, they often feel stressed. A new business demands lot o fwork that can cause stress. All the other options are incorrect as they are not a result of being a lot on plate in a business.",
    "(e)",


    "If you want to set a romantic atmosphere you might light a candle where?\nAnswer Choices:\n(a) dimly lit room\n(b) synagogue\n(c) bedroom\n(d) birthday cake\n(e) roses",
    "A romantic atmosphere can be set in bedroom and not in a synagogue. Bedroom is a place where one sleeps unlike a dimly lit room or a birthday cake. Candles can be lit in a bedroom and not in roses.",
    "(c)",

    "Q: What is the likelihood of drowning when getting wet?\nAnswer Choices:\n(a) shrinking\n(b) feeling cold\n(c) become cold\n(d) cool off\n(e) could",
    "One could drown in too much water. So the likelihood of drowning when getting wet is they could. All other options are not likelihood.",
    "(e)",
  )

    # 1e1d0ce1-b0ea-4ad8-9971-b2b44948123b: !Template
}