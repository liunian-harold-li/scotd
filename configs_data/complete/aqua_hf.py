# use py as the config file because yaml throws a tone of errors when parsing strings
cfg = {
  "TYPE": ("numerical", ),
  "SOURCE": "hf",
  "CLEAN_STRATEGY": "general",
  "HF_IDENTIFIER": "aqua_rat",
  "HF_NAME": "raw",
  "FEW_SHOT_PROMPT": (),

  "MODULE_NAME": "dataset.hf_datasets",
  "CLASS_NAME": "AquaDataset",
  "INPUT_PREFIX": "Q: ",

  "STEP_PREFIX": "\nA: ", # this is the prompt for the answer

  "ANSWER_PREFIX": " So the answer is ",

  "EXAMPLE_SUFFIX": "\n\n",
  "TRAIN": "train",
  "VALID": "validation",
  "TEST": "test",

  "FEW_SHOT_COT_PROMPT": (
    "A person is traveling at 20 km/hr and reached his destiny in 2.5 hr then find the distance?\nAnswer Choices:\n(a) 53 km\n(b) 55 km\n(c) 52 km\n(d) 60 km\n(e) 50 km",
    
    "The distance that the person traveled would have been 20 km/hr * 2.5 hrs = 50 km.",
    
    "(e)",

    
    "John found that the average of 15 numbers is 40. If 10 is added to each number then the mean of the numbers is?\nAnswer Choices:\n(a) 50\n(b) 45\n(c) 65\n(d) 78\n(e) 64",

    "If 10 is added to each number, then the mean of the numbers also increases by 10. So the new mean would be 50.", 
    
    "(a)",
    
    "If a / b = 3/4 and 8a + 5b = 22,then find the value of a.\nAnswer Choices:\n(a) 1/2\n(b) 3/2\n(c) 5/2\n(d) 4/2\n(e) 7/2",
    
    "If a / b = 3/4, then b = 4a / 3. So 8a + 5(4a / 3) = 22. This simplifies to 8a + 20a / 3 = 22, which means 44a / 3 = 22. So a is equal to 3/2.", 
    
    "(b)",
    
    "How many keystrokes are needed to type the numbers from 1 to 500?\nAnswer Choices:\n(a) 1156\n(b) 1392\n(c) 1480\n(d) 1562\n(e) 1788",
    
    "There are 9 one-digit numbers from 1 to 9. There are 90 two-digit numbers from 10 to 99. There are 401 three-digit numbers from 100 to 500. 9 + 90(2) + 401(3) = 1392.",
    
    "(b)",
  )

  # "Q: A person is traveling at 20 km/hr and reached his destiny in 2.5 hr then find the distance?\nAnswer Choices:\n(a) 53 km\n(b) 55 km\n(c) 52 km\n(d) 60 km\n(e) 50 km\nA: The distance that the person traveled would have been 20 km/hr * 2.5 hrs = 50 km. So the answer is (e).\n\nQ: John found that the average of 15 numbers is 40. If 10 is added to each number then the mean of the numbers is?\nAnswer Choices:\n(a) 50\n(b) 45\n(c) 65\n(d) 78\n(e) 64\nA: If 10 is added to each number, then the mean of the numbers also increases by 10. So the new mean would be 50. So the answer is (a).\n\nQ: If a / b = 3/4 and 8a + 5b = 22,then find the value of a.\nAnswer Choices:\n(a) 1/2\n(b) 3/2\n(c) 5/2\n(d) 4/2\n(e) 7/2\nA: If a / b = 3/4, then b = 4a / 3. So 8a + 5(4a / 3) = 22. This simplifies to 8a + 20a / 3 = 22, which means 44a / 3 = 22. So a is equal to 3/2. So the answer is (b).\n\nQ: How many keystrokes are needed to type the numbers from 1 to 500?\nAnswer Choices:\n(a) 1156\n(b) 1392\n(c) 1480\n(d) 1562\n(e) 1788\nA: There are 9 one-digit numbers from 1 to 9. There are 90 two-digit numbers from 10 to 99. There are 401 three-digit numbers from 100 to 500. 9 + 90(2) + 401(3) = 1392 So the answer is (b).\n\n",
}