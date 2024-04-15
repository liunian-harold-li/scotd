# use py as the config file because yaml throws a tone of errors when parsing strings
cfg = {
  "TYPE": ("commonsense", ),
  "SOURCE": "hf",
  "CLEAN_STRATEGY": "general",
  "HF_IDENTIFIER": "drop",
  "HF_NAME": "default",
  "FEW_SHOT_PROMPT": (),

  "MODULE_NAME": "dataset.hf_datasets",
  "CLASS_NAME": "DropDataset",
  "INPUT_PREFIX": "Q: ",
  "STEP_PREFIX": "\nA: Let's think step by step. ", # this is the prompt for the answer
  "ANSWER_PREFIX": "\nThe answer is: ",
  "EXAMPLE_SUFFIX": "\n\n",

  "FEW_SHOT_COT_PROMPT": (
    "Coming off their Monday night road win over the Saints, the Vikings went home for a Week 6 NFC North duel with the Detroit Lions. In the first quarter, the Vikes got an early lead as Lions QB Dan Orlovsky unintentionally ran out of the back of his own end zone, giving Minnesota a safety which Orlovsky didn't know why the whistles were blown. In the second quarter, Detroit got the lead as kicker Jason Hanson got a 40-yard field goal. In the third quarter, the Lions increased their lead as Orlovsky completed a 12-yard TD pass to WR Calvin Johnson. The Vikings answered with QB Gus Frerotte completing an 86-yard TD pass to WR Bernard Berrian. In the end of the fourth quarter, the Vikes got in field goal range due to a controversial pass interference call on Leigh Bodden. The Vikes sealed the win with kicker Ryan Longwell nailing the game-winning 26-yard field goal. Who threw the longest pass?",

    "Orlorvsky completed a 12-yard TD pass. Gus Frerotte completed an 86-yard TD pass. Ryan Longwell nailed the 26-yard field goal. Of 12, 86, and 26, 86 is the longest. Thus, Gus Frerotte completed the longest pass in the game.",

    "Gus Frerotte",



    "As of the census of 2000, there were 14,702 people, 5,771 households, and 4,097 families residing in the county.  The population density was 29 people per square mile (11/km²).  There were 7,374 housing units at an average density of 14 per square mile (6/km²).  The racial makeup of the county was 98.02% Race (United States Census), 0.69% Race (United States Census) or Race (United States Census), 0.35% Race (United States Census), 0.11% Race (United States Census), 0.05% Race (United States Census), 0.08% from Race (United States Census), and 0.71% from two or more races.  0.44% of the population were Race (United States Census) or Race (United States Census) of any race. How many more households are there than families?",
    
    "There are 5771 households. There are 4097 families. We need to know how many more households there are than families. We need to calculate 5771 - 4097 = 1674.",
    
    "1674",

  ),

  "TRAIN": "train",
  "VALID": "validation",
}