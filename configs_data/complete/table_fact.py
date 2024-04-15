# use py as the config file because yaml throws a tone of errors when parsing strings
cfg = {
  "TYPE": ("commonsense", ),
  "SOURCE": "hf",
  "CLEAN_STRATEGY": "general",
  "HF_IDENTIFIER": "tab_fact",
  "HF_NAME": "tab_fact",
  "FEW_SHOT_PROMPT": (),

  "MODULE_NAME": "dataset.hf_datasets",
  "CLASS_NAME": "TableFactDataset",
  "INPUT_PREFIX": "Q: ",
  "STEP_PREFIX": "\nA: Let's think step by step. ", # this is the prompt for the answer
  "ANSWER_PREFIX": "\nThe answer is: ",
  "EXAMPLE_SUFFIX": "\n\n",

  "FEW_SHOT_COT_PROMPT": (
    "Table caption: list of boston celtics broadcasters\nTable Contents:\n year\tflagship station\tplay - by - play\tcolor commentator (s)\tstudio host\n2009 - 10\tweei\tsean grande\tcedric maxwell\tjohn ryder\n2008 - 09\tweei\tsean grande\tcedric maxwell\tjohn ryder\n2007 - 08\tweei\tsean grande\tcedric maxwell\tjohn ryder\n2006 - 07\twrko\tsean grande\tcedric maxwell\tjimmy myers and jimmy young\n2005 - 06\twrko\tsean grande\tcedric maxwell\tjimmy myers and jimmy young\n2004 - 05\twwzn\tsean grande\tcedric maxwell\tjimmy myers\n2003 - 04\twwzn\tsean grande\tcedric maxwell\tjimmy myers\n2002 - 03\twwzn\tsean grande\tcedric maxwell\tjimmy myers\n2001 - 02\twwzn\tsean grande\tcedric maxwell\tdave jageler & marty tirrell\n2000 - 01\tweei\thoward david\tcedric maxwell\tted sarandis\nIs the following statement correct ? jimmy myers studio host in the year 2004 - 05 have play - by - play of sean grande\n",

    "In the year 2004 - 05 (the first collum), jimmy myers studio has "
  ),

  "TRAIN": "train",
  "VALID": "validation",
}