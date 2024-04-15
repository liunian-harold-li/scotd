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
    "Dialogue: M: Mary, tomorrow is your mom's fiftieth birthday. Do you know? W: Of course I do. How shall we celebrate it? M: First of all a birthday present. What about buying her a beautiful skirt? W: That's a good idea. It would make her look younger. And a big birthday cake too, with fifty candles. M: That's right. Shall we have a special dinner? W: How about a Chinese dinner? M: Fine. Where should we have it? W: We can have it at home. I've learned to cook a few dishes from a Chinese friend. I'm sure Mom would like them. M: All right. Are you going to do the shopping as well? W: Why don't we go together, Dad? M: OK. When? W: How about this afternoon?\nQuestion: Who'll cook the special dinner?\nChoices:\nMary.\nMary's friend.\nMary's mother.",

    "The question is about dinner. Mary (W) proposed to have a Chinese dinner. She then mentioned that she has learned to cook a few dishes from a Chinese friend. Thus, Mary will cook the special dinner.",

    "Mary.",

    "Dialogue: M: Good morning, I'm one of the students who rented your flat. It's 55 Park Road. W: Oh, yes. Everything all right? M: Not exactly. I'm afraid there are a couple of problems. W: Oh! I'm sorry to hear that. What kind of problems? M: Well, we haven't had any hot water for a couple of days now. I wonder if you could send someone to have a look at it. W: Of course. I'll get someone to come around at the weekend. M: Well, could he come around a bit sooner? I don't think we can manage until the weekend. W: I see. Okay. I'll send someone over this afternoon then. M: There's also the matter of the fridge. We all assumed there would be one in the flat when we moved in, because that's what we read from the advertisement in the newspaper. W: Ah, yes. Sorry about that. I got rid of the old fridge, but I didn't get around to ordering a new one yet. I'm really sorry. I'll order one today and get it delivered to you tomorrow. M: We bought one on the Internet actually. But could you pay us back? W: Of course. Just tell me how much you paid for it. M: It's 260 pounds. Thank you.\nQuestion: What will the woman do to settle the problem about the fridge?\nChoices:\nPay the students for the new one.\nGet someone to fix the old one.\nOrder one on the Internet.",

    "The question is about the fridge. The man complained about the matter of the fridge. The women said she will order one today and get it delivered to the man tomorrow. The man then said he bought one on the Internet and want the women to pay him back. The women said of course. Thus, the womaon will pay the man back for the new fridge.",

    "Pay the students for the new one.",

  )

    # 1e1d0ce1-b0ea-4ad8-9971-b2b44948123b: !Template
}