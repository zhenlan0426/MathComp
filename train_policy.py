import torch
from torch.nn.utils import clip_grad_value_
import torch.optim as optim
import numpy as np
import pandas as pd
import pickle
import random
import re
from transformers import AutoTokenizer
import os
import sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"
MAX_LEN = 1200

version = sys.argv[1]
MODEL_PATH = f"../Model/PRM_LORA{version}_merged_code_policy_01"
next_version = str(int(version) + 1)

#### Data

# separate out question and solution and only train on solution
patterns = [r"``` and should only print the final answer.",\
            r"print the final result.\nApproach:",\
            r"print the final output, as an integer not other python object such as list or tuple."]

def search_patterns(text, patterns):
    for pattern in patterns:
        # Compile the pattern
        regex = re.compile(pattern)
        # Find all matches of the pattern in the text
        matches = list(regex.finditer(text))
        # If there is one match, get the end position
        if matches:
            return matches[0].end()
    
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-math-7b-rl")

# {1}
import pickle
with open(f"../llmOutputs/PRM/completed_paths_y_code{version}.pickle", "rb") as f:
    completed_paths_y = pickle.load(f)

texts = []
for y,score,text,code,prob_i,exit_i in completed_paths_y:
    if y == 1:
        texts.append(text.replace("<｜begin▁of▁sentence｜>User: ",""))

input_ids = []
lengths = []
for text in texts:
    idx = search_patterns(text,patterns)
    question = tokenizer.encode(text[:idx],add_special_tokens=True)
    answer = tokenizer.encode(text[idx:],add_special_tokens=False)
    lengths.append(len(question))
    input_ids.append(question+answer)

# Pi > 0
with open(f"../llmOutputs/PRM/data_pi1_code{version}.pickle", "rb") as f:
    data_pi = pickle.load(f)

for text,y,idx in data_pi:
    if y > 0:
        question = tokenizer.encode(text[:idx].replace("<｜begin▁of▁sentence｜>User: ",""),add_special_tokens=True)
        answer = tokenizer.encode(text[idx:],add_special_tokens=False)
        lengths.append(len(question))
        input_ids.append(question+answer)

l1 = len(texts)
l2 = len(input_ids) - l1
ys = [1] * l1 + [l1/l2] * l2

def from_gen(texts,ys,lengths):
    data = list(zip(texts,ys,lengths))
    random.shuffle(data)
    for text,y,l in data:
        text = torch.tensor(text[:MAX_LEN],device='cuda')[None]
        yield text,y,l


#### Model
epochs = 1
accumulation_steps = 64
verbose = 1024
lr = 2e-5
clip = 2e-3
from transformers import AutoModelForCausalLM,BitsAndBytesConfig
import torch
from peft import (
    get_peft_model,
    PeftType,
    LoraConfig)
quantization_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH,\
                                            device_map="auto",
                                            torch_dtype="auto",
                                            quantization_config=quantization_config,
                                            trust_remote_code=True,
                                            attn_implementation="flash_attention_2"
                                            )
model.gradient_checkpointing_enable()
peft_config = LoraConfig(r=8, # low rank 
                         lora_alpha = 16, # see below 
                         lora_dropout = 0.1, 
                         bias="none",#'none', 'all' or 'lora_only' 
                         target_modules = [ "q_proj", 
                                            "k_proj", 
                                            "v_proj", 
                                            "o_proj", 
                                            "gate_proj", 
                                            "up_proj", 
                                            "down_proj" 
                                        ],
                        #  use_dora=True,
                        )
model = get_peft_model(model, peft_config)
model.gradient_checkpointing_enable()
# model.config.pad_token_id = tokenizer.pad_token_id
model.print_trainable_parameters()
trainable_params = [param for param in model.parameters() if param.requires_grad]
optimizer = torch.optim.AdamW(trainable_params,lr = lr)

def empty_cache():
    gc.collect()
    torch.cuda.empty_cache()

import math
import gc
loss_fn = torch.nn.CrossEntropyLoss()
train_loss = 0
count_loss = 0

for epoch in range(epochs):
    for i,(text,y,l) in enumerate(from_gen(input_ids,ys,lengths)):
        if i > 0:
            del outs,loss
            empty_cache()
        outs = model(text).logits # 1,l,C
        loss = loss_fn(outs[0,l:-1],text[0,l+1:]) * y # (l,C), (l,)
        if math.isinf(loss.item()) or math.isnan(loss.item()): continue
        loss.backward()
        train_loss += loss.item()
        count_loss += 1
            
        if (i + 1) % accumulation_steps == 0:
            # clip_grad_value_(trainable_params,clip)
            clip_grad_value_(trainable_params,clip)
            optimizer.step()
            optimizer.zero_grad()

        if (i + 1) % verbose == 0:
            print(f"iter: {i}, \n train loss: {train_loss/count_loss}")
            train_loss = 0
            count_loss = 0

#### save model
next_version = str(int(version) + 1)
peft_model_id = f"../Model/PRM_LORA{next_version}_code_policy_01"
# !mkdir peft_model_id
model.save_pretrained(peft_model_id)

del model,texts,outs
gc.collect()
torch.cuda.empty_cache()
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH,\
                                    device_map="auto",
                                    torch_dtype="auto",
                                    attn_implementation="flash_attention_2"
                                    )
from peft import PeftModel
peft_model_id = f"../Model/PRM_LORA{next_version}_code_policy_01"
base_model = PeftModel.from_pretrained(model, peft_model_id)
base_model2 = base_model.merge_and_unload()
base_model2.save_pretrained(f"../Model/PRM_LORA{next_version}_merged_code_policy_01")