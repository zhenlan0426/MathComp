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
neg_weight = 0.5
SFT_weight = 1
min_loss = -0.01

version = sys.argv[1]
MODEL_PATH = f"../Model/PRM_LORA{version}_merged_code_policy_01"
next_version = str(int(version) + 1)

#### Data
# separate out question and solution and only train on solution
def search_patterns(text):
    pattern = r"\n\nAssistant:"
    matches = list(re.finditer(pattern, text))
    # If there is one match, get the end position
    if matches:
        return matches[0].end()
    raise Exception("no match")
    
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-math-7b-rl")
import re
clean_text = lambda x:re.sub(r"(<math>|<\/math>|<cmath>|<\/cmath>|\\begin\{align\*\}|\\end\{align\*\})", "", x)

# {0,1}
with open(f"../llmOutputs/PRM/completed_paths_y_code{version}.pickle", "rb") as f:
    completed_paths_y = pickle.load(f)
data = []
for y,score,text,code,prob_i,exit_i in completed_paths_y:
    data.append([clean_text(text),y])
texts,ys = zip(*data)

ys = np.array(ys)
ys = (ys-ys.mean())/ys.std()
ys[ys<0] *= neg_weight

input_ids = []
lengths = []
for text in texts:
    idx = search_patterns(text)
    question = tokenizer.encode(text[:idx],add_special_tokens=True)
    answer = tokenizer.encode(text[idx:],add_special_tokens=False)
    lengths.append(len(question))
    input_ids.append(question+answer)

# Pi
with open(f"../llmOutputs/PRM/data_pi1_code{version}.pickle", "rb") as f:
    data_pi = pickle.load(f)
texts2,ys2,lengths_raw = zip(*data_pi)
ys2 = np.array(ys2)
ys2 = ys2/ys2.std() * len(texts) / len(texts2)

# combined
ys = ys.tolist() + ys2.tolist()
for text,idx in zip(texts2,lengths_raw):
    question = tokenizer.encode(clean_text(text[:idx]),add_special_tokens=True)
    answer = tokenizer.encode(clean_text(text[idx:]),add_special_tokens=False)
    lengths.append(len(question))
    input_ids.append(question+answer)

# SFT - Math
def gen_prompt_codeIn1(problem):
    return f"""User: {problem}\n
Determine a sympy-based approach for solving the problem. When defining symbol, incorporate all constraints mentioned in the problem statement, e.g. real, integer, even, odd, positive, prime. If a variable represents a positive integer, Symbol('n', integer=True, positive=True). Your final answer should be integer, not expression, list, tuple or dictionary!
Write the entire script covering all the steps (use comments and document it well) and print the final result.\n\nAssistant:
"""

def gen_prompt_codeIn2(problem):
    return f"""User: {problem}\n
You are an expert at solving math problem. Analyze this problem and think step by step to develop a python solution. Your solution should include reasoning steps in Python comments, explaining your thought process and the mathematical principles you applied. print the final output, as an integer not other python object such as list or tuple.\n\nAssistant:"""

def gen_prompt3(problem):
    return "User: "+problem+'''\n
Carefully read and understand the problem and use all information in problem statement. No Python code. Show your work step-by-step, explain your reasoning, calculations, mathematical concepts and formulas in detail.
Write your final answer as a single integer in the last line of your response, enclosed within \\boxed{}.\n\nAssistant:
'''

def add_prompt(problem):
    if np.random.rand()<0.5:
        return gen_prompt_codeIn1(problem)
    else:
        return gen_prompt_codeIn2(problem)
    
sft = pd.read_csv("../Data/MATH/math.csv")
# sft = sft.loc[sft.boxed_number == sft.parsed]
sft = sft.loc[(sft.boxed_number == sft.parsed) & (sft.level == 'Level 5')]
sft['code_wPrompt'] = sft.problem.apply(add_prompt)
for q,a in zip(sft.code_wPrompt.tolist(),sft.code_solution.tolist()):
    question = tokenizer.encode(clean_text(q),add_special_tokens=True)
    answer = tokenizer.encode(clean_text(a),add_special_tokens=False)
    lengths.append(len(question))
    input_ids.append(question+answer)
ys = ys + [SFT_weight] * sft.shape[0]

sft['pure_wPrompt'] = sft.problem.apply(gen_prompt3)
for q,a in zip(sft.pure_wPrompt.tolist(),sft.solution.tolist()):
    question = tokenizer.encode(clean_text(q),add_special_tokens=True)
    answer = tokenizer.encode(clean_text(a),add_special_tokens=False)
    lengths.append(len(question))
    input_ids.append(question+answer)
ys = ys + [SFT_weight] * sft.shape[0]

# SFT - AIME (prompt included). [9:] remove "Problem:"
with open(f"../Data/ai-mathematical-olympiad-prize/10prob.pickle", "rb") as f:
    outs = pickle.load(f)
with open(f"../Data/AMC/aime_final.pickle", "rb") as f:
    outs2 = pickle.load(f)
for q,a in outs:
    question = tokenizer.encode("User: "+clean_text(q[9:])+"\n\nAssistant:",add_special_tokens=True)
    answer = tokenizer.encode(clean_text(a),add_special_tokens=False)
    lengths.append(len(question))
    input_ids.append(question+answer)
ys = ys + [SFT_weight] * len(outs)

for q,a in outs2:
    question = tokenizer.encode("User: "+clean_text(q[9:])+"\n\nAssistant:",add_special_tokens=True)
    answer = tokenizer.encode(clean_text(a),add_special_tokens=False)
    lengths.append(len(question))
    input_ids.append(question+answer)
ys = ys + [SFT_weight] * len(outs2)

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
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-math-7b-rl")
tokenizer.save_pretrained(f"../Model/PRM_LORA{next_version}_merged_code_policy_01")