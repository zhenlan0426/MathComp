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
clip_ratio = 0.07

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
ys2 = ys2/ys2.std()

# combined
ys = ys.tolist() + ys2.tolist()
for text,idx in zip(texts2,lengths_raw):
    question = tokenizer.encode(clean_text(text[:idx]),add_special_tokens=True)
    answer = tokenizer.encode(clean_text(text[idx:]),add_special_tokens=False)
    lengths.append(len(question))
    input_ids.append(question+answer)

data = list(zip(input_ids,ys,lengths))
random.shuffle(data)
input_ids,ys,lengths = list(zip(*data))

def from_gen(*data):
    for da in zip(*data,strict=True):
        if len(da) == 4:
            text = torch.tensor(da[0][:MAX_LEN],device='cuda')[None]
            logp_old = torch.tensor(da[1],device='cuda')
            yield text,logp_old,*da[2:]
        else:
            text = torch.tensor(da[0][:MAX_LEN],device='cuda')[None]
            yield text,*da[1:]

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

import torch.nn.functional as F
def logP_from_logits(logits, text):
    """
    Extracts log probabilities of the selected classes from logits.

    Args:
        logits (torch.Tensor): Logits of shape (l, C).
        text (torch.Tensor): Text of shape (l,), where each element is a class index.

    Returns:
        torch.Tensor: Log probabilities of shape (l,).
    """
    log_probs = F.log_softmax(logits, dim=-1)  # Normalize to log probabilities
    log_probs_of_text = log_probs.gather(1, text.unsqueeze(1)).squeeze(1) # Gather log probabilities using fancy indexing
    return log_probs_of_text
# get old logP
logP_list = []
for text,y,l in from_gen(input_ids,ys,lengths):
    with torch.no_grad():
        logits = model(text).logits[0,l:-1]
        logP = logP_from_logits(logits, text[0,l+1:]).cpu().numpy()
    assert (text.shape[1] - l - 1) == logP.shape[0]
    logP_list.append(logP)

def empty_cache():
    gc.collect()
    torch.cuda.empty_cache()

def loss_fn(logp,logp_old,adv,clip_ratio):
    ratio = torch.exp(logp - logp_old)
    clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
    loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()
    # approx_kl = (logp_old - logp).mean().item()
    return loss_pi

import math
import gc

train_loss = 0
count_loss = 0

for epoch in range(epochs):
    for i,(text,logP_old,adv,l) in enumerate(from_gen(input_ids,logP_list,ys,lengths)):
        if i > 0:
            del logits,logP,loss
            empty_cache()
        logits = model(text).logits[0,l:-1] # 1,l,C
        logP = logP_from_logits(logits, text[0,l+1:])
        loss = loss_fn(logP,logP_old,adv,clip_ratio)
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

del logits,logP,loss
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