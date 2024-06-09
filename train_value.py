from transformers import LlamaForSequenceClassification
import torch
from torch.nn.utils import clip_grad_value_
import torch.optim as optim
import numpy as np
import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"
MAX_LEN = 1200

version = sys.argv[1]
Model_Path = f'../Model/PRM_LORA_merge{version}_code'
head_path = f'../Model/model_score{version}_code.pth'
next_version = str(int(version) + 1)

#### Data
import pickle
import re

def clean_text(x,remove_template):
    x = re.sub(r"(<math>|<\/math>|<cmath>|<\/cmath>|\\begin\{align\*\}|\\end\{align\*\})", "", x)
    if remove_template:
        x = x.replace("User: ","").replace("\n\nAssistant:","")
    return x

# RL data
with open(f"../llmOutputs/PRM/data_V1_code{version}.pickle", "rb") as f:
    data_V = pickle.load(f)
with open(f"../llmOutputs/PRM/completed_paths_y_code{version}.pickle", "rb") as f:
    completed_paths_y = pickle.load(f)

data = []
for text,y in data_V:
    data.append([clean_text(text,True),y])
for y,score,text,code,prob_i,exit_i in completed_paths_y:
    data.append([clean_text(text,True),y])

# SFT data # TODO: remove this later? if loss is too low, e.g. <0.1, overfit or topic classification
# AIME (prompt included)
with open(f"../Data/ai-mathematical-olympiad-prize/10prob.pickle", "rb") as f:
    outs = pickle.load(f)
with open(f"../Data/AMC/aime_final.pickle", "rb") as f:
    outs2 = pickle.load(f)
for q,s in outs:
    if np.random.rand()<0.5:
        data.append([clean_text(q+s,True),1])
for q,s in outs2:
    if np.random.rand()<0.25:
        data.append([clean_text(q+s,True),1])
import random
random.shuffle(data)
texts,ys = zip(*data)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-math-7b-rl")
texts = tokenizer.batch_encode_plus(texts,return_attention_mask=False,add_special_tokens=True,\
                                    truncation=True, max_length=MAX_LEN)['input_ids']
def from_gen(texts,ys):
    data = list(zip(texts,ys))
    random.shuffle(data)
    for text,y in data:
        text = torch.tensor(text,device='cuda')[None]
        y = torch.tensor([y],device='cuda',dtype=torch.float32)
        yield text,y
        
#### Model
epochs = 1
accumulation_steps = 64
verbose = 1024
lr = 6e-5
clip = 6e-3
from transformers import LlamaForSequenceClassification,BitsAndBytesConfig
import torch
from peft import (
    get_peft_model,
    LoraConfig)
quantization_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = LlamaForSequenceClassification.from_pretrained(Model_Path,\
                                                       num_labels=1,\
                                                       device_map="auto",
                                                       torch_dtype="auto",
                                                       quantization_config=quantization_config,
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
                                        ] 
                        )
base_model = get_peft_model(model.model, peft_config)
base_model.gradient_checkpointing_enable()
# model.config.pad_token_id = tokenizer.pad_token_id
base_model.print_trainable_parameters()
model.score = model.score.float()
model.score.load_state_dict(torch.load(head_path))
model.score.weight.requires_grad_(True);
base_params = [param for param in base_model.parameters() if param.requires_grad]
trainable_params = list(model.score.parameters())
                    # list(topic_model.parameters())
optimizer = torch.optim.Adam(trainable_params,lr = lr)
loss_fn = torch.nn.BCEWithLogitsLoss()
train_loss = 0
count_loss = 0
for epoch in range(epochs):
    for i,(text,y) in enumerate(from_gen(texts,ys)):
        with torch.no_grad():
            hidden_states = base_model(text)[0][:,-1].float() # 1,d
        logits = model.score(hidden_states)[:,0] # 1,
        loss = loss_fn(logits,y)
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
            
        torch.cuda.empty_cache()
base_params = [param for param in base_model.parameters() if param.requires_grad]
trainable_params =  base_params + list(model.score.parameters())
                    # list(topic_model.parameters())
optimizer = torch.optim.Adam(trainable_params,lr = lr)
loss_fn = torch.nn.BCEWithLogitsLoss()
train_loss = 0
count_loss = 0

for epoch in range(epochs):
    for i,(text,y) in enumerate(from_gen(texts,ys)):
        hidden_states = base_model(text)[0][:,-1].float() # 1,d
        logits = model.score(hidden_states)[:,0] # 1,
        loss = loss_fn(logits,y)
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
        torch.cuda.empty_cache()

#### save model
torch.save(model.score.state_dict(), f'../Model/model_score{next_version}_code.pth')
peft_model_id = f"../Model/PRM_LORA{next_version}_code"
if not os.path.exists(peft_model_id):
    os.makedirs(peft_model_id)
base_model.save_pretrained(peft_model_id)
del model,base_model,texts,hidden_states,logits,loss
import gc
gc.collect()
torch.cuda.empty_cache()
from transformers import LlamaModel
model = LlamaModel.from_pretrained(Model_Path,\
                                    device_map="auto",
                                    torch_dtype="auto",
                                    attn_implementation="flash_attention_2"
                                    )
from peft import PeftModel
peft_model_id = f"../Model/PRM_LORA{next_version}_code"
base_model = PeftModel.from_pretrained(model, peft_model_id)
base_model2 = base_model.merge_and_unload()
base_model2.save_pretrained(f'../Model/PRM_LORA_merge{next_version}_code')

if train_loss/count_loss<0.37:
    sys.exit(0)
else:
    print(f"train loss: {train_loss/count_loss}")
    sys.exit(1)