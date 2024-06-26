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
def clean_text(x,remove_template=True):
    x = re.sub(r"(<math>|<\/math>|<cmath>|<\/cmath>|\\begin\{align\*\}|\\end\{align\*\})", "", x)
    if remove_template:
        x = x.replace("User: ","").replace("\n\nAssistant:","")
    return x
# RL data
with open(f"../llmOutputs/PRM/data_{version}.pickle", "rb") as f:
    data_ = pickle.load(f)
with open(f"../llmOutputs/PRM/completed_paths_y_code{version}.pickle", "rb") as f:
    completed_paths_y = pickle.load(f)

data_cleaned = []
for dict_level in data_:
    temp_dict = dict()
    for parent,children in dict_level.items():
        temp_dict[clean_text(parent)] = [clean_text(child) for child in children]
    data_cleaned.append(temp_dict)
data_ = data_cleaned

data = []
for y,score,text,code,prob_i,exit_i in completed_paths_y:
    data.append([clean_text(text,True),y])

import random
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-math-7b-rl")

def from_gen(values):
    random.shuffle(values)
    texts,ys = list(zip(*values))
    texts = tokenizer.batch_encode_plus(texts,return_attention_mask=False,add_special_tokens=True,\
                                    truncation=True, max_length=MAX_LEN)['input_ids']
    for text,y in zip(texts,ys):
        text = torch.tensor(text,device='cuda')[None]
        y = torch.tensor([y],device='cuda',dtype=torch.float32)
        yield text,y

import torch
import numpy as np

logit2prob = lambda x: 1/(1+np.exp(-x))
def update_steps(string, step):
    pattern = r'\[STEPS LEFT=(\d+)\]'
    new_pat = f'[STEPS LEFT={step}]' if step else '[completed]'
    return re.sub(pattern, new_pat, string)

def HasAnswer(text):
    patterns = [
        r'answer is.*\\boxed\{(.*?)\}',
        r"answer is[:\s]*\$(.+?)\$",
        r"answer is[:\s]*(.+)",
        r'print\((\d+)\)'
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return True
    return False

def extract_number(text):
    patterns = [
        r'answer is.*\\boxed\{(.*?)\}',
        r"answer is[:\s]*\$(.+?)\$",
        r"answer is[:\s]*(.+)",
        r'print\((\d+)\)'
    ]
    for pattern in patterns:
        match = list(re.finditer(pattern, text))
        if match:
            out = match[-1].group(1)
            try:
                out = float(out)
                return out
            except:
                return "error"
    return "error"

def IsFinished(node):
    if "No Python code." in node:
        return HasAnswer(node)
    else:
        matches = re.findall(r'print\(([^)]*)\)', node)
        return len(matches)>0

def get_value(parent,children,rem):
    # children is a list of continuations
    # return (parent: max value), (parent: [(child1,adv1),(child2,adv2),...])
    parent_minus1 = update_steps(parent,rem) # level-1 to eval the children
    # terminal level
    if rem == 0:
        values = []
        for child in children:
            text = parent_minus1 + child
            if IsFinished(text):
                input_ids = tokenizer.encode(text, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    hidden_states = base_model(input_ids)[0][:,-1].float() # 1,l,d -> 1,d
                    logits = model.score(hidden_states)[0]
                values.append(logit2prob(logits.item()))
            else:
                values.append(0.0)
    else:
        # non-terminal level
        input_ids = tokenizer.encode(parent_minus1, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = base_model(input_ids, use_cache=True)
            past_key_values = outputs.past_key_values

        values = []
        for child in children:
            input_child = tokenizer.encode(child, return_tensors="pt").to("cuda")
            with torch.no_grad():
                hidden_states = base_model(input_child, past_key_values=past_key_values, use_cache=True)[0][:,-1].float()
                logits = model.score(hidden_states)[0]
            values.append(logit2prob(logits.item()))
    
    v = np.mean(values)
    return (parent,max(values)), (parent,[(child,q-v) for child,q in zip(children,values)])
        
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
# base_model.gradient_checkpointing_enable()
# model.config.pad_token_id = tokenizer.pad_token_id
base_model.print_trainable_parameters()
model.score = model.score.float()
model.score.load_state_dict(torch.load(head_path))
model.score.weight.requires_grad_(True);

base_params = [param for param in base_model.parameters() if param.requires_grad]
trainable_params =  base_params + list(model.score.parameters())
                    # list(topic_model.parameters())
optimizer = torch.optim.Adam(trainable_params,lr = lr)
loss_fn = torch.nn.BCEWithLogitsLoss()
train_loss = 0
count_loss = 0

### train on completed path ###
i = 0
for text,y in from_gen(data):
    hidden_states = base_model(text)[0][:,-1].float() # 1,d
    logits = model.score(hidden_states)[:,0] # 1,
    loss = loss_fn(logits,y)
    loss.backward()
    train_loss += loss.item()
    count_loss += 1
    i += 1
       
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
print('intial training finished')

### backward training from ternimal values ###
policys = [] # [(parent,[(c1,adv1),...])]
for rem,dict_level in enumerate(reversed(data_)):
    values = [] # [(text,score),...]
    # get model target
    model.eval()
    for parent,children in dict_level.items():
        v,pi = get_value(parent,children,rem)
        values.append(v)
        policys.append(pi)
    
    # train value function
    model.train()
    for text,y in from_gen(values):
        hidden_states = base_model(text)[0][:,-1].float() # 1,d
        logits = model.score(hidden_states)[:,0] # 1,
        loss = loss_fn(logits,y)
        loss.backward()
        train_loss += loss.item()
        count_loss += 1
        i += 1
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
    print(f'level {rem} finshed')

#### save model
import pickle
with open(f"../llmOutputs/PRM/data_pi1_code{version}.pickle", "wb") as f:
    pickle.dump(policys, f)
    
torch.save(model.score.state_dict(), f'../Model/model_score{next_version}_code.pth')
peft_model_id = f"../Model/PRM_LORA{next_version}_code"
if not os.path.exists(peft_model_id):
    os.makedirs(peft_model_id)
base_model.save_pretrained(peft_model_id)
del model,base_model,text,hidden_states,logits,loss
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

# if train_loss/count_loss<0.37:
#     sys.exit(0)
# else:
#     print(f"train loss: {train_loss/count_loss}")
#     sys.exit(1)