# from peft import (
#     get_peft_model,
#     PeftType,
#     PrefixTuningConfig,
#     PromptEncoderConfig,
#     # PromptTuningConfig,
#     LoraConfig)
import numpy as np
import torch
from transformers import LogitsProcessor
import random
import os
import json
import re
from collections import Counter

get_name = lambda x: x.split('/')[-1]
# config_map = {'PrefixTuningConfig':PrefixTuningConfig,'PromptEncoderConfig':PromptEncoderConfig,'LoraConfig':LoraConfig}

# def random_peft_config():
#     config_type = np.random.choice(['PrefixTuningConfig','PromptEncoderConfig','LoraConfig'])
#     if config_type == 'PrefixTuningConfig':
#         config_kwargs = dict(task_type = 'CAUSAL_LM',
#                             num_virtual_tokens = int(np.random.choice([8,16,24])), 
#                             encoder_hidden_size = int(np.random.choice([512,1024])), 
#                             prefix_projection = np.random.choice([True,False]),
#                             )
#     elif config_type == 'PromptEncoderConfig':
#         config_kwargs = dict(task_type = 'CAUSAL_LM', 
#                             num_virtual_tokens = int(np.random.choice([8,16,24])), 
#                             encoder_hidden_size = int(np.random.choice([1024,2048,4096])),
#                             encoder_dropout = float(np.random.rand()*0.25),
#                             encoder_num_layers = int(np.random.choice([1,2])),
#                             encoder_reparameterization_type = np.random.choice(['MLP','LSTM'])
#                             )
#     elif config_type == 'LoraConfig':
#         config_kwargs = dict(r=int(np.random.choice([8,16])),
#                             lora_alpha = 16,
#                             lora_dropout = float(np.random.rand()*0.25), 
#                             bias=np.random.choice(['none', 'all' , 'lora_only' ]),
#                             target_modules = ["q_proj","k_proj", "v_proj","o_proj"] if np.random.rand()<0.5 else ["q_proj","k_proj", "v_proj","o_proj","gate_proj", "up_proj", "down_proj" ]
#                             )
#     return {'config_type':config_type,'config_kwargs':config_kwargs}

def sample_consecutive_chunk(input_list, max_length):
    if len(input_list) <= max_length:
        return input_list
    max_start_index = len(input_list) - max_length
    start_index = random.randint(0, max_start_index)
    out = input_list[start_index:start_index + max_length]
    out[0] = input_list[0] # Start of sentence has to be included
    return out

def create_next_model_folder(base_path="../Model/FT"):
    # List all items in the base directory
    all_items = os.listdir(base_path)
    # Filter for folders that follow the naming pattern "modelX"
    model_folders = [item for item in all_items if item.startswith("model") and os.path.isdir(os.path.join(base_path, item))]
    # Extract the numbers from these folder names, default to 0 if no folders exist
    model_numbers = [int(folder.replace("model", "")) for folder in model_folders]
    highest_number = max(model_numbers) if model_numbers else 0
    # Define the new folder name and path
    new_folder_name = f"model{highest_number + 1}"
    new_folder_path = os.path.join(base_path, new_folder_name)
    # Create the new folder
    os.makedirs(new_folder_path, exist_ok=True)
    # Return the new folder path
    return new_folder_path
    
def clean_author(text):
    # Trim trailing newline characters
    text = text.rstrip('\n')
    # Split the text into lines
    lines = text.split('\n')
    # Check if the last line starts with '~' and remove it if it does
    # this remove author at the end
    if lines and lines[-1].startswith('~'):
        lines = lines[:-1]
    # Join the lines back into a text string
    cleaned_text = '\n'.join(lines)
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.(com|org|net|edu)\b'
    # Replace all occurrences of the email pattern with an empty string
    cleaned_text = re.sub(email_pattern, '', cleaned_text)
    return cleaned_text

class sentence_transformers(object):
    def __init__(self,model,tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.cos_sim = torch.nn.CosineSimilarity()
    
    @staticmethod
    def transform_query(query: str) -> str:
        """ For retrieval, add the prompt for query (not for documents)."""
        return f'Represent this sentence for searching relevant passages: {query}'

    def encode(self,docs,IsQuery=False,batch_size=32):
        # docs are a list of string
        out = []
        if IsQuery:
            docs = [self.transform_query(d) for d in docs]
        for i in range(0,len(docs),batch_size):
            inputs = self.tokenizer.batch_encode_plus(docs[i:i+batch_size], \
                                                 padding=True, \
                                                 return_attention_mask=True,\
                                                 truncation=True,\
                                                 max_length=self.model.config.max_position_embeddings,\
                                                 return_tensors='pt').to(self.model.device)
            with torch.no_grad():
                out.append(self.model(**inputs).last_hidden_state[:,0].detach().cpu().numpy())
        return np.concatenate(out)
    
    def get_loss(self,context,query,score):
        # context,query are outputs from tokenizer.batch_encode_plus
        # score is max(P_ij - P_j,0) the incremental success rate with/wo context
        context_v = self.model(**context).last_hidden_state[:,0] # (batch,d)
        query_v = self.model(**query).last_hidden_state[:,0]
        logits = self.cos_sim(context_v,query_v)
        loss = self.loss_fn(logits,score)
        return loss
    

class DigitsOnlyLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer):
        # 102400 is the out_features of lm_head
        super().__init__()
        allowed_token_ids = [tokenizer.convert_tokens_to_ids(str(digit)) for digit in range(10)]
        eos_token = tokenizer.eos_token_id
        if eos_token is not None:
            allowed_token_ids.append(eos_token)
        self.token_mask = torch.full((102400,), -float('Inf'), dtype=torch.float, device='cuda')
        self.token_mask[allowed_token_ids] = 0
    
    def __call__(self, input_ids, scores):
        # Mask logits for tokens that are not allowed
        scores += self.token_mask
        return scores
    
class NoRepeatTokenLogitsProcessor(LogitsProcessor):
    def __init__(self):
        super().__init__()
        # Create a set of token IDs for the disallowed tokens
        self.disallowed_token_ids = {185,207,10,12,9} # "\n"," ", +, -, *
        
    def __call__(self, input_ids, scores):
        if input_ids.shape[1] > 0:
            last_token_id = input_ids[:, -1].item()
            if last_token_id in self.disallowed_token_ids:
                scores[:, last_token_id] = float('-inf')
        return scores
    
def naive_parse(answer):
    out = []
    start = False
    end = False
    for l in reversed(list(answer)):
        if l in '0123456789' and not end:
            start = True
            out.append(l)
        else:
            if start:
                end = True
        
    out = reversed(out)
    return int(''.join(out))

def gen_prompt(problem):
    
    return f"""
### Instruction:\n{problem}\n\n
### Response: Let's think step by step. The final response should be a single number in the last line of your response.
"""

def gen_code(problem,solution):
    
    return f"""
### Instruction:\n{problem}\n
### Solution:\n{solution}\n
### Response: Let's think step by step. Given the Instruction and solution, write python code to execute the calculation. The code should be
enclosed between ```python\n actual code...``` and should only print the final answer.
"""

def aggregate(answers):
    pred = Counter(answers).most_common(2)
    if len(pred) == 1:
        if pred[0][0] == "parsing error":
            return 37
        else:
            return pred[0][0]
    else:
        return pred[1][0] if pred[0][0] == "parsing error" else pred[0][0]

def logprob_agg(answers,normalize):
    # answers [(answer, cum_logprob, lens),...]
    def compare(answer_tuple):
        answer, cum_logprob, lens = answer_tuple
        if answer == 'parsing error':
            return float('-inf')
        elif normalize:
            return cum_logprob/lens
        else:
            return cum_logprob
    return max(answers,key=compare)[0]