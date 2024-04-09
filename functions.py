from peft import (
    get_peft_model,
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig,
    # PromptTuningConfig,
    LoraConfig)
import numpy as np
import random
import os
import json
import re

config_map = {'PrefixTuningConfig':PrefixTuningConfig,'PromptEncoderConfig':PromptEncoderConfig,'LoraConfig':LoraConfig}

def random_peft_config():
    config_type = np.random.choice(['PrefixTuningConfig','PromptEncoderConfig','LoraConfig'])
    if config_type == 'PrefixTuningConfig':
        config_kwargs = dict(task_type = 'CAUSAL_LM',
                            num_virtual_tokens = int(np.random.choice([8,16,24])), 
                            encoder_hidden_size = int(np.random.choice([512,1024])), 
                            prefix_projection = np.random.choice([True,False]),
                            )
    elif config_type == 'PromptEncoderConfig':
        config_kwargs = dict(task_type = 'CAUSAL_LM', 
                            num_virtual_tokens = int(np.random.choice([8,16,24])), 
                            encoder_hidden_size = int(np.random.choice([1024,2048,4096])),
                            encoder_dropout = float(np.random.rand()*0.25),
                            encoder_num_layers = int(np.random.choice([1,2])),
                            encoder_reparameterization_type = np.random.choice(['MLP','LSTM'])
                            )
    elif config_type == 'LoraConfig':
        config_kwargs = dict(r=int(np.random.choice([8,16])),
                            lora_alpha = 16,
                            lora_dropout = float(np.random.rand()*0.25), 
                            bias=np.random.choice(['none', 'all' , 'lora_only' ]),
                            target_modules = ["q_proj","k_proj", "v_proj","o_proj"] if np.random.rand()<0.5 else ["q_proj","k_proj", "v_proj","o_proj","gate_proj", "up_proj", "down_proj" ]
                            )
    return {'config_type':config_type,'config_kwargs':config_kwargs}

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

# class JSONEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, (np.integer, np.floating, np.ndarray)):
#             return obj.item() if obj.shape == () else obj.tolist()
#         return super(JSONEncoder, self).default(obj)
    
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