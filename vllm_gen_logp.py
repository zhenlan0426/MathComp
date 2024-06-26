from vllm import LLM, SamplingParams
LOCAL = True
from functions import *
dtype = 'auto'
gpu_memory_utilization = 0.95

import torch
import pandas as pd
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import subprocess
import sys

n = 5 # beams
n_sol = 7
samples = 5
max_depth = 16
max_pct = 0.8
temperature = 0.5
min_len = 77

version = sys.argv[1]
MODEL_PATH = f"../Model/PRM_LORA{version}_merged_code_policy_01" #_merged_code_policy_01SFT


#### Model
llm = LLM(model=MODEL_PATH,
          dtype=dtype,
          enforce_eager=True,
          gpu_memory_utilization=gpu_memory_utilization,
          swap_space=8,
          max_model_len=2048,
          kv_cache_dtype="fp8_e5m2",
          tensor_parallel_size=1)
tokenizer = llm.get_tokenizer()

# stop_words = [tokenizer.eos_token if tokenizer is not None and tokenizer.eos_token is not None else '</s>']
stop_words = [tokenizer.eos_token,"```output","```Output","```output\n","```Output\n","```\nOutput" , ")\n```" , "``````output","``````Output"]
# stop_words.append("\n")
sampling_params = SamplingParams(temperature=1,
                                 max_tokens=256,
                                #  min_tokens=32,
                                 stop=stop_words,
                                 include_stop_str_in_output=True,
                                 logprobs = 0,
                                 prompt_logprobs =0,)


def gen_prompt_codeIn1(problem):
    return f"""{problem}\n
Determine a sympy-based approach for solving the problem. When defining symbol, incorporate all constraints mentioned in the problem statement, e.g. real, integer, even, odd, positive, prime. If a variable represents a positive integer, Symbol('n', integer=True, positive=True). Your final answer should be integer, not expression, list, tuple or dictionary!
Write the entire script covering all the steps (use comments and document it well) and print the final result."""

def gen_prompt_codeIn2(problem):
    return f"""{problem}\n
You are an expert at solving math problem. Analyze this problem and think step by step to develop a python solution. Your solution should include reasoning steps in Python comments, explaining your thought process and the mathematical principles you applied. print the final output, as an integer not other python object such as list or tuple."""

def gen_prompt3(problem):
    return problem+'''\n
Carefully read and understand the problem and use all information in problem statement. No Python code. Show your work step-by-step, explain your reasoning, calculations, mathematical concepts and formulas in detail.
Write your final answer as a single integer in the last line of your response, enclosed within \\boxed{}.
'''

from transformers import LlamaForSequenceClassification
prm_tokenizer = tokenizer
prm_model = LlamaForSequenceClassification.from_pretrained(f'../Model/PRM_LORA_merge{version}_code',\
                                                    num_labels=1,\
                                                    device_map="cpu",
                                                    torch_dtype="auto",
                                                    ).eval()
base_model = prm_model.model
prm_model.score.load_state_dict(torch.load(f'../Model/model_score{version}_code.pth'))

#### Data
import json
with open('../Data/AMC/aime_normal.json', 'r') as file:
    df = json.load(file)
# to have consistent format as in Kaggle
df = pd.DataFrame(df)
df.rename(columns={'question': 'problem'}, inplace=True)
df.final_answer = df.final_answer.apply(lambda x:int(x[0]))
df2 = pd.read_csv("../Data/ai-mathematical-olympiad-prize/train.csv")
df2.rename(columns={'answer': 'final_answer'}, inplace=True)
df = pd.concat([df2,df[['problem','final_answer']]],axis=0)

#### functions
logit2prob = lambda x: 1/(1+np.exp(-x))
def clean_text(x,remove_template):
    x = re.sub(r"(<math>|<\/math>|<cmath>|<\/cmath>|\\begin\{align\*\}|\\end\{align\*\})", "", x)
    if remove_template:
        x = x.replace("User: ","").replace("\n\nAssistant:","")
    return x

def process_inputs(inputs):
    # inputs is a list of str
    outs = []
    for problem in inputs:
        problem = clean_text(problem,False)
        base_prompt1 = tokenizer.apply_chat_template([{"role": "user","content": gen_prompt_codeIn1(problem)}],tokenize=False,add_generation_prompt=True)
        base_prompt2 = tokenizer.apply_chat_template([{"role": "user","content": gen_prompt_codeIn2(problem)}],tokenize=False,add_generation_prompt=True)
        base_prompt3 = tokenizer.apply_chat_template([{"role": "user","content": gen_prompt3(problem)}],tokenize=False,add_generation_prompt=True)
        # 21: remove [bos], which will get added in vllm.generate
        outs.append(base_prompt1[21:])
        outs.append(base_prompt2[21:])
        outs.append(base_prompt3[21:])
    return outs

def eval_prm(candidates):
    all_log_probs = []
    for i in range(len(candidates)):
        text = clean_text(candidates[i],True)
        input_ids = prm_tokenizer.encode(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            hidden_states = base_model(input_ids)[0][:,-1] # 1,l,d -> 1,d
            logits = prm_model.score(hidden_states)[0]
        all_log_probs.append(logit2prob(logits.item()))
    return all_log_probs

def is_integer(num):
    if isinstance(num, float):
        return num.is_integer()
    elif isinstance(num, int):
        return True
    else:
        return False
    
def is_between_0_and_999(num):
    return 0 <= num <= 999

import re
def extract_number(text):
    patterns = [
        r'[Tt]he answer is.*\\boxed\{(.*?)\}',
        r"[Tt]he answer is[:\s]*\$([0-9]+)\$",
        r"[Tt]he answer is[:\s]*([0-9]+)"
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return 'parse err'

def group_and_sum(A, B):
    '''
    A = ['a','b','a']
    B = [1,2,3]
    -> {'a': 4, 'b': 2}
    '''
    result_dict = {}
    for a, b in zip(A, B):
        if a in result_dict:
            result_dict[a] += b
        else:
            result_dict[a] = b
    return result_dict

def group_and_average(A, B):
    from collections import defaultdict
    # Create a dictionary to store sums and counts for averaging
    sum_dict = defaultdict(lambda: [0, 0])  # Each key maps to [sum, count]
    # Pair elements from A and B and aggregate sums and counts
    for key, value in zip(A, B):
        sum_dict[key][0] += value
        sum_dict[key][1] += 1
    # Calculate averages
    averages = {key: sum_count[0] / sum_count[1] for key, sum_count in sum_dict.items()}
    return averages,[averages[a] for a in A]

def max_dict(d):
    return max(d.items(), key=lambda x: x[1])[0]

def tot_agg(completed_paths):
    if completed_paths:
        answers,scores = zip(*completed_paths)
        groups = group_and_sum(answers, scores)
        return max_dict(groups)
    else:
        return 37 # empty completed_paths
    
def repeat_elements(lst, k):
    return [i for i in lst for _ in range(k)]

def flatten(nested_list):
    """Flatten a nested list."""
    out = []
    lengths = []
    for sublist in nested_list:
        lengths.append(len(sublist))
        for item in sublist:
            out.append(item)
    return out,lengths

def unflatten(flat_list, lengths):
    """Unflatten a flat list into a nested list based on lengths."""
    nested_list = []
    index = 0
    for length in lengths:
        nested_list.append(flat_list[index:index + length])
        index += length
    return nested_list

dict2val = lambda d:next(iter(d.values())).logprob
out2logp = lambda out:[dict2val(d) for d in out.prompt_logprobs[1:]] + [dict2val(d) for d in out.outputs[0].logprobs] # start from second token, skip [BOS]

def filter_input(batch_response,current_level_node):
    # one question filter
    prm_inputs = []
    parents = []
    logps = []
    for candidate,parent in zip(batch_response,current_level_node):
        if candidate.outputs[0].text not in parent:
            prm_input = parent + candidate.outputs[0].text
            prm_inputs.append(prm_input)
            parents.append(parent)
            logps.append(out2logp(candidate))
    # Get the indices of unique elements in prm_inputs
    unique_indices = [i for i, x in enumerate(prm_inputs) if prm_inputs.index(x) == i]
    prm_inputs = [prm_inputs[i] for i in unique_indices]
    parents = [parents[i] for i in unique_indices]
    logps = [logps[i] for i in unique_indices]
    return prm_inputs,parents,len(prm_inputs),logps

def filter_inputs(batch_responses,current_level_nodes,lengths):
    # all question filter
    # returned value should be flattened
    batch_responses,current_level_nodes = unflatten(batch_responses,lengths),unflatten(current_level_nodes,lengths)
    prm_inputs = []
    lengths = []
    parent_list = []
    logp_list = []
    uncompleted = [path for path in completed_paths if len(path)<n_sol]
    assert len(batch_responses) == len(uncompleted)
    for batch_response,current_level_node,path in zip(batch_responses,current_level_nodes,uncompleted):
        prm_input,parents,length,logps = filter_input(batch_response,current_level_node)
        if length == 0:# all bad
            while len(path)<n_sol:
                # make complete for this question as there will be no continued effort
                path.append(None)
        else:
            prm_inputs.extend(prm_input)
            parent_list.extend(parents)
            logp_list.extend(logps)
            lengths.append(length)
    return prm_inputs,parent_list,lengths,logp_list

def HasAnswer(text):
    patterns = [
        r'answer is.*\\boxed\{(.*?)\}',
        r"answer is[:\s]*\$(.+?)\$",
        r"answer is[:\s]*(.+)"
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
        r"answer is[:\s]*(.+)"
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

def sample_k(items, probabilities, k):
    """Samples k items without replacement from a list based on probabilities, with temperature scaling."""
    # Temperature scaling
    scaled_probs = np.exp(np.array(probabilities)/temperature)
    normalized_probs = scaled_probs / np.sum(scaled_probs)  # Normalize scaled probs

    # Sampling
    sampled_items = np.random.choice(
        items, size=k, replace=False, p=normalized_probs
    )
    return sampled_items

def get_next_node(prm_inputs,prm_scores,completed_paths,logps):
    # need to update completed_paths in-place
    next_level_nodes = []
    next_level_scores = []
    combined = list(zip(prm_inputs,prm_scores,logps))    
    for node,score,logp in combined:
        finish = IsFinished(node)
        if finish: # finished
            if len(node.split("Assistant:")[1]) > min_len:
                completed_paths.append((score,node,logp))
        else: # not inished
            next_level_nodes.append(node)
            next_level_scores.append(score)
    if len(next_level_nodes) < n:
        return next_level_nodes
    next_level_nodes = sample_k(next_level_nodes, next_level_scores, n)
    return next_level_nodes


def get_next_nodes(prm_inputs,prm_scores,lengths,logp_list):
    # for completed_paths, next_level_nodes would be removed
    # returned value should be flattened
    prm_inputs,prm_scores,logps = unflatten(prm_inputs,lengths),unflatten(prm_scores,lengths),unflatten(logp_list,lengths)
    uncompleted = [path for path in completed_paths if len(path)<n_sol]
    assert len(uncompleted) == len(lengths)
    assert len(prm_inputs) == len(lengths)
    assert len(prm_scores) == len(lengths)
    assert len(logps) == len(lengths)
    next_level_nodes,lengths = [],[]
    for prm_input,prm_score,completed_path,logp in zip(prm_inputs,prm_scores,uncompleted,logps):
        next_node = get_next_node(prm_input,prm_score,completed_path,logp)
        if len(completed_path) < n_sol:
            next_level_nodes.extend(next_node)
            lengths.append(len(next_node))
    return next_level_nodes,lengths

import gc
def create_llm():
    gc.collect()
    torch.cuda.empty_cache()
    llm = LLM(model=MODEL_PATH,
          dtype=dtype,
          enforce_eager=True,
          gpu_memory_utilization=gpu_memory_utilization,
          swap_space=8,
          max_model_len=2048,
          kv_cache_dtype="fp8_e5m2",
          tensor_parallel_size=1)
    tokenizer = llm.get_tokenizer()
    return llm,tokenizer

#### generation
current_level_nodes = process_inputs(df.problem.tolist())
lengths = [3] * df.shape[0]
current_level = 1
completed_paths = [[] for _ in range(df.shape[0])]
data_V = []
data_pi = []

while (current_level < max_depth) and (current_level_nodes):
    # everything at this level is flattened
    current_level_nodes = repeat_elements(current_level_nodes,samples)
    lengths = [l*samples for l in lengths]
    batch_responses = llm.generate(current_level_nodes, sampling_params)
    prm_inputs,parent_list,lengths,logp_list = filter_inputs(batch_responses,current_level_nodes,lengths)
    
    # release VRAM to prm_model
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    prm_model.to('cuda')
    prm_scores = eval_prm(prm_inputs)
        
    # save for Q-learning
    averages,averages_dup = group_and_average(parent_list,prm_scores)
    data_V.extend(list(averages.items()))
    advantages = [q-v for q,v in zip(prm_scores,averages_dup)]
    data_pi.extend(list(zip(prm_inputs,advantages,[len(p) for p in parent_list],logp_list))) # pi(a|s) only train on action part
    
    # release VRAM to llm
    prm_model.to('cpu')
    llm,tokenizer = create_llm()
    
    current_level_nodes,lengths = get_next_nodes(prm_inputs,advantages,lengths,logp_list)
    current_level += 1

#### exec code
import os
import glob
def delete_py_files(folder):
    # Use glob to find all .py files in the folder and subfolders
    py_files = glob.glob(os.path.join(folder, '**', '*.py'), recursive=True)
    # Iterate over the list of .py files and delete each one
    for file_path in py_files:
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
# Example usage
folder_path = 'temp'
delete_py_files(folder_path)
def repl(match):
    if "real" not in match.group():
        return "{}{}".format(match.group()[:-1], ', real=True)')
    else:
        return "{}{}".format(match.group()[:-1], ')')
    
from multiprocessing import Pool
from itertools import chain

def extract_code(text):
    text = text.split("Assistant:")[1]
    match = re.search(r"print\(.+?\)", text)  # Non-greedy match within parentheses
    if match:
        return text[:match.end()].strip()
    raise Exception("no match")

def process_paths(args):
    paths, y, idx = args
    paths = [p for p in paths if p]
    out = [] # (isCorrect,score,node,code,prob_i,exit_i,logp)
    for j,path in enumerate(paths):# path (score,node,logp)
        input = path[1]
        if "No Python code." in input:
            yhat = extract_number(input)
            if yhat == "error":
                out.append([0,path[0],path[1],'no code',idx,7,path[2]])
            else:
                out.append([int(y==yhat),path[0],path[1],'no code',idx,8,path[2]])
        else: # code
            if input[-12:]=="print(result": # stop token was not included. print(result) might miss a ")"
                input += ")"
            splits = input.split('```')
            if len(splits) < 2:
                try:
                    code = "from sympy import *\n" + extract_code(input) # not delimited by ```python
                    node = input
                except:
                    out.append([0,path[0],path[1],'no code',idx,1,path[2]])
                    continue
            else:
                code = "from sympy import *\n" + input.split('```')[1][7:] 
                node = '```'.join(splits[:4]) # only return up to the first python code. later code/reason not relevant
            # execute code
            with open(f'temp/code_{idx}_{j}.py', 'w') as fout:
                fout.write(code)
            # timeout err
            try:
                process = subprocess.run([sys.executable, f'temp/code_{idx}_{j}.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=7.1)
            except subprocess.TimeoutExpired:
                out.append([0,path[0],node,code,idx,2,path[2]])
                with open(f'temp/2/code_{idx}_{j}.py', 'w') as fout:
                    fout.write(code)
                continue
            if process.stderr:# code.py err
                out.append([0,path[0],node,code,idx,3,path[2]])
                code = code + '\n\n"""' + process.stderr.decode('utf-8') + '"""'
                with open(f'temp/3/code_{idx}_{j}.py', 'w') as fout:
                    fout.write(code)           
                continue
            else:
                stdout = process.stdout.decode('utf8')
                try:
                    answer = eval(stdout)
                    if is_integer(answer):
                        out.append([int(int(answer)==y),path[0],node,code,idx,4,path[2]])
                        with open(f'temp/4/code_{idx}_{j}.py', 'w') as fout:
                            fout.write(code)                     
                        continue
                    else:
                        out.append([0,path[0],node,code,idx,5,path[2]])
                        code = code + '\n\n"""' + stdout + '"""'
                        with open(f'temp/5/code_{idx}_{j}.py', 'w') as fout:
                            fout.write(code)          
                        continue
                except:
                    out.append([0,path[0],node,code,idx,6,path[2]])
                    code = code + '\n\n"""' + stdout + '"""'
                    with open(f'temp/6/code_{idx}_{j}.py', 'w') as fout:
                        fout.write(code)
                    continue
    return out

# Prepare arguments for multiprocessing
ys = df.final_answer.tolist()
arguments = [(paths, y, idx) for idx, (paths, y) in enumerate(zip(completed_paths, ys))]
with Pool(processes=16) as pool:
    results = pool.map(process_paths, arguments)
completed_paths_y = list(chain(*results))

#### Save outputs
import pickle
with open(f"../llmOutputs/PRM/data_V1_code{version}.pickle", "wb") as f:
    pickle.dump(data_V, f)
with open(f"../llmOutputs/PRM/data_pi1_code{version}.pickle", "wb") as f:
    pickle.dump(data_pi, f)    
with open(f"../llmOutputs/PRM/completed_paths_code{version}.pickle", "wb") as f:
    pickle.dump(completed_paths, f)
with open(f"../llmOutputs/PRM/completed_paths_y_code{version}.pickle", "wb") as f:
    pickle.dump(completed_paths_y, f)

    
# performance report
import csv
data = pd.DataFrame([paths[:-1] for paths in completed_paths_y],columns=['isCorrect','score','node','code','prob_i','exit_i'])
data = data.sort_values(by=['prob_i', 'score'], ascending=False)
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(data.iloc[:,0].values,data.iloc[:,1].values)
mean_acc = data.groupby('prob_i')['isCorrect'].first().mean() * 985
max_acc = data.groupby(['prob_i']).isCorrect.max().sum()
value_counts = json.dumps(data.exit_i.value_counts().to_dict())
log_data = pd.read_csv('training_log.csv')
_,_,best_mean_acc,best_max_acc,_ = log_data.iloc[-1].tolist()
with open('training_log.csv', mode='a', newline='') as log_file:
    log_writer = csv.writer(log_file)
    log_writer.writerow([version, auc, mean_acc, max_acc, value_counts])
if mean_acc > best_mean_acc or max_acc > best_max_acc:
    sys.exit(0)
else:
    print("no improve!")
    sys.exit(1)
