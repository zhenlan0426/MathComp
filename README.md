# MathComp  
[Kaggle Competition: AI Mathematical Olympiad Prize](https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize)  

This repository contains code for my approach to solving complex mathematical problems in LaTeX format using reinforcement learning (RL) and large language models (LLMs).  

## Overview  
The goal of this competition is to develop algorithms and models capable of solving challenging math problems. My approach involves iteratively improving a value function and policy using a reinforcement learning loop similar to that of openAI O1/O3 and DeepSeek R1.

## Repository Contents  
- **`rl-iterations.sh`** – Bash script for running the RL training loop.  
- **`vllm_gen.py`** – Generates reasoning paths using beam search, guided by a value function and policy.  
- **`train_value.py`** – Trains the value function using both completed and intermediate reasoning paths.  
- **`train_policy.py`** – Trains the policy using both completed and intermediate reasoning paths.  

