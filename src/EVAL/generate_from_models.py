import pandas as pd
import numpy as np
import os
from collections import namedtuple
import torch
from tqdm import tqdm
import warnings
import json
import openai
import time
import os
import transformers
import sys
import pathlib 
import argparse
from peft import AutoPeftModelForCausalLM

sys.path.append('src/')  
from DPO.utils import save_to
from utils import generate
warnings.filterwarnings("ignore")

GENERATION_KWARGS = {'max_new_tokens': 30, 'no_repeat_ngram_size': 2, 'do_sample': True, 'min_new_tokens': 5, 'top_p': 0.75}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sft-path', required=True)
    parser.add_argument('--dpo-path', required=True)
    parser.add_argument('--orpo-path', required=True)
    parser.add_argument('--ppo-path', required=True)
    parser.add_argument('--model-name', required=True)
    args = parser.parse_args()
    
    sft = AutoPeftModelForCausalLM.from_pretrained(args.sft_path, device_map='auto')
    dpo = AutoPeftModelForCausalLM.from_pretrained(args.dpo_path, device_map='auto')
    orpo = AutoPeftModelForCausalLM.from_pretrained(args.orpo_path, device_map='auto')
    ppo = AutoPeftModelForCausalLM.from_pretrained(args.ppo_path, device_map='auto')
    
    ### SFT and DPO have the same tokenizer -- but not sure for the other two
    sft_tokenizer = transformers.AutoTokenizer.from_pretrained(args.sft_path)
    orpo_tokenizer = transformers.AutoTokenizer.from_pretrained(args.orpo_path)
    ppo_tokenizer = transformers.AutoTokenizer.from_pretrained(args.ppo_path)
    
    test_set = pd.read_json('data/argumentation/test_cckg.json')
    
    sft_args = []
    dpo_args = []
    orpo_args = []
    ppo_args = []
    for i, entry in tqdm(test_set.iterrows(), total=len(test_set)):
        topic = entry.topic
        stance = 'SUPPORTING' if entry.label == 1 else 'COUNTER'
        prompt = f"<s> [INST] ### Prompt:  Generate a {stance} argument for the topic: {topic} [/INST]\n### Argument: "
        
        with torch.no_grad():
            y_sft = generate(prompt, sft, sft_tokenizer, **GENERATION_KWARGS)
            y_dpo = generate(prompt, dpo, sft_tokenizer, **GENERATION_KWARGS)
            y_orpo = generate(prompt, orpo, orpo_tokenizer, **GENERATION_KWARGS)
            y_ppo = generate(prompt, ppo, ppo_tokenizer, **GENERATION_KWARGS)
        
        sft_args.append(y_sft.split('### Argument: ')[-1].strip())
        dpo_args.append(y_dpo.split('### Argument: ')[-1].strip())
        orpo_args.append(y_orpo.split('### Argument: ')[-1].strip())
        ppo_args.append(y_ppo.split('### Argument: ')[-1].strip())
        
        if i % 100 == 0:
            print("SFT:\n", y_sft, '\n')
            print("DPO:\n", y_dpo, '\n')
            print("ORPO:\n", y_orpo, '\n')
            print("PPO:\n", y_ppo, '\n')
            
    save_to(sft_args, name='sft_args.json', output_dir=f'results/{args.model_name}/')
    save_to(dpo_args, name='dpo_args.json', output_dir=f'results/{args.model_name}/')
    save_to(orpo_args, name='orpo_args.json', output_dir=f'results/{args.model_name}/')
    save_to(ppo_args, name='ppo_args.json', output_dir=f'results/{args.model_name}/')
        
if __name__ == "__main__":
    main()