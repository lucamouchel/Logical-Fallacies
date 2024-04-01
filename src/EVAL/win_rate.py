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
from DPO.env import OPENAI_API_KEY
from DPO.utils import save_to, get_gpt_response
from utils import generate

warnings.filterwarnings("ignore")
openai.api_key = OPENAI_API_KEY

GENERATION_KWARGS = {'max_new_tokens': 30, 'no_repeat_ngram_size': 2, 'do_sample': True}

def compare_models(stance, topic, arguments):
    possible_answers = ''
    for i, argument in enumerate(arguments):
        possible_answers += f"{i+1}. {argument}\n"
    prompt = f"""Which of these {stance} arguments is better for the topic: {topic} Please consider the fact that some of these arguments might be in the form of logical fallacies. If one of the arguments is a logical fallacy, it should not be the best. 
    \nArguments: \n{possible_answers}\nIf the arguments are equally good, return number {len(arguments) + 1}.\nThe better argument is number:"""
    
    response = get_gpt_response(prompt, necessary_tokens=1, model='gpt-4')
    try:
        response = int(response)
        return response
    except:
        print("Invalid response. Please try again.")
        print(arguments)
        print()
        print(response)
        return len(arguments) + 1 ### TIE
   
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
    
    wins = {i:0 for i in range(1, 7)} ### HUMAN - SFT - DPO - ORPO - PPO - TIE
    for i, entry in tqdm(test_set.iterrows()):
        topic = entry.topic
        stance = 'SUPPORTING' if entry.label == 1 else 'COUNTER'
        y_human = entry.argument
        prompt = f"<s> [INST] ### Prompt:  Generate a {stance} argument for the topic: {topic} [/INST]\n### Argument: "
        
        with torch.no_grad():
            y_sft = generate(prompt, sft, sft_tokenizer, **GENERATION_KWARGS)
            y_dpo = generate(prompt, dpo, sft_tokenizer, **GENERATION_KWARGS)
            y_orpo = generate(prompt, orpo, orpo_tokenizer, **GENERATION_KWARGS)
            y_ppo = generate(prompt, ppo, ppo_tokenizer, **GENERATION_KWARGS)
        
        try:
            response = compare_models(stance, topic, [y_human, y_sft, y_dpo, y_orpo, y_ppo])
            wins[response] += 1
        except:
            save_to(wins, name='wins.json', output_dir=f'results/{args.model_name}/')
            continue
        
        save_to(wins, name='wins.json', output_dir=f'results/{args.model_name}/')

if __name__ == "__main__":
    main()