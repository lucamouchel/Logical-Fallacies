import torch
import transformers
import pandas as pd
import numpy as np
import os
from transformers import AutoModelForCausalLM
from tqdm import tqdm
import warnings
import json
import openai
import time
import os
import transformers
import pathlib
import argparse
import sys 
from datetime import datetime
sys.path.append('src/')
from src.utils import save_to
from EVAL.utils import get_gpt_feedback, remove_incomplete_last_sentence
from EVAL.utils import generate
from peft import AutoPeftModelForCausalLM
GENERATION_KWARGS = {'max_new_tokens': 30, 'no_repeat_ngram_size': 2, 'do_sample': True, 'top_p': 0.9, 'top_k':10}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--type', required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    model_path = args.model_path
    model_name = 'llama' if 'llama' in model_path.lower() else 'mistral'

    debate_data = pd.read_csv('data/test_debate.txt', header=None, sep='\t')
    debate_data = debate_data.sample(n=100, random_state=42).reset_index(drop=True)
   
   
    model = AutoPeftModelForCausalLM.from_pretrained(model_path, device_map='auto')    
   
   
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    
    generated = []
    data = []
    f_rates = {}
    f_rate = 0
    arguments = []
    date = datetime.now()
    for i, entry in tqdm(debate_data.iterrows(), total=len(debate_data)):
        topic = entry[3]
        stance = 'SUPPORTING' if entry[6] == 'support' else 'COUNTER'
        prompt = f"<s> [INST] ### Prompt: Generate a {stance} argument for the topic: {topic} [INST]\n### Argument:"
        y = generate(prompt, model, tokenizer, **GENERATION_KWARGS)
        y = y.split('### Argument:')[-1].strip()
        y_old = y
        try:
            y = remove_incomplete_last_sentence(y)
        except:
            y = y_old
        arguments.append(y) 
        feedback = get_gpt_feedback(topic, y, stance=stance, type_=args.type)
        if feedback['fallacy_type']!='None' :
            f_rate+=1
        if feedback['fallacy_type'] in f_rates.keys():
            f_rates[feedback['fallacy_type']] += 1
        else:
            f_rates[feedback['fallacy_type']] = 1

        if i % 10 == 0:
            print(f_rates)
            save_to(arguments, name=f'out_of_domain_{args.type}.json', output_dir=f'results/out_of_domain/{args.type}/')

        data.append(feedback)


    save_to(data, name=f'f-rate_{args.type}.json', output_dir=f'results/out_of_domain/{args.type}/')
    print(f_rates)
    print(f"f rate for {args.type}:", f_rate)
    print("FALLACY TYPES")
    
    save_to(f_rates, name=f'fallacy_counts_{args.type}.json', output_dir=f'results/out_of_domain/{args.type}/')
    for k,v in f_rates.items():
        print(k.upper(), ':', v)
        
    save_to(arguments, name=f'out_of_domain_{args.type}.json', output_dir=f'results/out_of_domain/{args.type}/')

if __name__ == "__main__":
    main()
