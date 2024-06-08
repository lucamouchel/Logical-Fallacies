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
sys.path.append('src/')
from DPO.utils import save_to
from EVAL.utils import generate
from peft import AutoPeftModelForCausalLM
GENERATION_KWARGS = {'max_new_tokens': 30, 'no_repeat_ngram_size': 2, 'do_sample': True}

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
    debate_data = debate_data.sample(n=200, random_state=42)
   
   
    model = AutoPeftModelForCausalLM.from_pretrained(model_path, device_map='auto')    
   
   
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    
    generated = []
    
    for i, entry in tqdm(debate_data.iterrows(), total=len(debate_data)):
        topic = entry[3]
        stance = 'SUPPORTING' if entry[6] == 'support' else 'COUNTER'
        prompt = f"<s> [INST] ### Prompt: Generate a {stance} argument for the topic: {topic} [INST]\n### Argument:"
        y = generate(prompt, model, tokenizer, **GENERATION_KWARGS)
        print(y)
        generated.append({'topic': topic, 'stance': stance, 'generated': y.split('### Argument:')[-1].strip()})
    
    save_to(generated, name=f'generated_debates_{args.type}.json', output_dir=f'src/TRANSFER/')
if __name__ == "__main__":
    main()
