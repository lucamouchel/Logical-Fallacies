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
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--type', required=True, help='sft, dpo, cpo, kto,..')
    args = parser.parse_args()
    
    model = AutoPeftModelForCausalLM.from_pretrained(args.model_path, device_map='auto')
    
    ### SFT and DPO have the same tokenizer -- but not sure for the other two
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_path)
    test_set = pd.read_json('data/argumentation/test_cckg.json')
    
    arguments = []
    for i, entry in tqdm(test_set.iterrows(), total=len(test_set)):
        topic = entry.topic
        stance = 'SUPPORTING' if entry.label == 1 else 'COUNTER'
        prompt = f"<s> [INST] ### Prompt:  Generate a {stance} argument for the topic: {topic} [/INST]\n### Argument: "
       
        with torch.no_grad():
            y = generate(prompt, model, tokenizer, **GENERATION_KWARGS)
           
        
        arguments.append(y.split('### Argument: ')[-1].strip())
       
        
        if i % 100 == 0:
            print("ARGUMENT:\n", y, '\n')    
    save_to(arguments, name=f'{args.type}_args.json', output_dir=f'results/{args.model_name}/')

        
if __name__ == "__main__":
    main()