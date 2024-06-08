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
import pathlib 
import argparse
from peft import AutoPeftModelForCausalLM
import sys
sys.path.append('src/')
from DPO.env import OPENAI_API_KEY

from utils import save_to, process_gpt_output, get_gpt_response
from EVAL.utils import generate, get_gpt_feedback, evaluate

warnings.filterwarnings("ignore")
openai.api_key = OPENAI_API_KEY

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file-path', default="FWEg")
    parser.add_argument('--type', default="AAA")
    return parser.parse_args()

dataset = pd.read_json('data/argumentation/test_cckg.json')[:200]
dataset.reset_index(drop=True, inplace=True)
def evaluate(dataset):
    args = parse_args()
    f_rate = 0
    f_rates = {}
    data = []

    with open('results/mistral_bis/mistral_again.json', 'r') as f:
            arguments = json.load(f) 
            # arguments= [arg['argument'] for arg in arguments]

    for i, entry in tqdm(dataset.iterrows(), total=len(dataset)):
        topic = entry.topic
        stance = 'supporting' if entry.label==1 else 'counter'
        y = arguments[i]
       
        feedback = get_gpt_feedback(topic, y, stance=stance, type_=args.type)
        if feedback['fallacy_type']!='None' :
            f_rate+=1
        if feedback['fallacy_type'] in f_rates.keys():
            f_rates[feedback['fallacy_type']] += 1
        else:
            f_rates[feedback['fallacy_type']] = 1

        if i%10==0:
            print("Fallacy rate so far: ", f_rate) 
            print(f_rates)
        data.append(feedback)

    save_to(data, name=f'new_f-rate.json', output_dir=f'results/mistral_bis/{args.type}/')
    print(f_rates)
    print(f"f rate for {args.type}", f_rate)
    print("FALLACY TYPES")
    
    save_to(f_rates, name=f'fallacy_counts.json', output_dir=f'results/llama_bis_51/{args.type}/')
    for k,v in f_rates.items():
        print(k.upper(), ':', v)

if __name__ == "__main__":
    evaluate(dataset=dataset)

