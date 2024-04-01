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
from DPO.utils import save_to, process_gpt_output, get_gpt_response, generate, evaluate
warnings.filterwarnings("ignore")
openai.api_key = OPENAI_API_KEY

GENERATION_KWARGS = {'max_new_tokens': 30, 'no_repeat_ngram_size': 2, 'do_sample': True}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ppo-path', default=None)
    parser.add_argument('--model-name', default='llama', required=True)
    return parser.parse_args()

def main(): 
    args = parse_args()
    ppo_path = args.ppo_path
    test_set = pd.read_json('data/argumentation/test_cckg.json')
    model = AutoPeftModelForCausalLM.from_pretrained(ppo_path, device_map='cuda:2')
    tokenizer = transformers.AutoTokenizer.from_pretrained(ppo_path)
    evaluate(test_set, model=model, tokenizer=tokenizer, type_='ppo', model_name=args.model_name, **GENERATION_KWARGS)

if __name__ == "__main__":
    main()

