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
from env import OPENAI_API_KEY
import pathlib 
import argparse
from peft import AutoPeftModelForCausalLM
import sys
sys.path.append('src/')
from utils import save_to, process_gpt_output, get_gpt_response
from EVAL.utils import generate, get_gpt_feedback, evaluate

warnings.filterwarnings("ignore")
openai.api_key = OPENAI_API_KEY

GENERATE_KWARGS = {'max_new_tokens': 30, 'no_repeat_ngram_size': 2, 'do_sample': True}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sft-path', default=None)
    parser.add_argument('--dpo-path', default=None)
    parser.add_argument('--num-iterations-made',default=0, type=int, help='Number of iterations made for the selft instruction')
    parser.add_argument('--eval-type', default='win-rate', help='Type of evaluation to perform. Options: win-rate, fallacy-count')
    parser.add_argument('--generate-and-save', default='true', type=str)
    parser.add_argument('--model-name', default='llama')
    parser.add_argument('--task', required=True)
    return parser.parse_args()

def main(): 
    args = parse_args()
    ref_model_dir = args.sft_path
    dpo_model_dir = args.dpo_path
    test_set = pd.read_json('data/argumentation/test_cckg.json')
 
    sft_model = AutoPeftModelForCausalLM.from_pretrained(ref_model_dir, device_map='auto')
    dpo_model = AutoPeftModelForCausalLM.from_pretrained(dpo_model_dir, device_map='auto')
    tokenizer = transformers.AutoTokenizer.from_pretrained(ref_model_dir)
    evaluate(test_set, model=sft_model, tokenizer=tokenizer, type_='sft', model_name=args.model_name, **GENERATE_KWARGS)
    evaluate(test_set, model=dpo_model, tokenizer=tokenizer, type_='dpo', model_name=args.model_name, **GENERATE_KWARGS)
                

if __name__ == "__main__":
    main()

