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
from EVAL.utils import generate, get_gpt_feedback, evaluate, evaluate_from_file

warnings.filterwarnings("ignore")
openai.api_key = OPENAI_API_KEY

GENERATE_KWARGS = {'max_new_tokens': 35, 'no_repeat_ngram_size': 2, 'do_sample': True}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sft-path', default=None)
    parser.add_argument('--dpo-path', default=None)
    parser.add_argument('--num-iterations-made',default=0, type=int, help='Number of iterations made for the selft instruction')
    parser.add_argument('--eval-type', default='win-rate', help='Type of evaluation to perform. Options: win-rate, fallacy-count')
    parser.add_argument('--model-name', default='llama')
    parser.add_argument('--eval-from-file', action='store_true')
    parser.add_argument('--use-rag', action='store_true')

    return parser.parse_args()

def main(): 
    args = parse_args()
    ref_model_dir = args.sft_path
    dpo_model_dir = args.dpo_path
    test_set = pd.read_json('data/argumentation/test_cckg.json')
 
    if args.eval_from_file:
        evaluate_from_file(test_set, type_='dpo', model_name=args.model_name)

    else: 
        if ref_model_dir:
            tokenizer = transformers.AutoTokenizer.from_pretrained(ref_model_dir)
            sft_model = AutoPeftModelForCausalLM.from_pretrained(ref_model_dir, device_map='auto')
            evaluate(test_set, model=sft_model, tokenizer=tokenizer, type_=f"sft{'_rag' if args.use_rag else ''}", eval_from_file=False, model_name=args.model_name, use_rag=args.use_rag, **GENERATE_KWARGS)
        
        if dpo_model_dir:
            dpo_model = AutoPeftModelForCausalLM.from_pretrained(dpo_model_dir, device_map='auto')
            tokenizer = transformers.AutoTokenizer.from_pretrained(dpo_model_dir)
            evaluate(test_set, model=dpo_model, tokenizer=tokenizer, type_=f"dpo{'_rag' if args.use_rag else ''}", eval_from_file=False, model_name=args.model_name, use_rag=args.use_rag, **GENERATE_KWARGS)
                    

if __name__ == "__main__":
    main()

