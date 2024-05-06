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
from DPO.utils import save_to, process_gpt_output, get_gpt_response
from EVAL.utils import generate, get_gpt_feedback, evaluate

warnings.filterwarnings("ignore")
openai.api_key = OPENAI_API_KEY

GENERATE_KWARGS = {'max_new_tokens': 35, 'no_repeat_ngram_size': 2, 'do_sample': True}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpo-path', default=None)
    parser.add_argument('--model-name', default='llama')
    parser.add_argument('--eval-from-file', action='store_true')

    return parser.parse_args()

def main(): 
    args = parse_args()
    model_dir = args.cpo_path
    test_set = pd.read_json('data/argumentation/test_cckg.json')
 
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir)
    model = AutoPeftModelForCausalLM.from_pretrained(model_dir, device_map='auto')
    evaluate(test_set, model=model, tokenizer=tokenizer, type_="cpo", eval_from_file=args.eval_from_file, model_name=args.model_name, use_rag=False, **GENERATE_KWARGS)
    
if __name__ == "__main__":
    main()

