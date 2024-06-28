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
from env import OPENAI_API_KEY
from src.utils import save_to, process_gpt_output, get_gpt_response
from EVAL.utils import evaluate

warnings.filterwarnings("ignore")
openai.api_key = OPENAI_API_KEY


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alignement-method', default=None, help="method to align the prompts and completions (e.g. 'sft', 'dpo', 'cpo', 'kto', FIPO)")
    parser.add_argument('--model-name', required=True, help="name of the model to evaluate (e.g. llama, mistral, etc...)")
    return parser.parse_args()

def main(): 
    args = parse_args()
    test_set = pd.read_json('data/argumentation/test_cckg.json')
    evaluate(test_set, type_=args.alignment_method, model_name=args.model_name)
    
if __name__ == "__main__":
    main()

