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
from EVAL.utils import evaluate, evaluate_out_of_domain
import EVAL.utils
warnings.filterwarnings("ignore")
openai.api_key = OPENAI_API_KEY

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', default='llama')
    parser.add_argument('--type', required=True)
    parser.add_argument('--eval-type', default='debates', required=True)
    return parser.parse_args()

def main(): 
    args = parse_args()
    evaluate_out_of_domain(eval_type=args.eval_type, type_=args.type, model_name=args.model_name)

if __name__ == "__main__":
    main()

