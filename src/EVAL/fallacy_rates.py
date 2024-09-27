import pandas as pd
import openai
import os
import argparse
import sys
sys.path.append(os.path.abspath('src/'))
from env import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alignment-method', default=None, help="method to align the prompts and completions (e.g. 'sft', 'dpo', 'cpo', 'kto', FIPO)")
    parser.add_argument('--model-name', required=True, help="name of the model to evaluate (e.g. llama, mistral, etc...)")
    return parser.parse_args()

def main(): 
    args = parse_args()
    test_set = pd.read_json('data/argumentation/test_cckg.json')
    from eval_utils import evaluate

    evaluate(test_set, type_=args.alignment_method, model_name=args.model_name)
    
if __name__ == "__main__":
    main()

