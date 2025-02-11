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
import time

sys.path.append(os.path.abspath('src/'))
from env import OPENAI_API_KEY
from utils import save_to, get_gpt_response
from eval_utils import generate

warnings.filterwarnings("ignore")
openai.api_key = OPENAI_API_KEY

######### compare SFT responses with DPO, ORPO, PPO responses #########

def compare_models(stance, topic, arguments):
    possible_answers = ''
    for i, argument in enumerate(arguments):
        possible_answers += f"{i+1}. {argument}\n"
    prompt = f"""Which of these {stance} arguments is better for the topic: {topic} Please consider the fact that some of these arguments might be in the form of logical fallacies. Please evaluate the arguments and whether they are logical fallacies. If one of the arguments is a logical fallacy, it should not be the best. 
    \nArguments: \n{possible_answers}\nIf the arguments are equally good, return number {len(arguments) + 1}.\nThe better argument is number:"""

    response = get_gpt_response(prompt, necessary_tokens=1, model='gpt-4')
    try:
        response = int(response)
        return response
    except:
        print("Invalid response. Please try again.")
        return len(arguments) + 1 ### TIE
   
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', required=True, help='llama, mistral, etc...')
    parser.add_argument('--compare-with', required=True, help="Either set only 1 alignement method e.g., 'dpo', or multiple methods separated by a semi-colon e.g., dpo;kto;fipo...")
    args = parser.parse_args() 
    
    test_set = pd.read_json('data/argumentation/test_cckg.json')

    with open(f'results/{args.model_name}/sft_args.json', 'r') as f:
        sft_args = json.load(f)

    comparisons = args.compare_with.split(';')
    combinations = [['sft', comparison.strip()] for comparison in comparisons]
    for combination in combinations:
        to_compare = combination[1]
        with open(f'results/{args.model_name}/{to_compare}/f-rate.json', 'r') as f:
            arguments = json.load(f)
            arguments = [arg['argument'] for arg in arguments]

        wins = {'sft': 0, to_compare: 0, 'tie': 0}
        winner = []

        for i, entry in tqdm(test_set.iterrows()):
    
            if i % 201 == 0 and i != 0:
                print("going to sleep to not overcook gpt4")
                time.sleep(60) 

            topic = entry.topic
            stance = 'SUPPORTING' if entry.label == 1 else 'COUNTER'
            y_sft = sft_args[i][1]
            y = arguments[i]

            try:
                response = compare_models(stance, topic, [y_sft, y])
                assert type(response) == int
                if response == 1:
                    wins['sft'] += 1
                    winner.append('sft')
                elif response == 2:
                    wins[to_compare] += 1
                    winner.append(to_compare)
                elif response == 3:
                    wins['tie'] += 1
                    winner.append('tie')

                if i%5 == 0:
                    print(wins)
                if i == len(arguments)- 1:
                    break
            except:
                save_to(wins, name='wins.json', output_dir=f'results/{args.model_name}/')
                exit()
        from collections import Counter
        print(Counter(winner))
        save_to(winner, name="winners.json", output_dir=f'results/{args.model_name}/{to_compare}/')
        print(wins)

if __name__ == "__main__":
    main()