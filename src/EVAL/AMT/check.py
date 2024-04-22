import pandas as pd 


humans = pd.read_csv('/root/Logical-Fallacies/src/EVAL/AMT/annotations/SFT_vs_PPO.csv')

humans = humans[['Input.topic', 'Input.sft_index', 'Answer.q1_valid']]

humans = humans.drop_duplicates(subset=['Input.topic'])
print(humans)
import json
import sys 
sys.path.append('/root/Logical-Fallacies/src/')
from EVAL.utils import get_gpt_feedback, generate





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

from DPO.env import OPENAI_API_KEY
from DPO.utils import save_to, get_gpt_response

warnings.filterwarnings("ignore")
openai.api_key = OPENAI_API_KEY

######### compare SFT responses with DPO, ORPO, PPO responses #########

def compare_models(stance, topic, arguments):
    possible_answers = ''
    for i, argument in enumerate(arguments):
        possible_answers += f"{i+1}. {argument}\n"
    prompt = f"""Which of these {stance} arguments is better for the topic: {topic} Please consider the fact that some of these arguments might be in the form of logical fallacies. If one of the arguments is a logical fallacy, it should not be the best. 
    \nArguments: \n{possible_answers}\nIf the arguments are equally good, return number {len(arguments) + 1}.\nThe better argument is number:"""
    
    response = get_gpt_response(prompt, necessary_tokens=1, model='gpt-4')
    try:
        response = int(response)
        return response
    except:
        print("Invalid response. Please try again.")
        print(arguments)
        print()
        print(response)
        return len(arguments) + 1 ### TIE
   
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', required=True)
    args = parser.parse_args()
    
    test_set = pd.read_json('data/argumentation/test_cckg.json')
    
    with open(f'results/{args.model_name}/sft_args.json', 'r') as f:
        sft_args = json.load(f)
        
    combinations = [['sft', 'ppo']]
    for combination in combinations:
        to_compare = combination[1]
        with open(f'results/{args.model_name}/{combination[1]}_args.json', 'r') as f:
            arguments = json.load(f)

        wins = {'sft': 0, to_compare: 0, 'tie': 0}
        for i, entry in tqdm(test_set[:50].iterrows()):
            human_entry = humans.iloc[i]
            if human_entry['Answer.q1_valid'] == 10 or human_entry['Answer.q1_valid'] == -10:
                continue
            
            human_entry['Answer.q1_valid'] -= 1 

            preferred_response = 'sft' if human_entry['Answer.q1_valid'] == human_entry['Input.sft_index'] else to_compare
        
            if i % 201 == 0 and i != 0:
                print("going to sleep to not overcook gpt4")
                time.sleep(60) 
            topic = entry.topic
            stance = 'SUPPORTING' if entry.label == 1 else 'COUNTER'
            y_sft = sft_args[i]
            
          
            y = arguments[i]
            
            response = compare_models(stance, topic, [y_sft, y])
            assert type(response) == int
            if response == 1:
                wins['sft'] += 1
                if preferred_response == to_compare:
                    print(topic)
                    print(stance)
                    print(f"HUMANS believe {to_compare} is better than SFT. GPT4 believes SFT is better.")
                    print('SFT Argument:', y_sft)
                    print(f'{to_compare} Argument:', y)
                    print('\n')
                

            elif response == 2:
                wins[to_compare] += 1
                if preferred_response == 'sft':
                    print(topic)
                    print(stance)
                    print(f"HUMANS believe SFT is better than {to_compare}. GPT4 believes {to_compare} is better.")
                    print('SFT Argument:', y_sft)
                    print(f'{to_compare} Argument:', y)
                    print('\n')
            elif response == 3:
                wins['tie'] += 1

if __name__ == "__main__":
    main()