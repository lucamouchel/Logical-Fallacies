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

sys.path.append('src/')  
from DPO.env import OPENAI_API_KEY
from DPO.utils import save_to, get_gpt_response

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
        print(arguments)
        print()
        print(response)
        return len(arguments) + 1 ### TIE
   
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', default='llama')
    args = parser.parse_args() 
     
    debate_data = pd.read_csv('data/test_debate.txt', header=None, sep='\t')
    debate_data = debate_data.sample(n=100, random_state=42).reset_index(drop=True)
   
    #test_set = pd.read_json('data/argumentation/test_cckg.json')

    with open(f'results/out_of_domain/sft/f-rate_sft.json', 'r') as f:
        sft_args = json.load(f)
        sft_args = [arg['argument'] for arg in sft_args]


    combinations = [['sft', 'custom_cpo']]
    for combination in combinations:
        to_compare = combination[1]
        
        with open(f'results/out_of_domain/{to_compare}/f-rate_{to_compare}.json', 'r') as f:
            arguments = json.load(f)
            arguments = [arg['argument'] for arg in arguments]

       

        wins = {'sft': 0, to_compare: 0, 'tie': 0}
        winner = []

        for i, entry in tqdm(debate_data.iterrows()):

            topic = entry[3]
            stance = 'supporting' if entry[6] == 'support' else 'counter'

            y_sft = sft_args[i]
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
                save_to(wins, name='wins.json', output_dir=f'results/out_of_domain/{to_compare}/')
                exit()
        from collections import Counter
        print(Counter(winner))
        print(wins)
        save_to(wins, name='wins.json', output_dir=f'results/out_of_domain/{to_compare}/')

        # save_to(wins, name=f'sft_vs_{to_compare}2.json', output_dir=f'results/{args.model_name}/')

if __name__ == "__main__":
    main()