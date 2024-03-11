from numpy import fix
from torch import fix_
import data_loader as dl 

import pandas as pd 
import json 
import pathlib
import string 

def capitalize_if_not_uppercase(sentence):
    if not sentence.split(' ')[0].istitle():
        return sentence.capitalize()
    return sentence

def fix_string(s):
    if s.endswith(','):
        s = s.rstrip(',') + '.'
    if s[-1] not in string.punctuation:
        s += '.'
    return capitalize_if_not_uppercase(s)

def get_data(data_dir, split='train'):
    with open(f'{data_dir+split}.json', 'r') as f:
        df = json.load(f)
    
    data = {'prompt': [], 'chosen': [], 'rejected': []}
    for item in df:
        prompt = list(item.keys())[0]
        
        chosen, rejected = item[prompt]['responses']
        chosen = fix_string(chosen)
        rejected = fix_string(rejected)

        data['prompt'].append(prompt)
        data['chosen'].append(chosen)
        data['rejected'].append(rejected)

    pathlib.Path('data/generated/').mkdir(parents=True, exist_ok=True)
    output_path = f'data/dpo/arguments/{split}.json'
    with open(output_path, 'w') as json_file:
        json.dump(data, json_file, indent=4, sort_keys=False)



def translate_augmented_data_for_dpo(path, save_as):
    with open(path) as f:
        data = json.load(f)
        
    data = pd.DataFrame(data)
    
    result = {'prompt': [], 'chosen': [], 'rejected': []}

    for i, row in data.iterrows():
        evidence = row['evidence']
        original_claim = row['original claim']
        claim_type = row['claim type']
        fallacy_claim = row['fallacy claim']
        
        prompt = f"Given the following evidence: {evidence}, generate a {claim_type} claim."
        prompt = fix_string(prompt)
        original_claim = fix_string(original_claim)
        fallacy_claim = fix_string(fallacy_claim)
        result['prompt'].append(prompt)
        result['chosen'].append(original_claim)
        result['rejected'].append(fallacy_claim)
        
    pathlib.Path('data/generated/').mkdir(parents=True, exist_ok=True)
    output_path = f'data/dpo/claims/{save_as}.json'
    with open(output_path, 'w') as json_file:
        json.dump(result, json_file, indent=4, sort_keys=False)
        
        
        
#translate_augmented_data_for_dpo('data/generated/claims/train_generated.json', 'train')