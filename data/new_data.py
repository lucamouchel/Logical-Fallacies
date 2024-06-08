import json 
import pandas as pd 
from tqdm import tqdm
import pathlib
ORDER = {'Not a Fallacy': 0,  'faulty generalization': 1, 'false causality': 2, 'fallacy of relevance': 3, 'fallacy of extension': 4, 'equivocation': 5, 'ad populum': 6, 'appeal to emotion': 7, 'ad hominem': 8, 'circular reasoning': 9, 'fallacy of credibility': 10, 'fallacy of logic': 11, 'false dilemma': 12, 'intentional': 13}

with open('data/dpo/arguments/test.json', 'r') as f:
    dpo_data = json.load(f)

with open('data/generated/test_cckg.json', 'r') as f:
    generated_data = json.load(f)



dpo_data = pd.read_json('data/dpo/arguments/test.json')



print(ORDER)

new_data = []
for i, entry in tqdm(dpo_data.iterrows()):
    rejected = entry.rejected
    found = 0
    for generated in generated_data:

        if generated['argument'].lower() == rejected.lower() or generated['argument'].lower() == rejected.lower()[:-1]:
            found = 1
            f_type = generated['fallacy type']
            break
        
            
    if found == 0:
        print(rejected)
    try:
        new_data .append( {'prompt': entry.prompt, 'chosen': entry.chosen, 'rejected': rejected, 'fallacy_type': ORDER[f_type]})
    except:
        print("342")

import os

pathlib.Path('data/dpo/arguments').mkdir(parents=True, exist_ok=True)
output_path = os.path.join('data/dpo/arguments', 'dpo_with_fallacies_test.json')
with open(output_path, 'w') as json_file:
    json.dump(new_data, json_file, indent=4, sort_keys=False)

