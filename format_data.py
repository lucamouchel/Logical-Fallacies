import pandas as pd
import json

with open('data/dpo/arguments/test.json', 'r') as f:
    train = json.load(f)
    

data = []
seen = set()
for i, item in enumerate(train['prompt']):
    prompt = item
    if 'supporting argument' in prompt: 
        prompt = prompt.replace('supporting argument', 'SUPPORTING argument')
    else: 
        prompt = prompt.replace('counter argument', 'COUNTER argument')
        
    chosen_label = train['chosen'][i]
    
    if chosen_label in seen:
     
        continue
    else:
        seen.add(chosen_label)
        
        d = {'prompt': prompt, 'argument': chosen_label}
        data.append(d)

with open('data/sft_refiner/test.json', 'w') as f:
    json.dump(data, f, indent=4, sort_keys=False)