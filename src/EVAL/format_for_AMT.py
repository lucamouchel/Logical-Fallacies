import pandas as pd 
import numpy as np
import json 
test_set = pd.read_json('data/argumentation/test_cckg.json')

def open_args(path):
    with open(path) as f:
        args = json.load(f)
    return args


sft_args = open_args('results/llama/sft_args.json')
dpo_args = open_args('results/llama/dpo_args.json')
orpo_args = open_args('results/llama/orpo_args.json')
ppo_args = open_args('results/llama/ppo_args.json')
human_args = pd.read_json('data/argumentation/test_cckg.json')['argument']
  
  
for i in range(4):
    samples = []
    for j, entry in test_set.iterrows():
        topic = entry.topic
        stance = 'supporting' if entry.label == 1 else 'counter'
        y_sft = sft_args[j]
        if i == 0:
            y = human_args[j]
            out = 'human'
        elif i == 1:
            y = dpo_args[j]
            out='dpo'
        elif i == 2:
            y = orpo_args[j]
            out='orpo'
        elif i == 3:
            y = ppo_args[j]
            out='ppo'
            
        arguments = [None, None]
        
        shuffle_idx = np.random.choice([0,1])
        arguments[shuffle_idx] = y_sft 
        arguments[1-shuffle_idx] = y
        sample = {'topic': topic, 'stance': stance, 'argument1': arguments[0], 'argument2': arguments[1], 'sft_index' : shuffle_idx}
        samples.append(sample)
        pd.DataFrame(samples).to_csv('src/EVAL/AMT/samples_sft_vs_{}.csv'.format(out), index=False)
        
    
    