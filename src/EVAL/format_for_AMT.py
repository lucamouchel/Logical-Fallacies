import pandas as pd 
import numpy as np
import json 
test_set = pd.read_json('data/argumentation/test_cckg.json')[149:200]
test_set.reset_index(drop=True, inplace=True)
def open_args(path):
    with open(path) as f:
        args = json.load(f)
    return args


sft_args = open_args('results/llama/sft_args.json')
dpo_args = open_args('results/llama/dpo_args.json')
orpo_args = open_args('results/llama/orpo_args.json')
ppo_args = open_args('results/llama/ppo_args.json')
human_args = pd.read_json('data/argumentation/test_cckg.json')['argument']
kto_args = open_args('results/llama/kto_args.json')
cpo_args = open_args('results/llama/cpo_args.json')
cpo_custom = open_args('results/llama_bis/cpo_results/f-rate.json')
cpo_custom = [arg['argument'] for arg in cpo_custom[149:]]

sft_args=sft_args[149:]


for i in range(6):
    samples = []
    if i < 4:
        continue
    for j, entry in test_set.iterrows():
        topic = entry.topic
        stance = 'supporting' if entry.label == 1 else 'counter'

        y_sft = sft_args[j][1]
        print(topic)
        y_cpo_custom = cpo_custom[j]
        print(y_sft)
        print(y_cpo_custom)
        print()
        arguments = [None, None]
        out='cpo_custom_51_samples'
        shuffle_idx = np.random.choice([0,1])
        arguments[shuffle_idx] = y_sft 
        arguments[1-shuffle_idx] = y_cpo_custom
        sample = {'topic': topic, 'stance': stance, 'argument1': arguments[0], 'argument2': arguments[1], 'sft_index' : shuffle_idx}
        samples.append(sample)
        pd.DataFrame(samples).to_csv('src/EVAL/AMT/samples_sft_vs_{}.csv'.format(out), index=False)
    exit()
    for j, entry in test_set.iterrows():
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
        elif i == 4:
            y = kto_args[j]
            out='kto'
        elif i == 5:
            y = cpo_args[j]
            out='cpo'
        arguments = [None, None]
        
        shuffle_idx = np.random.choice([0,1])
        arguments[shuffle_idx] = y_sft 
        arguments[1-shuffle_idx] = y
        sample = {'topic': topic, 'stance': stance, 'argument1': arguments[0], 'argument2': arguments[1], 'sft_index' : shuffle_idx}
        samples.append(sample)
        pd.DataFrame(samples).to_csv('src/EVAL/AMT/samples_sft_vs_{}.csv'.format(out), index=False)
        
    
    