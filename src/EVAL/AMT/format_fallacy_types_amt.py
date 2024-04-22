import pandas as pd 

import numpy as np
import json
def open_args(path):
    with open(path) as f:
        args = json.load(f)
    return args

sft_data = open_args('results/llama/arguments/sft/f-rate.json')
dpo_data = open_args('results/llama/arguments/dpo/f-rate.json')
orpo_data = open_args('results/llama/arguments/orpo/f-rate.json')
ppo_data = open_args('results/llama/arguments/ppo/f-rate.json')

test_set = pd.read_json('data/argumentation/test_cckg.json')

sft, dpo, orpo, ppo = [], [], [], []
for i, entry in test_set.iterrows():
    topic = entry.topic
    stance = 'supporting' if entry.label == 1 else 'counter'
    y_sft = sft_data[i]
    y_dpo = dpo_data[i]
    y_orpo = orpo_data[i]
    y_ppo = ppo_data[i]
    assert topic == y_sft['topic'] == y_dpo['topic'] == y_orpo['topic'] == y_ppo['topic']
    
    sft_sample = {'topic': topic, 'stance': stance, 'argument': y_sft['argument'], 'prediction': y_sft['fallacy_type']} 
    dpo_sample = {'topic': topic, 'stance': stance, 'argument': y_dpo['argument'], 'prediction': y_dpo['fallacy_type']}
    orpo_sample = {'topic': topic, 'stance': stance, 'argument': y_orpo['argument'], 'prediction': y_orpo['fallacy_type']}
    ppo_sample = {'topic': topic, 'stance': stance, 'argument': y_ppo['argument'], 'prediction': y_ppo['fallacy_type']}
    
    sft.append(sft_sample)
    dpo.append(dpo_sample)
    orpo.append(orpo_sample)
    ppo.append(ppo_sample)
    
pd.DataFrame(sft).to_csv('src/EVAL/AMT/fallacy_agreement_sft.csv', index=False, columns=['topic', 'stance', 'argument', 'fallacy_type'])
pd.DataFrame(dpo).to_csv('src/EVAL/AMT/fallacy_agreement_dpo.csv', index=False , columns=['topic', 'stance', 'argument', 'fallacy_type'])
pd.DataFrame(orpo).to_csv('src/EVAL/AMT/fallacy_agreement_orpo.csv', index=False, columns=['topic', 'stance', 'argument', 'fallacy_type'])
pd.DataFrame(ppo).to_csv('src/EVAL/AMT/fallacy_agreement_ppo.csv', index=False, columns=['topic', 'stance', 'argument', 'fallacy_type'])
    
    
