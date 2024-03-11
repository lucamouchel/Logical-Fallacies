import pandas as pd
import json
import matplotlib.pyplot as plt


df_sft = pd.read_csv('data/sft_probas.csv')
df_dpo = pd.read_csv('data/dpo_probas.csv')
dfs = [df_sft, df_dpo]
for i in range(1,5):
    dfs.append(pd.read_csv(f'data/dpo+sr_probas_iter{i}.csv'))

dfs = [df.sort_values(by='1') for df in dfs]


for df in dfs:
    df['index'] = [i for i in range(len(df))]


fig, ax = plt.subplots(1,1, figsize=(9, 8))

for i, df in enumerate(dfs):
    df = df[250:]
    if i == 0:
        df.plot(kind='line', x='index', y='1', ax=ax, label='SFT')
    elif i == 1:
        df.plot(kind='line', x='index', y='1', ax=ax, label='DPO')
    else:
        df.plot(kind='line', x='index', y='1', ax=ax, label=f'DPO with self-reward (iteration {i-1})')

plt.xlabel('Samples sorted by fallacy probability')
plt.ylabel('Fallacy probability')
plt.tight_layout()
plt.savefig('probas.png')


