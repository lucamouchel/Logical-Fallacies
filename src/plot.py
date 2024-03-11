import pandas as pd  
import seaborn as sns
from data_loader import *
import matplotlib.pyplot as plt


df = pd.read_csv('data/LOGIC/edu_all.csv')
print(df.shape)
df = df[df['updated_label'] != 'miscellaneous']
fig, ax = plt.subplots(1,1, figsize=(15, 12))
sns.barplot(df['updated_label'].value_counts(), ax=ax, orient='h', alpha=0.8, label='Original Fallacy Distribution')

train, dev, test = load_generated_data('cckg')
df2 = pd.concat([train, dev, test])

sns.barplot(df2['fallacy type'].value_counts(), ax=ax, orient='h', alpha=0.5, label='Generated with ChatGPT')


plt.ylabel('Fallacy Type')
plt.legend()
plt.savefig('fallacy_type.png')
