import json
from math import e 
import data_loader
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 
from itertools import cycle
train, dev,test = data_loader.load_logic_data()


df = pd.read_csv('data/LOGIC/edu_all.csv')
df = df[~(df.updated_label == 'miscellaneous')]
print(df.updated_label.value_counts())

custom_palette = sns.color_palette("tab20c")

# Plotting the pie chart
data = dict(df.updated_label.value_counts())
values = list(data.values())
labels = list(data.keys())
pd.read_json('data/dpo/arguments/test_cckg.json')
# Generate a custom color palette with distinct colors for each label
default_palette = plt.cm.Set3.colors
extra_color = 'lightgray'
color_palette = cycle(default_palette )

# Plotting the pie chart with distinct colors
fig, ax = plt.subplots(figsize=(26, 20))
ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=140, colors=['lightgray' if i == len(values) -1 else next(color_palette) for i in range(len(values))], textprops={'fontsize': 32})
#ax.set_title('Distribution of Fallacies', fontsize=28)
plt.rcParams['font.size'] = 32  # Set font size for the entire plot

# Adding the legend on top of the title
#ax.legend(labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=len(labels)//2)

# Adjust the layout to prevent overlapping
plt.tight_layout()

# Showing the plot
plt.show()



plt.savefig('data/LOGIC/fig.png')