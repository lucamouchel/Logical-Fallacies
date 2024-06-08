import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Data for Theta win rates
theta_win_rates = [31, 26, 28, 27.75 ,17]
#theta_win_rates = [61, 49, 52.5, 46, 63.5]
plt.rcParams.update({'font.size': 28})  # Adjust font size here

matchups = ['', '', '', '', '']

# Creating a DataFrame
df = pd.DataFrame(data=theta_win_rates, index=matchups, columns=[''])

# Reshape the data for a single row heatmap
df = df.transpose()

# Create the heatmap with a red color gradient
plt.figure(figsize=(20, 2))  # Adjust the figure size to better fit the single row
ax = sns.heatmap(df, annot=True, cmap='Reds', cbar=True, fmt=".1f", vmin=10, vmax=40)

# Move the x-axis labels to the top
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0) 

# Remove all tick marks
ax.tick_params(axis='both', which='both', length=0)  # Set 'length=0' to remove ticks

plt.show()


plt.savefig('f-rates.svg')
plt.show()
