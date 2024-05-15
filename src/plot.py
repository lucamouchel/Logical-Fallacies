import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
# Data
categories = ['SFT', 'DPO', 'ORPO', 'PPO', 'CPO', 'KTO']
llama_values = [68.5, 70, 79, 75, 71.75, 72]
mistral_values = [67.5 , 66 , 69.5 , 72 , 68.75 , 67.5 ]  # Exaggerated Mistral values

# Create dataframe
data = {'Categories': categories * 2,
        'Values': llama_values + mistral_values,
        'Model': ['Llama-2 7B'] * len(categories) + ['Mistral 7B (Exaggerated)'] * len(categories)}
df = pd.DataFrame(data)

# Set style
sns.set(style="whitegrid")

# Radar chart
plt.figure(figsize=(8, 6))
ax = plt.subplot(111, polar=True)

# Plot data
sns.lineplot(data=df, x='Categories', y='Values', hue='Model', ax=ax, linewidth=2, style='Model', markers=True)

# Add legend
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))

# Show plot
plt.title('Not A Fallacy - Llama-2 7B vs. Mistral 7B (Exaggerated)', size=20, y=1.1)
plt.show()


plt.savefig('radar_chart.png')