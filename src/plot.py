import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
import json

# Given data
with open('data/dpo/arguments/probas_1.json', 'r') as f:
    data = json.load(f)

# Extracting values from the dictionary
values = list(filter(lambda x: x > 0.5, data.values()))
print(np.mean(values))
# Plotting the distribution with increased bins
plt.figure(figsize=(10, 6))
plt.hist(values, bins=20, density=True, alpha=0.6, color='g')  # Increased the number of bins to 40

# Fit a normal distribution to the data
mu, std = norm.fit(values)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)

# Kernel Density Estimation (KDE)
kde = KernelDensity(kernel='gaussian', bandwidth=0.05).fit(np.array(values)[:, np.newaxis])
x_d = np.linspace(xmin, xmax, 1000)
logprob = kde.score_samples(x_d[:, np.newaxis])
plt.fill_between(x_d, np.exp(logprob), alpha=0.5, color='b')

# Empirical Distribution Function (EDF)
sorted_values = np.sort(values)
n = len(sorted_values)
y = np.arange(1, n + 1) / n
plt.plot(sorted_values, y, color='r', linestyle='-', linewidth=2)

# Labels and title
plt.title('Distribution of Values with Fitted Normal Distribution, KDE, and EDF')
plt.xlabel('Values')
plt.ylabel('Density / Probability')
plt.legend(['Fitted Normal Distribution', 'KDE', 'EDF', 'Histogram'])

# Show plot
plt.grid(True)
plt.savefig('distribution2.png')
