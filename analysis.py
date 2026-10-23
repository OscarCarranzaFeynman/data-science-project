"""Reproducible synthetic-data study. Run with the project uv environment."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

base = Path(__file__).resolve().parent
(base/'data').mkdir(exist_ok=True)
(base/'figures').mkdir(exist_ok=True)
rng = np.random.default_rng(7204)
scale = 10
population = rng.exponential(scale=scale, size=100000)
sizes = [5, 20, 100]
means_5 = rng.exponential(scale=scale, size=(5000, 5)).mean(axis=1)
means_20 = rng.exponential(scale=scale, size=(5000, 20)).mean(axis=1)
means_100 = rng.exponential(scale=scale, size=(5000, 100)).mean(axis=1)
sample_means = pd.DataFrame({'n5':means_5, 'n20':means_20, 'n100':means_100})
sample_means.to_csv(base/'data/sample_means.csv', index=False)
pd.DataFrame({'value':population}).to_csv(base/'data/synthetic_population.csv', index=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].hist(population, bins=80, density=True, color='#7c3aed', edgecolor='white')
axes[0].axvline(scale, color='#0891b2', label='Theoretical mean = 10')
axes[0].set(xlabel='Individual observation', ylabel='Density', title='Positive skew in individual observations')
axes[0].legend()
axes[1].hist(means_5, bins=45, density=True, alpha=.45, color='#7c3aed', label='n = 5')
axes[1].hist(means_20, bins=45, density=True, alpha=.45, color='#2563eb', label='n = 20')
axes[1].hist(means_100, bins=45, density=True, alpha=.45, color='#0891b2', label='n = 100')
axes[1].axvline(scale, color='#111827', linewidth=1)
axes[1].set(xlabel='Mean of a sample', ylabel='Density', title='Larger samples concentrate the mean')
axes[1].legend()
fig.tight_layout()
fig.savefig(base/'figures/distributions.png', dpi=160)
plt.close(fig)

empirical_se = sample_means.std(ddof=1).to_numpy()
theoretical_se = scale/np.sqrt(sizes)
fig, ax = plt.subplots(figsize=(8, 4.8))
ax.plot(sizes, theoretical_se, 'o-', color='#7c3aed', label='Theoretical σ / √n')
ax.plot(sizes, empirical_se, 's--', color='#0891b2', label='Empirical SD of 5,000 means')
ax.set(xlabel='Sample size', ylabel='Standard error of the mean', title='Simulation agrees with the standard-error formula')
ax.set_xticks(sizes)
ax.grid(alpha=.2)
ax.legend()
fig.tight_layout()
fig.savefig(base/'figures/standard_error.png', dpi=160)
plt.close(fig)

summary = pd.DataFrame({'sample_size':sizes, 'empirical_mean':sample_means.mean().to_numpy(), 'empirical_se':empirical_se, 'theoretical_se':theoretical_se})
summary.to_csv(base/'data/summary.csv', index=False)
(base/'results.json').write_text(json.dumps(summary.to_dict(orient='records'), indent=2))
print(summary.to_string(index=False))
