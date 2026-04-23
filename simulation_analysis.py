import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson, chisquare
import numpy as np
import os

file_path = "luria_delbruck_simulation.csv"

if not os.path.exists(file_path):
    st.error(f"File not found: {file_path}")
    st.stop()


# load data
df = pd.read_csv("luria_delbruck_simulation.csv")

st.title("Luria–Delbrück Simulation")

'''
Submitted by Maayan Yoles and Elizabeth Vaisbourd\n
Q: Write a short explanation of the Luria–Delbrück experiment and how it provided evidence for random mutation (Darwinian evolution) rather than mutation as a direct response to environmental challenges (Lamarckian evolution). \n
A: The Luria-Delbruck experiment proved that mutations occur at random, rather than when needed. In a theoretical case that mutations occur as response to environmental challenges, there would be an even chance for all bacteria of the final generation to show the desired phenotype - survival when plated with lethal phages. Such even chance would result on survival distribution with a narrow probability centered around an individual's probability to gain the correct mutation, which is very low. This distribution would fit a Poisson distribution because each of these mutation events is independent. The distributions received in the experiments showed a long tail, indicating that mutations arised not only in the last generation that was plated, but occasionally in earlier generations as well. 
'''


# user choices
mu_val = st.selectbox(
    "Choose mu / p_induced",
    sorted(df["mu/p_induced"].unique())
)

gen_val = st.selectbox(
    "Choose number of generations",
    sorted(df["generations"].unique())
)

# filter data
df_filtered = df[
    (df["mu/p_induced"] == mu_val) &
    (df["generations"] == gen_val)
]

darwin = df_filtered[df_filtered["model"] == "Darwinian"]["mutants"]
lamarck = df_filtered[df_filtered["model"] == "Lamarckian"]["mutants"]
combined = df_filtered[df_filtered["model"] == "Combined"]["mutants"]


# compute stats
mean_d, var_d = darwin.mean(), darwin.var()
mean_l, var_l = lamarck.mean(), lamarck.var()
mean_c, var_c = combined.mean(), combined.var()

def poisson_fit_score(data):
    data = np.asarray(data, dtype=int)
    lam = np.mean(data)
    n = len(data)

    x = np.arange(0, np.max(data) + 1)
    observed = np.bincount(data, minlength=len(x))
    expected = poisson.pmf(x, lam) * n

    # keep only bins with meaningful expected counts
    mask = expected >= 1

    if mask.sum() < 2:
        p = np.nan
    else:
        observed_test = observed[mask]
        expected_test = expected[mask]
        expected_test = expected_test * (observed_test.sum() / expected_test.sum())
        chi2, p = chisquare(observed_test, expected_test)

    return lam, p, x, expected


# Poisson parameters
lam_d = np.mean(darwin)
lam_l = np.mean(lamarck)
lam_c = np.mean(combined)

# x values for Poisson curves
x_d = np.arange(0, np.max(darwin) + 1)
x_l = np.arange(0, np.max(lamarck) + 1)
x_c = np.arange(0, np.max(combined) + 1)

# integer-centered bins
bins_d = np.arange(-0.5, darwin.max() + 1.5, 1)
bins_l = np.arange(-0.5, lamarck.max() + 1.5, 1)
bins_c = np.arange(-0.5, combined.max() + 1.5, 1)

# FIG1

fig, axes = plt.subplots(1, 3, figsize=(13, 5))

# Darwinian
axes[0].hist(darwin, bins=bins_d, density=True, alpha=0.7, color='blue', label='Darwinian')
axes[0].plot(x_d, poisson.pmf(x_d, lam_d), color='black', linewidth=2, label='Poisson fit')
axes[0].set_title(f"Darwinian\nmean={darwin.mean():.2f}, var={darwin.var():.2f}")
axes[0].set_xlabel("Number of mutants")
axes[0].set_ylabel("Density")
axes[0].legend()

# Lamarckian
axes[1].hist(lamarck, bins=bins_l, density=True, alpha=0.7, color='red', label='Lamarckian')
axes[1].plot(x_l, poisson.pmf(x_l, lam_l), color='black', linewidth=2, label='Poisson fit')
axes[1].set_title(f"Lamarckian\nmean={lamarck.mean():.2f}, var={lamarck.var():.2f}")
axes[1].set_xlabel("Number of mutants")
axes[1].set_ylabel("Density")
axes[1].legend()

# Combined
axes[2].hist(combined, bins=bins_c, density=True, alpha=0.7, color='purple', label='Combined')
axes[2].plot(x_c, poisson.pmf(x_c, lam_c), color='black', linewidth=2, label='Poisson fit')
axes[2].set_title(f"Combined\nmean={combined.mean():.2f}, var={combined.var():.2f}")
axes[2].set_xlabel("Number of mutants")
axes[2].set_ylabel("Density")
axes[2].legend()

plt.tight_layout()
st.pyplot(fig)

'''
Fig 1. Distrubutions of mutants in each regime.
'''


# --- capped distributions relative to Lamarckian max ---
lamarck_max = int(lamarck.max())

# keep Lamarckian unchanged
lamarck_vals = lamarck

# cap only Darwinian and Combined
darwin_capped = darwin.copy()
darwin_capped[darwin_capped > lamarck_max] = lamarck_max + 1

combined_capped = combined.copy()
combined_capped[combined_capped > lamarck_max] = lamarck_max + 1

fig_capped, axes_capped = plt.subplots(1, 3, figsize=(13, 5))

# Darwinian
axes_capped[0].hist(darwin_capped, bins=np.arange(-0.5, lamarck_max + 2.5, 1), alpha=0.7, color='blue')
axes_capped[0].set_yscale('log')
axes_capped[0].set_title("Darwinian")
axes_capped[0].set_xlabel("Number of mutants")
axes_capped[0].set_ylabel("Frequency")

# Lamarckian - unchanged
axes_capped[1].hist(lamarck_vals, bins=np.arange(-0.5, lamarck_max + 1.5, 1), alpha=0.7, color='red')
axes_capped[1].set_title("Lamarckian")
axes_capped[1].set_xlabel("Number of mutants")
axes_capped[1].set_ylabel("Frequency")

# Combined
axes_capped[2].hist(combined_capped, bins=np.arange(-0.5, lamarck_max + 2.5, 1), alpha=0.7, color='purple')
axes_capped[2].set_yscale('log')
axes_capped[2].set_title("Combined")
axes_capped[2].set_xlabel("Number of mutants")
axes_capped[2].set_ylabel("Frequency")

# relabel last bin as "max+"
ticks = list(range(0, lamarck_max + 2))
labels = [str(i) for i in range(0, lamarck_max + 1)] + [f"{lamarck_max}+"]
for ax in [axes_capped[0], axes_capped[2]]:
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=90)

axes_capped[1].set_xticks(range(0, lamarck_max + 1))

plt.tight_layout()
st.pyplot(fig_capped)

'''
Fig 2. Distrubutions of mutants in each regime, x-axis highlights the tail of mutants in Darwinian and Combined regimes relative to Lamarckian model.
'''


# FIG3

# prepare summary (mean mutants per condition per model)
heatmap_data = (
    df.groupby(["model", "mu/p_induced", "generations"])["mutants"]
    .mean()
    .reset_index()
)

# plot 3 heatmaps (one per model)
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))

models = ["Darwinian", "Lamarckian", "Combined"]

for ax, model in zip(axes2, models):
    pivot = heatmap_data[heatmap_data["model"] == model] \
        .pivot(index="generations", columns="mu/p_induced", values="mutants")
    #pivot = pivot.sort_index(ascending=True)
    
    sns.heatmap(pivot, ax=ax, annot=True, fmt=".1f")
    ax.set_title(model)
    ax.set_xlabel("mu / p_induced")
    ax.set_ylabel("generations")

plt.tight_layout()
st.pyplot(fig2)
'''
Fig 3. Mean mutant number for array of mutation probabilities and generation times.
'''
