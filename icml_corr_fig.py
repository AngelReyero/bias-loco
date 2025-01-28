from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import argparse


y_method = 'poly'

snr = 4
p = 50
n = 1000
sparsity = 0.1
intra_cor=[0,0.05, 0.15, 0.3, 0.5, 0.65, 0.85]
cor_meth='toep'
beta= np.array([2, 1])
super_learner=False

if super_learner:
    df = pd.read_csv(f"results_csv/correlation_{y_method}_p{p}_n{n}_super.csv",)
else: 
    df = pd.read_csv(f"results_csv/correlation_{y_method}_p{p}_n{n}.csv",)


# Display the first few rows of the DataFrame
print(df.head())

df = df[df['method'] != 'PFI']

# Change method '0.5CPI' to 'S-CPI'
df['method'] = df['method'].replace('0.5*CPI', 'Sobol-CPI(1)')
df['method'] = df['method'].replace('Robust-CPI', 'Sobol-CPI(100)')

palette = {'Sobol-CPI(100)': 'purple', 'Sobol-CPI(1)': 'blue', 'LOCO-W':'green', 'PFI':'orange', "LOCO-HD": "red"}

auc_scores = []
null_imp = []

# Iterate through each row of the DataFrame
for index, row in df.iterrows():
    # Extract the predictions for the current experiment (as a list)
    y_pred = row.filter(like="imp_V").values
    y=row.filter(like="tr_V").values
    y = np.array(y).astype(int) 
    auc = roc_auc_score(y, y_pred)
    auc_scores.append(auc)
    null_imp.append(np.mean(abs(y_pred[y==0])))

# Add the AUC scores as a new column to the DataFrame
df['AUC'] = auc_scores
df['null_imp'] = null_imp

fig, ax = plt.subplots(1,2, figsize=(16, 4))


# Plot for imp_V0 (top-left subplot)
sns.set(rc={'figure.figsize':(6,3)})
sns.lineplot(data=df, x='intra_cor', y='AUC', hue='method', palette=palette, ax=ax[ 0])  # Top-left subplot


# Format top-left subplot
ax[0].tick_params(axis='x', labelsize=15)  # Adjust x-axis tick label font size
ax[ 0].tick_params(axis='y', labelsize=15) 
ax[ 0].set_xlabel(r'')
ax[ 0].set_ylabel(f'AUC', fontsize=20)
ax[ 0].legend().remove()

sns.lineplot(data=df, x='intra_cor', y='null_imp', hue='method', palette=palette, ax=ax[1])  # Top-right subplot

ax[ 1].tick_params(axis='x', labelsize=15)  # Adjust x-axis tick label font size
ax[ 1].tick_params(axis='y', labelsize=15) 
ax[ 1].set_xlabel(r'')
ax[ 1].set_ylabel(f'Bias null covariates', fontsize=20)
ax[ 1].legend().remove()

# Adjust subplot layout for better spacing
plt.subplots_adjust(hspace=0.4, wspace=0.3)
fig.text(0.5, -0.04, 'Correlation', ha='center', fontsize=20)

#handles, labels = ax[0].get_legend_handles_labels()
#fig.legend(handles, labels, loc='center left', bbox_to_anchor=(-0.25, 0.5), fontsize=20)


if super_learner:
    plt.savefig(f"final_figures/corr_{y_method}_n{n}_p{p}_super.pdf", bbox_inches="tight")
else:
    plt.savefig(f"final_figures/corr_{y_method}_n{n}_p{p}.pdf", bbox_inches="tight")



plt.show()