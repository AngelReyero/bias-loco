from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import argparse

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import toep




p =200
sparsity = 0.2
method = "lin"

seed= 0




cor=0.6
cor_meth='toep'
beta= np.array([2, 1])
snr=2
alpha = 0.05

df = pd.read_csv(f"p_values/results_csv/{method}_n_p{p}_cor{cor}_bt.csv",)


# Display the first few rows of the DataFrame
print(df.head())

palette = {
    'Sobol-CPI(10)': 'purple',
    'Sobol-CPI(10)_sqrt': 'purple',
    'Sobol-CPI(10)_n': 'purple',
    'Sobol-CPI(1)': 'blue',
    'Sobol-CPI(1)_sqrt': 'blue',
    'Sobol-CPI(1)_n': 'blue',
    'LOCO-W': 'green',
    'PFI': 'orange',
    'LOCO': 'red',
    'LOCO_n': 'red',
    'LOCO_sqrt': 'red',
    'Sobol-CPI(100)': 'cyan',
    'Sobol-CPI(100)_sqrt': 'cyan',
    'Sobol-CPI(100)_n': 'cyan',
    'Sobol-CPI(10)_bt': 'purple',
    'Sobol-CPI(1)_bt': 'blue',
    'LOCO_bt': 'red',
    'Sobol-CPI(100)_bt': 'cyan',
}



auc_scores = []
null_imp = []
non_null = []
power = []
type_I = []

if method=='lin':
    mat = toep(p, cor)
    cond_v = np.zeros(p)
    for i in range(p):
        Sigma_without_j = np.delete(mat, i, axis=1)
        Sigma_without_jj = np.delete(Sigma_without_j, i, axis=0)
        cond_v[i] = (
            mat[i, i]
            - Sigma_without_j[i, :]
            @ np.linalg.inv(Sigma_without_jj)
            @ Sigma_without_j[i, :].T
        )
        

# Iterate through each row of the DataFrame
for index, row in df.iterrows():
    # Extract the predictions for the current experiment (as a list)
    y_pred = row.filter(like="imp_V").values
    pval = row.filter(like="pval").values
    selected = pval<=alpha
    y=row.filter(like="tr_V").values
    y = np.array(y).astype(int) 
    auc = roc_auc_score(y, y_pred)
    auc_scores.append(auc)
    null_imp.append(np.mean(abs(y_pred[y==0])))
    if method=='lin':
        non_null.append(np.mean(abs(y_pred[y==1]-cond_v[y==1])))
    power.append(sum(selected[y==1])/sum(y==1))
    type_I.append(sum(selected[y==0])/sum(y==0))



# Add the AUC scores as a new column to the DataFrame
df['AUC'] = auc_scores
df['null_imp'] = null_imp
if method=='lin':
    df['non_null'] = non_null
df['power'] = power
df['type_I'] = type_I

plt.figure()
sns.set(rc={'figure.figsize':(4,4)})


df['method'] = df['method'].replace('CPI', 'Sobol-CPI(1)')
df['method'] = df['method'].replace('R-CPI', 'Sobol-CPI(10)')
df['method'] = df['method'].replace('R-CPI2', 'Sobol-CPI(100)')
df['method'] = df['method'].replace('CPI_n', 'Sobol-CPI(1)_n')
df['method'] = df['method'].replace('R-CPI_n', 'Sobol-CPI(10)_n')
df['method'] = df['method'].replace('R-CPI2_n', 'Sobol-CPI(100)_n')
df['method'] = df['method'].replace('CPI_sqrt', 'Sobol-CPI(1)_sqrt')
df['method'] = df['method'].replace('R-CPI_sqrt', 'Sobol-CPI(10)_sqrt')
df['method'] = df['method'].replace('R-CPI2_sqrt', 'Sobol-CPI(100)_sqrt')
df['method'] = df['method'].replace('CPI_bt', 'Sobol-CPI(1)_bt')
df['method'] = df['method'].replace('R-CPI_bt', 'Sobol-CPI(10)_bt')
df['method'] = df['method'].replace('R-CPI2_bt', 'Sobol-CPI(100)_bt')

methods_to_plot = ['Sobol-CPI(1)', 'Sobol-CPI(10)', 'Sobol-CPI(100)', 'LOCO', 'LOCO-W'] 
filtered_df = df[df['method'].isin(methods_to_plot)]



fig, ax = plt.subplots(2, 2, figsize=(12, 10), gridspec_kw={'hspace': 0.1, 'wspace': 0.3})



# Plot for imp_V0 (top-left subplot)
sns.set(rc={'figure.figsize':(4,4)})
sns.lineplot(data=df, x='n', y='AUC', hue='method', palette=palette, ax=ax[0, 0])  # Top-left subplot

# Format top-left subplot
ax[0, 0].set_xscale('log')
ax[0, 0].tick_params(axis='x', labelsize=15)  # Adjust x-axis tick label font size
ax[0, 0].tick_params(axis='y', labelsize=15) 
ax[0, 0].set_xlabel(r'')
ax[0, 0].set_ylabel(f'AUC', fontsize=20)
ax[0, 0].legend().remove()





methods_to_plot = ['Sobol-CPI(1)', 'Sobol-CPI(10)', 'Sobol-CPI(100)', 'LOCO', 'LOCO-W'] 
filtered_df = df[df['method'].isin(methods_to_plot)]


# Plot for imp_V0 (top-left subplot)
sns.set(rc={'figure.figsize':(4,4)})
sns.lineplot(data=filtered_df, x='n', y='non_null', hue='method', palette=palette, ax=ax[0, 1])  # Top-left subplot

# Format top-left subplot
ax[0, 1].set_xscale('log')
ax[0, 1].tick_params(axis='x', labelsize=15)  # Adjust x-axis tick label font size
ax[0, 1].tick_params(axis='y', labelsize=15) 
ax[0, 1].set_xlabel(r'')
ax[0, 1].set_ylabel(f'Bias non-null covariates', fontsize=20)
ax[0, 1].legend().remove()






methods_to_plot = ['Sobol-CPI(1)_bt', 'Sobol-CPI(10)_bt', 'Sobol-CPI(100)_bt', 'LOCO_bt', 'LOCO-W'] 
filtered_df = df[df['method'].isin(methods_to_plot)]


sns.set(rc={'figure.figsize':(4,4)})
sns.lineplot(data=filtered_df, x='n', y='power', hue='method', palette=palette, ax=ax[1, 0])  # Top-left subplot

# Format top-left subplot
ax[1, 0].set_xscale('log')
ax[1, 0].tick_params(axis='x', labelsize=15)  # Adjust x-axis tick label font size
ax[1, 0].tick_params(axis='y', labelsize=15) 
ax[1, 0].set_xlabel(r'')
ax[1, 0].set_ylabel(f'Power', fontsize=20)
ax[1, 0].legend().remove()

sns.set(rc={'figure.figsize':(4,4)})
sns.lineplot(data=filtered_df, x='n', y='type_I', hue='method', palette=palette, ax=ax[1, 1])  # Top-left subplot

# Format top-left subplot
ax[1, 1].set_xscale('log')
ax[1, 1].tick_params(axis='x', labelsize=15)  # Adjust x-axis tick label font size
ax[1, 1].tick_params(axis='y', labelsize=15) 
ax[1, 1].set_xlabel(r'')
ax[1, 1].set_ylabel(f'Type-I error', fontsize=20)
ax[1, 1].legend().remove()


plt.savefig(f"final_figures/power_n_p{p}_cor{cor}_{method}.pdf", bbox_inches="tight")
plt.show()


