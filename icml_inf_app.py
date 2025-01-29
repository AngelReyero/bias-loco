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

markers = {
    'Sobol-CPI(10)':  "o",
    'Sobol-CPI(10)_sqrt': "^",
    'Sobol-CPI(10)_n': "D",
    'Sobol-CPI(1)':  "o",
    'Sobol-CPI(1)_sqrt': "^",
    'Sobol-CPI(1)_n': "D",
    'LOCO-W':  "o",
    'PFI': "D",
    'LOCO':  "o",
    'LOCO_n': "D",
    'LOCO_sqrt': "^",
    'Sobol-CPI(100)':  "o",
    'Sobol-CPI(100)_sqrt': "^",
    'Sobol-CPI(100)_n': "D",
    'Sobol-CPI(10)_bt': '*',
    'Sobol-CPI(1)_bt': '*',
    'LOCO_bt': '*',
    'Sobol-CPI(100)_bt': '*',
}

dashes = {
    'Sobol-CPI(10)':  (3, 5, 1, 5),
    'Sobol-CPI(10)_sqrt': (5, 5),
    'Sobol-CPI(10)_n': (1, 1),
    'Sobol-CPI(1)':  (3, 5, 1, 5),
    'Sobol-CPI(1)_sqrt': (5, 5),
    'Sobol-CPI(1)_n': (1, 1),
    'LOCO-W':  (3, 5, 1, 5),
    'PFI': (1, 1),
    'LOCO':  (3, 5, 1, 5),
    'LOCO_n': (1, 1),
    'LOCO_sqrt': (5, 5),
    'Sobol-CPI(100)':  (3, 5, 1, 5),
    'Sobol-CPI(100)_sqrt': (5, 5),
    'Sobol-CPI(100)_n': (1, 1),
    'Sobol-CPI(10)_bt': (3, 1, 3),
    'Sobol-CPI(1)_bt': (3, 1, 3),
    'LOCO_bt': (3, 1, 3),
    'Sobol-CPI(100)_bt': (3, 1, 3),
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



sns.set_style("white")

fig, ax = plt.subplots(1, 3, figsize=(18, 5), gridspec_kw={'hspace': 0.1, 'wspace': 0.3})




sns.lineplot(data=df, x='n', y='AUC', hue='method', palette=palette, ax=ax[0])  # Bottom-left subplot


# Format bottom-left subplot
ax[0].set_xscale('log')
ax[ 0].tick_params(axis='x', labelsize=15)  # Adjust x-axis tick label font size
ax[ 0].tick_params(axis='y', labelsize=15) 
ax[0].set_xlabel(r'')
ax[ 0].set_ylabel(f'AUC', fontsize=20)
ax[0].legend().remove()

# Plot for imp_V8 (bottom-right subplot)
sns.lineplot(data=df, x='n', y='non_null', hue='method', palette=palette, ax=ax[1])  # Bottom-right subplot


# Format bottom-right subplot
ax[ 1].set_xscale('log')
ax[1].tick_params(axis='x', labelsize=15)  # Adjust x-axis tick label font size
ax[ 1].tick_params(axis='y', labelsize=15) 
ax[ 1].set_xlabel(r'')
ax[1].set_ylabel(f'Bias non-null covariates', fontsize=20)
ax[ 1].legend().remove()

# Plot for imp_V8 (bottom-right subplot)
sns.lineplot(data=df, x='n', y='null_imp', hue='method', palette=palette, ax=ax[2])  # Bottom-right subplot


# Format bottom-right subplot
ax[2].set_xscale('log')
ax[ 2].tick_params(axis='x', labelsize=15)  # Adjust x-axis tick label font size
ax[ 2].tick_params(axis='y', labelsize=15) 
ax[ 2].set_xlabel(r'')
ax[ 2].set_ylabel(f'Bias null covariates', fontsize=20)
ax[ 2].legend().remove()


# Adjust subplot layout for better spacing
plt.subplots_adjust(hspace=0.4, wspace=0.3)
fig.text(0.5, -0.05, 'Number of samples', ha='center', fontsize=20)


plt.savefig(f"final_figures/appendix/inf_{method}_p{p}_cor{cor}_auc.pdf", bbox_inches="tight")

# Display the plot






fig, ax = plt.subplots(1, 3, figsize=(18, 5), gridspec_kw={'hspace': 0.1, 'wspace': 0.3})




sns.lineplot(data=df, x='n', y='tr_time', hue='method', palette=palette,markers=markers, dashes=dashes, style='method', ax=ax[0])  # Bottom-left subplot


# Format bottom-left subplot
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[ 0].tick_params(axis='x', labelsize=15)  # Adjust x-axis tick label font size
ax[ 0].tick_params(axis='y', labelsize=15) 
ax[0].set_xlabel(r'')
ax[ 0].set_ylabel(f'Time', fontsize=20)
ax[0].legend().remove()

# Plot for imp_V8 (bottom-right subplot)
sns.lineplot(data=df, x='n', y='power', hue='method', palette=palette, markers=markers, dashes=dashes, style='method', ax=ax[1])  # Bottom-right subplot


# Format bottom-right subplot
ax[ 1].set_xscale('log')
ax[1].tick_params(axis='x', labelsize=15)  # Adjust x-axis tick label font size
ax[ 1].tick_params(axis='y', labelsize=15) 
ax[ 1].set_xlabel(r'')
ax[1].set_ylabel(f'Power', fontsize=20)
ax[ 1].legend().remove()

# Plot for imp_V8 (bottom-right subplot)
sns.lineplot(data=df, x='n', y='type_I', hue='method', palette=palette, markers=markers, dashes=dashes, style='method', ax=ax[2])  # Bottom-right subplot


# Format bottom-right subplot
ax[2].set_xscale('log')
ax[ 2].tick_params(axis='x', labelsize=15)  # Adjust x-axis tick label font size
ax[ 2].tick_params(axis='y', labelsize=15) 
ax[ 2].set_xlabel(r'')
ax[ 2].set_ylabel(f'Type-I error', fontsize=20)
ax[ 2].legend().remove()


# Adjust subplot layout for better spacing
plt.subplots_adjust(hspace=0.4, wspace=0.3)
fig.text(0.5, -0.05, 'Number of samples', ha='center', fontsize=20)


plt.savefig(f"final_figures/appendix/inf_{method}_p{p}_cor{cor}_pow.pdf", bbox_inches="tight")








p =50
sparsity = 0.2
method = "poly"

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

markers = {
    'Sobol-CPI(10)':  "o",
    'Sobol-CPI(10)_sqrt': "^",
    'Sobol-CPI(10)_n': "D",
    'Sobol-CPI(1)':  "o",
    'Sobol-CPI(1)_sqrt': "^",
    'Sobol-CPI(1)_n': "D",
    'LOCO-W':  "o",
    'PFI': "D",
    'LOCO':  "o",
    'LOCO_n': "D",
    'LOCO_sqrt': "^",
    'Sobol-CPI(100)':  "o",
    'Sobol-CPI(100)_sqrt': "^",
    'Sobol-CPI(100)_n': "D",
    'Sobol-CPI(10)_bt': '*',
    'Sobol-CPI(1)_bt': '*',
    'LOCO_bt': '*',
    'Sobol-CPI(100)_bt': '*',
}

dashes = {
    'Sobol-CPI(10)':  (3, 5, 1, 5),
    'Sobol-CPI(10)_sqrt': (5, 5),
    'Sobol-CPI(10)_n': (1, 1),
    'Sobol-CPI(1)':  (3, 5, 1, 5),
    'Sobol-CPI(1)_sqrt': (5, 5),
    'Sobol-CPI(1)_n': (1, 1),
    'LOCO-W':  (3, 5, 1, 5),
    'PFI': (1, 1),
    'LOCO':  (3, 5, 1, 5),
    'LOCO_n': (1, 1),
    'LOCO_sqrt': (5, 5),
    'Sobol-CPI(100)':  (3, 5, 1, 5),
    'Sobol-CPI(100)_sqrt': (5, 5),
    'Sobol-CPI(100)_n': (1, 1),
    'Sobol-CPI(10)_bt': (3, 1, 3),
    'Sobol-CPI(1)_bt': (3, 1, 3),
    'LOCO_bt': (3, 1, 3),
    'Sobol-CPI(100)_bt': (3, 1, 3),
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



sns.set_style("white")

fig, ax = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'hspace': 0.1, 'wspace': 0.3})




sns.lineplot(data=df, x='n', y='AUC', hue='method', palette=palette, ax=ax[0])  # Bottom-left subplot


# Format bottom-left subplot
ax[0].set_xscale('log')
ax[ 0].tick_params(axis='x', labelsize=15)  # Adjust x-axis tick label font size
ax[ 0].tick_params(axis='y', labelsize=15) 
ax[0].set_xlabel(r'')
ax[ 0].set_ylabel(f'AUC', fontsize=20)
ax[0].legend().remove()



sns.lineplot(data=df, x='n', y='null_imp', hue='method', palette=palette, ax=ax[1])  # Bottom-right subplot


# Format bottom-right subplot
ax[1].set_xscale('log')
ax[ 1].tick_params(axis='x', labelsize=15)  # Adjust x-axis tick label font size
ax[ 1].tick_params(axis='y', labelsize=15) 
ax[ 1].set_xlabel(r'')
ax[ 1].set_ylabel(f'Bias null covariates', fontsize=20)
ax[ 1].legend().remove()


# Adjust subplot layout for better spacing
plt.subplots_adjust(hspace=0.4, wspace=0.3)
fig.text(0.5, -0.05, 'Number of samples', ha='center', fontsize=20)


plt.savefig(f"final_figures/appendix/inf_{method}_p{p}_cor{cor}_auc.pdf", bbox_inches="tight")

# Display the plot






fig, ax = plt.subplots(1, 3, figsize=(18, 5), gridspec_kw={'hspace': 0.1, 'wspace': 0.3})




sns.lineplot(data=df, x='n', y='tr_time', hue='method', palette=palette, markers=markers, dashes=dashes, style='method', ax=ax[0])  # Bottom-left subplot


# Format bottom-left subplot
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[ 0].tick_params(axis='x', labelsize=15)  # Adjust x-axis tick label font size
ax[ 0].tick_params(axis='y', labelsize=15) 
ax[0].set_xlabel(r'')
ax[ 0].set_ylabel(f'Time', fontsize=20)
ax[0].legend().remove()

# Plot for imp_V8 (bottom-right subplot)
sns.lineplot(data=df, x='n', y='power', hue='method', palette=palette, markers=markers, dashes=dashes, style='method', ax=ax[1])  # Bottom-right subplot


# Format bottom-right subplot
ax[ 1].set_xscale('log')
ax[1].tick_params(axis='x', labelsize=15)  # Adjust x-axis tick label font size
ax[ 1].tick_params(axis='y', labelsize=15) 
ax[ 1].set_xlabel(r'')
ax[1].set_ylabel(f'Power', fontsize=20)
ax[ 1].legend().remove()

# Plot for imp_V8 (bottom-right subplot)
sns.lineplot(data=df, x='n', y='type_I', hue='method', palette=palette, markers=markers, dashes=dashes, style='method', ax=ax[2])  # Bottom-right subplot


# Format bottom-right subplot
ax[2].set_xscale('log')
ax[ 2].tick_params(axis='x', labelsize=15)  # Adjust x-axis tick label font size
ax[ 2].tick_params(axis='y', labelsize=15) 
ax[ 2].set_xlabel(r'')
ax[ 2].set_ylabel(f'Type-I error', fontsize=20)
ax[ 2].legend().remove()


# Adjust subplot layout for better spacing
plt.subplots_adjust(hspace=0.4, wspace=0.3)
fig.text(0.5, -0.05, 'Number of samples', ha='center', fontsize=20)


plt.savefig(f"final_figures/appendix/inf_{method}_p{p}_cor{cor}_pow.pdf", bbox_inches="tight")








