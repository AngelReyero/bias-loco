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




p =50
sparsity = 0.25
method = "poly"

seed= 0




cor=0.6
cor_meth='toep'
beta= np.array([2, 1])
snr=2
alpha = 0.05

df = pd.read_csv(f"p_values/results_csv/{method}_n_p{p}_cor{cor}_sqd_crt.csv")


# Display the first few rows of the DataFrame
print(df.head())

palette = {
    'Sobol-CPI(10)': 'purple',
    'Sobol-CPI(10)_sqrt': 'purple',
    'Sobol-CPI(10)_n': 'purple',
    'Sobol-CPI(10)_n2': 'purple',
    'Sobol-CPI(1)': 'blue',
    'Sobol-CPI(1)_sqrt': 'blue',
    'Sobol-CPI(1)_n': 'blue',
    'Sobol-CPI(1)_n2': 'blue',
    'LOCO-W': 'green',
    'LOCO': 'red',
    'LOCO_n': 'red',
    'LOCO_sqrt': 'red',
    'LOCO_n2': 'red',
    'Sobol-CPI(100)': 'cyan',
    'Sobol-CPI(100)_sqrt': 'cyan',
    'Sobol-CPI(100)_n': 'cyan',
    'Sobol-CPI(100)_n2': 'cyan',
    'Sobol-CPI(10)_bt': 'purple',
    'Sobol-CPI(1)_bt': 'blue',
    'LOCO_bt': 'red',
    'Sobol-CPI(100)_bt': 'cyan',
    'HRT': 'brown'

}

markers = {
    'Sobol-CPI(10)':  "o",
    'Sobol-CPI(10)_sqrt': "^",
    'Sobol-CPI(10)_n': "D",
    'Sobol-CPI(10)_bt': '*',
    'Sobol-CPI(10)_n2': 's',
    
    'Sobol-CPI(1)':  "o",
    'Sobol-CPI(1)_sqrt': "^",
    'Sobol-CPI(1)_n': "D",
    'Sobol-CPI(1)_bt': '*',
    'Sobol-CPI(1)_n2': 's',
    
    'Sobol-CPI(100)':  "o",
    'Sobol-CPI(100)_sqrt': "^",
    'Sobol-CPI(100)_n': "D",
    'Sobol-CPI(100)_bt': '*',
    'Sobol-CPI(100)_n2': 's',
    
    'LOCO-W':  "o",
    'LOCO':  "o",
    'LOCO_n': "D",
    'LOCO_sqrt': "^",
    'LOCO_bt': '*',
    'LOCO_n2': 's',

    'HRT':'o',


}


dashes = {
    'Sobol-CPI(10)':  (3, 5, 1, 5),
    'Sobol-CPI(10)_sqrt': (5, 5),
    'Sobol-CPI(10)_n': (1, 1),
    'Sobol-CPI(10)_bt': (3, 1, 3),
    'Sobol-CPI(10)_n2': (2, 4),
    
    'Sobol-CPI(1)':  (3, 5, 1, 5),
    'Sobol-CPI(1)_sqrt': (5, 5),
    'Sobol-CPI(1)_n': (1, 1),
    'Sobol-CPI(1)_bt': (3, 1, 3),
    'Sobol-CPI(1)_n2': (2, 4),
    
    'Sobol-CPI(100)':  (3, 5, 1, 5),
    'Sobol-CPI(100)_sqrt': (5, 5),
    'Sobol-CPI(100)_n': (1, 1),
    'Sobol-CPI(100)_bt': (3, 1, 3),
    'Sobol-CPI(100)_n2': (2, 4),
    
    'LOCO-W':  (3, 5, 1, 5),
    'LOCO':  (3, 5, 1, 5),
    'LOCO_n': (1, 1),
    'LOCO_sqrt': (5, 5),
    'LOCO_bt': (3, 1, 3),
    'LOCO_n2': (2, 4),
    'HRT':(1,1)

}
default_dash = (1, 1)

# Ensure that all methods have a valid dash pattern
dashes2 = {method: dashes.get(method, default_dash) for method in df['method'].unique()}


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
df['method'] = df['method'].replace('CPI_sqd', 'Sobol-CPI(1)_n2')
df['method'] = df['method'].replace('R-CPI_sqd', 'Sobol-CPI(10)_n2')
df['method'] = df['method'].replace('R-CPI2_sqd', 'Sobol-CPI(100)_n2')
df['method'] = df['method'].replace('LOCO_sqd', 'LOCO_n2')
df['method'] = df['method'].replace('CRT', 'HRT')






sns.set_style("white")
fig, ax = plt.subplots(2, 2, figsize=(12, 10), gridspec_kw={'hspace': 0.15, 'wspace': 0.2})

methods_to_plot = ['Sobol-CPI(1)', 'Sobol-CPI(1)_sqrt', 'Sobol-CPI(1)_n', 'Sobol-CPI(1)_bt', 'Sobol-CPI(1)_n2', 'LOCO-W', 'HRT'] 
filtered_df = df[df['method'].isin(methods_to_plot)]
sns.lineplot(data=filtered_df, x='n', y='type_I', hue='method', palette=palette, markers=markers, dashes=dashes, style='method',ax=ax[0, 0])  # Top-left subplot

# Format top-left subplot
ax[0, 0].set_xscale('log')
ax[0, 0].tick_params(axis='x', labelsize=15)  # Adjust x-axis tick label font size
ax[0, 0].tick_params(axis='y', labelsize=15) 
ax[0, 0].set_xlabel(r'')
ax[0, 0].set_ylabel(f'', fontsize=20)
#ax[0, 0].legend().remove()

methods_to_plot = ['Sobol-CPI(10)', 'Sobol-CPI(10)_sqrt', 'Sobol-CPI(10)_n', 'Sobol-CPI(10)_bt', 'Sobol-CPI(10)_n2', 'LOCO-W', 'HRT'] 
filtered_df = df[df['method'].isin(methods_to_plot)]
sns.lineplot(data=filtered_df, x='n', y='type_I', hue='method', palette=palette, ax=ax[0, 1], markers=markers,dashes=dashes, style='method')  # Top-left subplot

# Format top-left subplot
ax[0, 1].set_xscale('log')
ax[0, 1].tick_params(axis='x', labelsize=15)  # Adjust x-axis tick label font size
ax[0, 1].tick_params(axis='y', labelsize=15) 
ax[0, 1].set_xlabel(r'')
ax[0, 1].set_ylabel(f'', fontsize=20)
#ax[0, 0].legend().remove()


methods_to_plot = ['Sobol-CPI(100)', 'Sobol-CPI(100)_sqrt', 'Sobol-CPI(100)_n', 'Sobol-CPI(100)_bt', 'Sobol-CPI(100)_n2','LOCO-W', 'HRT'] 
filtered_df = df[df['method'].isin(methods_to_plot)]
sns.lineplot(data=filtered_df, x='n', y='type_I', hue='method', palette=palette, ax=ax[1, 0], markers=markers, dashes=dashes, style='method')  # Top-left subplot

# Format top-left subplot
ax[1, 0].set_xscale('log')
ax[1, 0].tick_params(axis='x', labelsize=15)  # Adjust x-axis tick label font size
ax[1, 0].tick_params(axis='y', labelsize=15) 
ax[1, 0].set_xlabel(r'')
ax[1, 0].set_ylabel(f'', fontsize=20)
#ax[0, 0].legend().remove()

methods_to_plot = ['LOCO', 'LOCO_sqrt', 'LOCO_n', 'LOCO_bt', 'LOCO_n2', 'LOCO-W', 'HRT'] 
filtered_df = df[df['method'].isin(methods_to_plot)]
sns.lineplot(data=filtered_df, x='n', y='type_I', hue='method', palette=palette, ax=ax[1, 1], markers=markers, dashes=dashes, style='method')  # Top-left subplot

# Format top-left subplot
ax[1, 1].set_xscale('log')
ax[1, 1].tick_params(axis='x', labelsize=15)  # Adjust x-axis tick label font size
ax[1, 1].tick_params(axis='y', labelsize=15) 
ax[1, 1].set_xlabel(r'')
ax[1, 1].set_ylabel(f'', fontsize=20)
#ax[0, 0].legend().remove()

fig.text(0.5, 0.05, 'Number of samples', ha='center', fontsize=20)

fig.text(0.05, 0.45, 'Type_I', ha='center', fontsize=20, rotation=90)


plt.savefig(f"final_figures/appendix/type_I_{method}_p{p}_cor{cor}.pdf", bbox_inches="tight")

