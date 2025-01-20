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

sns.lineplot(data=filtered_df,x='n',y=f'AUC',hue='method',style='method', palette=palette)#, markers=markers, dashes=dashes)#,style='Regressor',markers=markers, dashes=dashes)

plt.xscale('log')
#plt.yscale('log')  

plt.legend(bbox_to_anchor=(-1, 0.5), loc='center left', borderaxespad=0., fontsize=17)

plt.subplots_adjust(right=0.75)


plt.ylabel(f'AUC',fontsize=17 )
plt.xlabel(r'Number of samples',fontsize=17 )
plt.savefig(f"p_values/visualization/AUC_n_p{p}_cor{cor}_{method}.pdf", bbox_inches="tight")




plt.figure()
sns.set(rc={'figure.figsize':(4,4)})
methods_to_plot = ['Sobol-CPI(1)', 'Sobol-CPI(10)', 'LOCO', 'LOCO-W', 'Sobol-CPI(100)'] 
filtered_df = df[df['method'].isin(methods_to_plot)]
sns.lineplot(data=filtered_df,x='n',y=f'null_imp',hue='method',style='method',palette=palette)#, markers=markers, dashes=dashes)#,style='Regressor',markers=markers, dashes=dashes)

plt.xscale('log')
plt.ylim(-0.1, 0.5)
#plt.title(f'p = {p} $\\rho$ = {cor}', fontsize=15)
#plt.legend(bbox_to_anchor=(-1.20, 0.5), loc='center left', borderaxespad=0., fontsize=15)
plt.legend().set_visible(False)
plt.subplots_adjust(right=0.75)


plt.ylabel(f'Bias null covariates',fontsize=17 )
plt.xlabel(r'Number of samples',fontsize=17 )
plt.savefig(f"p_values/visualization/null_imp_n_p{p}_cor{cor}_{method}.pdf", bbox_inches="tight")





if method=='lin':
    plt.figure()
    sns.set(rc={'figure.figsize':(4,4)})
    methods_to_plot = ['Sobol-CPI(1)', 'Sobol-CPI(10)', 'Sobol-CPI(100)', 'LOCO', 'LOCO-W'] 
    filtered_df = df[df['method'].isin(methods_to_plot)]
    sns.lineplot(data=filtered_df,x='n',y=f'non_null',hue='method',style='method',palette=palette)#, markers=markers, dashes=dashes)#,style='Regressor',markers=markers, dashes=dashes)

    plt.xscale('log')
    plt.ylim(0, 5)

    #plt.legend(bbox_to_anchor=(-1.20, 0.5), loc='center left', borderaxespad=0., fontsize=15)
    plt.legend().set_visible(False)
    plt.subplots_adjust(right=0.75)


    plt.ylabel(f'Bias non-null covariates',fontsize=17 )
    plt.xlabel(r'Number of samples',fontsize=17 )
    plt.savefig(f"p_values/visualization/non_null_n_p{p}_cor{cor}_{method}.pdf", bbox_inches="tight")





plt.figure()
sns.set(rc={'figure.figsize':(4,4)})
sns.lineplot(data=df,x='n',y=f'tr_time',hue='method',style='method',palette=palette, markers=markers, dashes=dashes)#,style='Regressor',markers=markers, dashes=dashes)

plt.xscale('log')
plt.yscale('log')

#plt.ylim(0, 5)

plt.legend(bbox_to_anchor=(-1.50, 0.5), loc='center left', borderaxespad=0., fontsize=17)
#plt.legend().set_visible(False)
plt.subplots_adjust(right=0.75)


plt.ylabel(f'Time',fontsize=17 )
plt.xlabel(r'Number of samples',fontsize=17 )
plt.savefig(f"p_values/visualization/time_n_p{p}_cor{cor}_{method}.pdf", bbox_inches="tight")


plt.figure()
methods_to_plot = ['Sobol-CPI(1)_bt', 'Sobol-CPI(10)_bt', 'Sobol-CPI(100)_bt', 'LOCO_bt', 'LOCO-W'] 
filtered_df = df[df['method'].isin(methods_to_plot)]
sns.set(rc={'figure.figsize':(4,4)})
sns.lineplot(data=filtered_df,x='n',y=f'power',hue='method',style='method',palette=palette)#, markers=markers, dashes=dashes)#,style='Regressor',markers=markers, dashes=dashes)

plt.xscale('log')
#plt.ylim(0, 5)

#plt.legend(bbox_to_anchor=(-1.20, 0.5), loc='center left', borderaxespad=0., fontsize=15)
plt.legend().set_visible(False)
plt.subplots_adjust(right=0.75)


plt.ylabel(f'Power',fontsize=17 )
plt.xlabel(r'Number of samples',fontsize=17 )
plt.savefig(f"p_values/visualization/power_n_p{p}_cor{cor}_{method}_main.pdf", bbox_inches="tight")




plt.figure()
sns.set(rc={'figure.figsize':(4,4)})
sns.lineplot(data=df,x='n',y=f'power',hue='method',style='method',palette=palette, markers=markers, dashes=dashes)#,style='Regressor',markers=markers, dashes=dashes)

plt.xscale('log')
#plt.ylim(0, 5)

#plt.legend(bbox_to_anchor=(-1.20, 0.5), loc='center left', borderaxespad=0., fontsize=15)
plt.legend().set_visible(False)
plt.subplots_adjust(right=0.75)


plt.ylabel(f'Power',fontsize=17 )
plt.xlabel(r'Number of samples',fontsize=17 )
plt.savefig(f"p_values/visualization/power_n_p{p}_cor{cor}_{method}.pdf", bbox_inches="tight")

plt.figure()
sns.set(rc={'figure.figsize':(4,4)})
methods_to_plot = ['Sobol-CPI(10)', 'Sobol-CPI(10)_sqrt', 'Sobol-CPI(10)_n', 'Sobol-CPI(10)_bt'] 
filtered_df = df[df['method'].isin(methods_to_plot)]
sns.lineplot(data=filtered_df,x='n',y=f'power',hue='method',style='method',palette=palette, markers=markers, dashes=dashes)#,style='Regressor',markers=markers, dashes=dashes)

plt.xscale('log')
#plt.ylim(0, 5)

plt.legend(bbox_to_anchor=(-1.5, 0.5), loc='center left', borderaxespad=0., fontsize=17)
#plt.legend().set_visible(False)
plt.subplots_adjust(right=0.75)


plt.ylabel(f'Power',fontsize=17 )
plt.xlabel(r'Number of samples',fontsize=17 )
plt.savefig(f"p_values/visualization/power_n_p{p}_cor{cor}_{method}_R-CPI.pdf", bbox_inches="tight")

plt.figure()
sns.set(rc={'figure.figsize':(4,4)})
methods_to_plot = ['Sobol-CPI(1)', 'Sobol-CPI(1)_sqrt', 'Sobol-CPI(1)_n', 'Sobol-CPI(1)_bt'] 
filtered_df = df[df['method'].isin(methods_to_plot)]
sns.lineplot(data=filtered_df,x='n',y=f'power',hue='method',style='method',palette=palette, markers=markers, dashes=dashes)#,style='Regressor',markers=markers, dashes=dashes)

plt.xscale('log')
#plt.ylim(0, 5)

plt.legend(bbox_to_anchor=(-1.50, 0.5), loc='center left', borderaxespad=0., fontsize=17)
#plt.legend().set_visible(False)
plt.subplots_adjust(right=0.75)


plt.ylabel(f'Power',fontsize=17 )
plt.xlabel(r'Number of samples',fontsize=17 )
plt.savefig(f"p_values/visualization/power_n_p{p}_cor{cor}_{method}_CPI.pdf", bbox_inches="tight")

plt.figure()
sns.set(rc={'figure.figsize':(4,4)})
methods_to_plot = ['Sobol-CPI(100)', 'Sobol-CPI(100)_sqrt', 'Sobol-CPI(100)_n', 'Sobol-CPI(100)_bt'] 
filtered_df = df[df['method'].isin(methods_to_plot)]
sns.lineplot(data=filtered_df,x='n',y=f'power',hue='method',style='method',palette=palette, markers=markers, dashes=dashes)#,style='Regressor',markers=markers, dashes=dashes)

plt.xscale('log')
#plt.ylim(0, 5)

plt.legend(bbox_to_anchor=(-1.50, 0.5), loc='center left', borderaxespad=0., fontsize=17)
#plt.legend().set_visible(False)
plt.subplots_adjust(right=0.75)


plt.ylabel(f'Power',fontsize=17 )
plt.xlabel(r'Number of samples',fontsize=17 )
plt.savefig(f"p_values/visualization/power_n_p{p}_cor{cor}_{method}_R-CPI2.pdf", bbox_inches="tight")

plt.figure()
sns.set(rc={'figure.figsize':(4,4)})
methods_to_plot = ['LOCO', 'LOCO_sqrt', 'LOCO_n', 'LOCO_bt'] 
filtered_df = df[df['method'].isin(methods_to_plot)]
sns.lineplot(data=filtered_df,x='n',y=f'power',hue='method',style='method',palette=palette, markers=markers, dashes=dashes)#,style='Regressor',markers=markers, dashes=dashes)

plt.xscale('log')
#plt.ylim(0, 5)

plt.legend(bbox_to_anchor=(-1.20, 0.5), loc='center left', borderaxespad=0., fontsize=17)
#plt.legend().set_visible(False)
plt.subplots_adjust(right=0.75)


plt.ylabel(f'Power',fontsize=17 )
plt.xlabel(r'Number of samples',fontsize=17 )
plt.savefig(f"p_values/visualization/power_n_p{p}_cor{cor}_{method}-LOCO.pdf", bbox_inches="tight")




plt.figure()
sns.set(rc={'figure.figsize':(4,4)})
sns.lineplot(data=df,x='n',y=f'type_I',hue='method',style='method',palette=palette, markers=markers, dashes=dashes)#,style='Regressor',markers=markers, dashes=dashes)

plt.xscale('log')
#plt.ylim(0, 5)

#plt.legend(bbox_to_anchor=(-1.20, 0.5), loc='center left', borderaxespad=0., fontsize=15)
plt.legend().set_visible(False)
plt.subplots_adjust(right=0.75)


plt.ylabel(f'Type-I error',fontsize=17 )
plt.xlabel(r'Number of samples',fontsize=17 )
plt.savefig(f"p_values/visualization/type_n_p{p}_cor{cor}_{method}.pdf", bbox_inches="tight")













