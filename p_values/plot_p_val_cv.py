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




p = 150
sparsity = 0.2


seed= 0




cor=0.3
cor_meth='toep'
beta= np.array([2, 1])
snr=2
alpha = 0.05

df = pd.read_csv(f"p_values/results_csv/lin_n_p{p}_cor{cor}.csv",)


# Display the first few rows of the DataFrame
print(df.head())

palette = {'r-CPI': 'purple', 'CPI': 'blue', 'LOCO-W':'green', 'PFI':'orange', "LOCO-HD": "red", 'r-CPI2' : "brown"}
palette = {
    'r-CPI': 'purple',
    'CPI': 'blue',
    'LOCO-W': 'green',
    'PFI': 'orange',
    'LOCO-HD': 'red',
    'r-CPI2': 'brown',
    'r-CPI_c': 'purple',
    'CPI_c': 'blue',
    'r-CPI2_c': 'brown'
}

markers = {
    'r-CPI': "o",
    'CPI': "o",
    'LOCO-W': "^",
    'PFI': "D",
    'LOCO-HD': "^",
    'r-CPI2': "o",
    'r-CPI_c': "D",
    'CPI_c': "D",
    'r-CPI2_c': "D"
}
dashes = {
    'r-CPI': (5, 5),              # Dashed line: dash length 5, space length 5
    'CPI': (5, 5),
    'LOCO-W': (5, 5),
    'PFI': (5, 5),
    'LOCO-HD': (5, 5),
    'r-CPI2': (5, 5),
    'r-CPI_c': (3, 5, 1, 5),      # Complex dash-dot pattern
    'CPI_c': (3, 5, 1, 5),
    'r-CPI2_c': (3, 5, 1, 5)
}

auc_scores = []
null_imp = []
non_null = []
power = []
type_I = []

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
    non_null.append(np.mean(abs(y_pred[y==1]-cond_v[y==1])))
    power.append(sum(selected[y==1])/sum(y==1))
    type_I.append(sum(selected[y==0])/sum(y==0))



# Add the AUC scores as a new column to the DataFrame
df['AUC'] = auc_scores
df['null_imp'] = null_imp
df['non_null'] = non_null
df['power'] = power
df['type_I'] = type_I

plt.figure()
sns.set(rc={'figure.figsize':(4,4)})
sns.lineplot(data=df,x='n',y=f'AUC',hue='method',style='method', palette=palette, markers=markers, dashes=dashes)#,style='Regressor',markers=markers, dashes=dashes)

plt.xscale('log')
#plt.yscale('log')  

plt.legend(bbox_to_anchor=(-1.20, 0.5), loc='center left', borderaxespad=0., fontsize=15)

plt.subplots_adjust(right=0.75)


plt.ylabel(f'AUC',fontsize=15 )
plt.xlabel(r'n',fontsize=15 )
plt.savefig(f"p_values/visualization/AUC_n_p{p}_cor{cor}.pdf", bbox_inches="tight")




plt.figure()
sns.set(rc={'figure.figsize':(4,4)})
sns.lineplot(data=df,x='n',y=f'null_imp',hue='method',style='method',palette=palette, markers=markers, dashes=dashes)#,style='Regressor',markers=markers, dashes=dashes)

plt.xscale('log')
plt.ylim(-0.1, 0.5)
plt.title(f'p = {p} $\\rho$ = {cor}', fontsize=15)
#plt.legend(bbox_to_anchor=(-1.20, 0.5), loc='center left', borderaxespad=0., fontsize=15)
plt.legend().set_visible(False)
plt.subplots_adjust(right=0.75)


plt.ylabel(f'Bias null covariates',fontsize=15 )
plt.xlabel(r'n',fontsize=15 )
plt.savefig(f"p_values/visualization/null_imp_n_p{p}_cor{cor}.pdf", bbox_inches="tight")








plt.figure()
sns.set(rc={'figure.figsize':(4,4)})
sns.lineplot(data=df,x='n',y=f'non_null',hue='method',style='method',palette=palette, markers=markers, dashes=dashes)#,style='Regressor',markers=markers, dashes=dashes)

plt.xscale('log')
plt.ylim(0, 5)

#plt.legend(bbox_to_anchor=(-1.20, 0.5), loc='center left', borderaxespad=0., fontsize=15)
plt.legend().set_visible(False)
plt.subplots_adjust(right=0.75)


plt.ylabel(f'Bias non-null covariates',fontsize=15 )
plt.xlabel(r'n',fontsize=15 )
plt.savefig(f"p_values/visualization/non_null_n_p{p}_cor{cor}.pdf", bbox_inches="tight")





plt.figure()
sns.set(rc={'figure.figsize':(4,4)})
sns.lineplot(data=df,x='n',y=f'tr_time',hue='method',style='method',palette=palette, markers=markers, dashes=dashes)#,style='Regressor',markers=markers, dashes=dashes)

plt.xscale('log')
#plt.ylim(0, 5)

#plt.legend(bbox_to_anchor=(-1.20, 0.5), loc='center left', borderaxespad=0., fontsize=15)
plt.legend().set_visible(False)
plt.subplots_adjust(right=0.75)


plt.ylabel(f'Time',fontsize=15 )
plt.xlabel(r'n',fontsize=15 )
plt.savefig(f"p_values/visualization/time_n_p{p}_cor{cor}.pdf", bbox_inches="tight")


plt.figure()
sns.set(rc={'figure.figsize':(4,4)})
sns.lineplot(data=df,x='n',y=f'power',hue='method',style='method',palette=palette, markers=markers, dashes=dashes)#,style='Regressor',markers=markers, dashes=dashes)

plt.xscale('log')
#plt.ylim(0, 5)

#plt.legend(bbox_to_anchor=(-1.20, 0.5), loc='center left', borderaxespad=0., fontsize=15)
plt.legend().set_visible(False)
plt.subplots_adjust(right=0.75)


plt.ylabel(f'Power',fontsize=15 )
plt.xlabel(r'n',fontsize=15 )
plt.savefig(f"p_values/visualization/power_n_p{p}_cor{cor}.pdf", bbox_inches="tight")


plt.figure()
sns.set(rc={'figure.figsize':(4,4)})
sns.lineplot(data=df,x='n',y=f'type_I',hue='method',style='method',palette=palette, markers=markers, dashes=dashes)#,style='Regressor',markers=markers, dashes=dashes)

plt.xscale('log')
#plt.ylim(0, 5)

#plt.legend(bbox_to_anchor=(-1.20, 0.5), loc='center left', borderaxespad=0., fontsize=15)
plt.legend().set_visible(False)
plt.subplots_adjust(right=0.75)


plt.ylabel(f'Type-I error',fontsize=15 )
plt.xlabel(r'n',fontsize=15 )
plt.savefig(f"p_values/visualization/type_n_p{p}_cor{cor}.pdf", bbox_inches="tight")













