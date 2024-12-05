from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import argparse
from utils import toep



p = 500
sparsity = 0.2


seed= 0
num_rep=10




cor=0.8
cor_meth='toep'
beta= np.array([2, 1])
snr=2


df = pd.read_csv(f"results_csv/n_p{p}_cor{cor}.csv",)


# Display the first few rows of the DataFrame
print(df.head())

palette = {'Robust-CPI': 'purple', '0.5*CPI': 'blue', 'LOCO-W':'green', 'PFI':'orange', "LOCO-HD": "red"}

auc_scores = []
null_imp = []
non_null = []

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
    y=row.filter(like="tr_V").values
    y = np.array(y).astype(int) 
    auc = roc_auc_score(y, y_pred)
    auc_scores.append(auc)
    null_imp.append(np.mean(abs(y_pred[y==0])))
    non_null.append(np.mean(abs(y_pred[y==1]-cond_v[y==1])))

# Add the AUC scores as a new column to the DataFrame
df['AUC'] = auc_scores
df['null_imp'] = null_imp
df['non_null'] = non_null

plt.figure()
sns.set(rc={'figure.figsize':(4,4)})
sns.lineplot(data=df,x='n',y=f'AUC',hue='method',palette=palette)#,style='Regressor',markers=markers, dashes=dashes)

plt.xscale('log')
#plt.yscale('log')  

plt.legend(bbox_to_anchor=(-1.20, 0.5), loc='center left', borderaxespad=0., fontsize=15)

plt.subplots_adjust(right=0.75)


plt.ylabel(f'AUC',fontsize=15 )
plt.xlabel(r'n',fontsize=15 )
plt.savefig(f"visualization/AUC_n_p{p}_cor{cor}.pdf", bbox_inches="tight")




plt.figure()
sns.set(rc={'figure.figsize':(4,4)})
sns.lineplot(data=df,x='n',y=f'null_imp',hue='method',palette=palette)#,style='Regressor',markers=markers, dashes=dashes)

plt.xscale('log')
plt.ylim(0, 5)
plt.title(f'p = {p} $\\rho$ = {cor}', fontsize=15)
#plt.legend(bbox_to_anchor=(-1.20, 0.5), loc='center left', borderaxespad=0., fontsize=15)
plt.legend().set_visible(False)
plt.subplots_adjust(right=0.75)


plt.ylabel(f'Bias null covariates',fontsize=15 )
plt.xlabel(r'n',fontsize=15 )
plt.savefig(f"visualization/null_imp_n_p{p}_cor{cor}.pdf", bbox_inches="tight")








plt.figure()
sns.set(rc={'figure.figsize':(4,4)})
sns.lineplot(data=df,x='n',y=f'non_null',hue='method',palette=palette)#,style='Regressor',markers=markers, dashes=dashes)

plt.xscale('log')
plt.ylim(0, 5)

#plt.legend(bbox_to_anchor=(-1.20, 0.5), loc='center left', borderaxespad=0., fontsize=15)
plt.legend().set_visible(False)
plt.subplots_adjust(right=0.75)


plt.ylabel(f'Bias non-null covariates',fontsize=15 )
plt.xlabel(r'n',fontsize=15 )
plt.savefig(f"visualization/non_null_n_p{p}_cor{cor}.pdf", bbox_inches="tight")












