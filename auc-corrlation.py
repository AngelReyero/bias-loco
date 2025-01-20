from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--y_method', type=str, required=True, help='The y_method to use')
args = parser.parse_args()

y_method = args.y_method

snr = 4
p = 50
n = 10000
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


plt.figure()
sns.set(rc={'figure.figsize':(4,4)})
sns.lineplot(data=df,x='intra_cor',y=f'AUC',hue='method',palette=palette)#,style='Regressor',markers=markers, dashes=dashes)

    

plt.legend(bbox_to_anchor=(-1, 0.5), loc='center left', borderaxespad=0., fontsize=17)

plt.subplots_adjust(right=0.75)


plt.ylabel(f'AUC',fontsize=17 )
plt.xlabel(r'Correlation',fontsize=17 )
if super_learner:
    plt.savefig(f"visualization/AUC_correlation_{y_method}_p{p}_n{n}_super.pdf", bbox_inches="tight")
else:
    plt.savefig(f"visualization/AUC_correlation_{y_method}_p{p}_n{n}.pdf", bbox_inches="tight")



plt.figure()
sns.set(rc={'figure.figsize':(4,4)})
sns.lineplot(data=df,x='intra_cor',y=f'null_imp',hue='method',palette=palette)#,style='Regressor',markers=markers, dashes=dashes)

#plt.xscale('log')
#plt.ylim(0, 5)
#plt.title(f'n={n}, p = {p}', fontsize=15)
#plt.legend(bbox_to_anchor=(-1.20, 0.5), loc='center left', borderaxespad=0., fontsize=15)
plt.legend().set_visible(False)
plt.subplots_adjust(right=0.75)


plt.ylabel(f'Bias null covariates',fontsize=17 )
plt.xlabel(r'Correlation',fontsize=17 )
plt.savefig(f"visualization/corr_{y_method}_null_imp_n{n}_p{p}.pdf", bbox_inches="tight")



plt.figure()
sns.set(rc={'figure.figsize':(4,4)})
sns.lineplot(data=df,x='intra_cor',y=f'tr_time',hue='method',palette=palette)#,style='Regressor',markers=markers, dashes=dashes)

#plt.xscale('log')
plt.yscale("log")
#plt.ylim(0, 5)
#plt.title(f'n={n}, p = {p}', fontsize=15)
#plt.legend(bbox_to_anchor=(-1.20, 0.5), loc='center left', borderaxespad=0., fontsize=15)
plt.legend().set_visible(False)
plt.subplots_adjust(right=0.75)


plt.ylabel(f'Time',fontsize=17 )
plt.xlabel(r'Correlation',fontsize=17 )
plt.savefig(f"visualization/corr_{y_method}_time_n{n}_p{p}.pdf", bbox_inches="tight")

