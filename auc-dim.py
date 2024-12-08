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

snr=4
dim=[10, 20, 35, 50, 100]
max_p=100
n=1000
cor_meth='toep'
cor=0.6
super_learner=True

if super_learner:
    df = pd.read_csv(f"results_csv/dimension_{y_method}_n{n}_cor{cor}_super.csv",)
else: 
    df = pd.read_csv(f"results_csv/dimension_{y_method}_n{n}_cor{cor}.csv",)


# Display the first few rows of the DataFrame
print(df.head())

palette = {'Robust-CPI': 'purple', '0.5*CPI': 'blue', 'LOCO-W':'green', 'PFI':'orange', "LOCO-HD": "red"}

auc_scores = []
null_imp = []

# Iterate through each row of the DataFrame
for index, row in df.iterrows():
    # Extract the predictions for the current experiment (as a list)
    y_pred = row.filter(like="imp_V").values
    y_pred=y_pred[0:row["d"]]
    y=row.filter(like="tr_V").values
    y = np.array(y).astype(int) 
    y=y[0:row['d']]
    auc = roc_auc_score(y, y_pred)
    auc_scores.append(auc)
    null_imp.append(np.mean(abs(y_pred[y==0])))

# Add the AUC scores as a new column to the DataFrame
df['AUC'] = auc_scores
df['null_imp'] = null_imp


plt.figure()
sns.set(rc={'figure.figsize':(4,4)})
sns.lineplot(data=df,x='d',y=f'AUC',hue='method',palette=palette)#,style='Regressor',markers=markers, dashes=dashes)

    

plt.legend(bbox_to_anchor=(-1.20, 0.5), loc='center left', borderaxespad=0., fontsize=15)

plt.subplots_adjust(right=0.75)


plt.ylabel(f'AUC',fontsize=15 )
plt.xlabel(r'Dimension',fontsize=15 )
if super_learner:
    plt.savefig(f"visualization/AUC_dimension_{y_method}_n{n}_cor{cor}_super.pdf", bbox_inches="tight")
else:
    plt.savefig(f"visualization/AUC_dimension_{y_method}_n{n}_cor{cor}.pdf", bbox_inches="tight")



plt.figure()
sns.set(rc={'figure.figsize':(4,4)})
sns.lineplot(data=df,x='d',y=f'null_imp',hue='method',palette=palette)#,style='Regressor',markers=markers, dashes=dashes)

plt.xscale('log')
#plt.ylim(0, 5)
plt.title(f'n = {n} $\\rho$ = {cor}', fontsize=15)
plt.legend(bbox_to_anchor=(-1.20, 0.5), loc='center left', borderaxespad=0., fontsize=15)
plt.legend().set_visible(False)
plt.subplots_adjust(right=0.75)


plt.ylabel(f'Bias null covariates',fontsize=15 )
plt.xlabel(r'Dimension',fontsize=15 )
plt.savefig(f"visualization/dim_null_imp_n{n}_cor{cor}.pdf", bbox_inches="tight")


