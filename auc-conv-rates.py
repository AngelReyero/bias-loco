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

p=50
cor=0.6
n_samples=[100, 250, 500, 1000, 2000, 5000]
beta= np.array([2, 1])
cor_meth='toep'
super_learner=True

if super_learner:
    df = pd.read_csv(f"results_csv/conv_rates_{y_method}_p{p}_cor{cor}_super.csv",)
else: 
    df = pd.read_csv(f"results_csv/conv_rates_{y_method}_p{p}_cor{cor}.csv",)


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
sns.lineplot(data=df,x='n_samples',y=f'AUC',hue='method',palette=palette)#,style='Regressor',markers=markers, dashes=dashes)

plt.xscale('log')
plt.xticks(fontsize=20)  # X-axis
plt.yticks(fontsize=18)
plt.legend().remove() 
#plt.legend(bbox_to_anchor=(-0.7, 0.5), loc='center left', borderaxespad=0., fontsize=15)

plt.subplots_adjust(right=0.75)


plt.ylabel(f'AUC',fontsize=17 )
plt.xlabel(r'Number of samples',fontsize=17 )
if super_learner:
    plt.savefig(f"visualization/AUC_conv_rates_{y_method}_p{p}_cor{cor}_super.pdf", bbox_inches="tight")
else:
    plt.savefig(f"visualization/AUC_conv_rates_{y_method}_p{p}_cor{cor}.pdf", bbox_inches="tight")




plt.figure()
sns.set(rc={'figure.figsize':(4,4)})
sns.lineplot(data=df,x='n_samples',y=f'null_imp',hue='method',palette=palette)#,style='Regressor',markers=markers, dashes=dashes)

plt.xscale('log')
plt.xticks(fontsize=20)  # X-axis
plt.yticks(fontsize=18)
#plt.ylim(0, 5)
plt.legend().remove() 
#plt.legend(bbox_to_anchor=(-1.1, 0.5), loc='center left', borderaxespad=0., fontsize=15)

plt.subplots_adjust(right=0.75)


plt.ylabel(f'Bias on the null covariates',fontsize=17)
plt.xlabel(r'Number of samples',fontsize=17 )
plt.savefig(f"visualization/null_imp_{y_method}_n_p{p}_cor{cor}.pdf", bbox_inches="tight")

