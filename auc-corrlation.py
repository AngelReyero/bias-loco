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
p = 10
n = 100
sparsity = 0.1
intra_cor=[0.05, 0.5]
cor_meth='toep'
beta= np.array([2, 1])
super_learner=True

if super_learner:
    df = pd.read_csv(f"results_csv/correlation_{y_method}_p{p}_n{n}_super.csv",)
else: 
    df = pd.read_csv(f"results_csv/correlation_{y_method}_p{p}_n{n}.csv",)


# Display the first few rows of the DataFrame
print(df.head())

palette = {'Robust-CPI': 'purple', '0.5*CPI': 'blue', 'LOCO-W':'green', 'PFI':'orange', "LOCO-HD": "red"}

auc_scores = []

# Iterate through each row of the DataFrame
for index, row in df.iterrows():
    # Extract the predictions for the current experiment (as a list)
    y_pred = row.filter(like="imp_V").values
    y=row.filter(like="tr_V").values
    y = np.array(y).astype(int) 
    auc = roc_auc_score(y, y_pred)
    auc_scores.append(auc)

# Add the AUC scores as a new column to the DataFrame
df['AUC'] = auc_scores


plt.figure()
sns.set(rc={'figure.figsize':(4,4)})
sns.lineplot(data=df,x='intra_cor',y=f'AUC',hue='method',palette=palette)#,style='Regressor',markers=markers, dashes=dashes)

    

plt.legend(bbox_to_anchor=(-1.20, 0.5), loc='center left', borderaxespad=0., fontsize=15)

plt.subplots_adjust(right=0.75)


plt.ylabel(f'AUC',fontsize=15 )
plt.xlabel(r'Correlation',fontsize=15 )
if super_learner:
    plt.savefig(f"visualization/AUC_correlation_{y_method}_p{p}_n{n}_super.pdf", bbox_inches="tight")
else:
    plt.savefig(f"visualization/AUC_correlation_{y_method}_p{p}_n{n}.pdf", bbox_inches="tight")


