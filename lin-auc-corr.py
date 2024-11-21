from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import argparse





p = 200
n = 500
sparsity = 0.2


seed= 0
num_rep=10




cors=[0, 0.15, 0.3, 0.5, 0.65,0.85]
cor_meth='toep'
beta= np.array([2, 1])
snr=2


df = pd.read_csv(f"results_csv/cor_p{p}_n{n}.csv",)


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
sns.lineplot(data=df,x='cor',y=f'AUC',hue='method',palette=palette)#,style='Regressor',markers=markers, dashes=dashes)

    

plt.legend(bbox_to_anchor=(-1.20, 0.5), loc='center left', borderaxespad=0., fontsize=15)

plt.subplots_adjust(right=0.75)


plt.ylabel(f'AUC',fontsize=15 )
plt.xlabel(r'cor',fontsize=15 )
plt.savefig(f"visualization/AUC_cor_p{p}_n{n}.pdf", bbox_inches="tight")

