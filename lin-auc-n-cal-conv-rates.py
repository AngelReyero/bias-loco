from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import argparse




p = 500
ns = [100, 300, 500, 700, 1000, 2000, 5000]
sparsity = 0.2


cor=0.8
cor_meth='toep'
beta= np.array([2, 1])
snr=2


df = pd.read_csv(f"results_csv/lin_n_cal_conv_rates_p{p}_cor{cor}.csv",)


# Display the first few rows of the DataFrame
print(df.head())


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
sns.lineplot(data=df,x='n',y=f'AUC',hue='method')#,style='Regressor',markers=markers, dashes=dashes)

plt.xscale('log')
#plt.yscale('log')

plt.legend(bbox_to_anchor=(-1.20, 0.5), loc='center left', borderaxespad=0., fontsize=15)

plt.subplots_adjust(right=0.75)


plt.ylabel(f'AUC',fontsize=15 )
plt.xlabel(r'n',fontsize=15 )
plt.savefig(f"visualization/AUC_lin_n_cal_conv_rates_p{p}_cor{cor}.pdf", bbox_inches="tight")
