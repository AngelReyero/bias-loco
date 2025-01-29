import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from utils import toep

snr=4
p=50
n=5000
intra_cor=[0,0.05, 0.15, 0.3, 0.5, 0.65, 0.85]
cor_meth='toep'
y_method='nonlin'
beta= np.array([2, 1])
super_learner=True


n_calib=[1, 5, 20, 50, 100, 250]



def theoretical_curve(y_method, j, intra_cor, beta=[2, 1]):
    if y_method == 'lin':
        return beta[j]**2*(1-intra_cor**2)
    elif y_method == 'nonlin':
        if j >4:
            return [0 for _ in intra_cor]
        elif j==0:
            return (1-intra_cor**2)/2
        elif j==1:
            theo=[]
            for cor in intra_cor:
                mat=toep(p, cor)
                sigma_1=mat[1]
                sigma_1=np.delete(sigma_1, 1)
                inv=np.delete(mat, 1, axis=0)
                inv=np.delete(inv, 1, axis=1)
                inv=np.linalg.inv(inv)
                theo.append((1-np.dot(np.dot(sigma_1,inv), sigma_1.T))*0.5)
            return theo


df = pd.read_csv(f"results_csv/n_cal_{y_method}_p{p}_n{n}.csv",)

# Display the first few rows of the DataFrame
print(df.head())



palette = {
    "n_cal1": "blue",
    "n_cal5": "green",
    "n_cal20": "orange",
    "n_cal50": "purple",
    "n_cal100": "red",
    "n_cal250": "brown",
    "n_cal500": "pink"
}
sns.set_style("white")
fig, ax = plt.subplots(1,2, figsize=(16, 4))


# Plot for imp_V0 (top-left subplot)
sns.set(rc={'figure.figsize':(6,3)})
sns.lineplot(data=df, x='intra_cor', y='imp_V0', hue='method', palette=palette, ax=ax[ 0])  # Top-left subplot
th_cv= theoretical_curve(y_method, 0, np.array(intra_cor), beta=[2, 1])
ax[0].plot(intra_cor, th_cv, label=r"Theoretical",linestyle='--', linewidth=1, color="black")

# Format top-left subplot
ax[0].tick_params(axis='x', labelsize=15)  # Adjust x-axis tick label font size
ax[ 0].tick_params(axis='y', labelsize=15) 
ax[ 0].set_xlabel(r'')
ax[ 0].set_ylabel(f'Importance of $X_0$', fontsize=20)
ax[ 0].legend().remove()

sns.lineplot(data=df, x='intra_cor', y='imp_V6', hue='method', palette=palette, ax=ax[1])  # Top-right subplot
th_cv= theoretical_curve(y_method, 6, np.array(intra_cor), beta=[2, 1])
ax[1].plot(intra_cor, th_cv, label=r"Theoretical",linestyle='--', linewidth=1, color="black")

ax[ 1].tick_params(axis='x', labelsize=15)  # Adjust x-axis tick label font size
ax[ 1].tick_params(axis='y', labelsize=15) 
ax[ 1].set_xlabel(r'')
ax[ 1].set_ylabel(f'Importance of $X_6$', fontsize=20)
ax[ 1].legend().remove()

# Adjust subplot layout for better spacing
plt.subplots_adjust(hspace=0.4, wspace=0.3)
fig.text(0.5, -0.04, 'Correlation', ha='center', fontsize=20)

#handles, labels = ax[0].get_legend_handles_labels()
#fig.legend(handles, labels, loc='center left', bbox_to_anchor=(-0.25, 0.5), fontsize=20)


plt.savefig(f"final_figures/appendix/n_cal{y_method}_n{n}_p{p}.pdf", bbox_inches="tight")



plt.show()