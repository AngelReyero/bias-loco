import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

import numpy as np
import pandas as pd
from utils import toep




y_method="nonlin"

snr = 4
p = 50
n = 10000
sparsity = 0.1
intra_cor=[0,0.05, 0.15, 0.3, 0.5, 0.65, 0.85]
cor_meth='toep'
beta= np.array([2, 1])
super_learner=False

def theoretical_curve(y_method, j, intra_cor, beta=[2, 1]):
    if y_method == 'lin':
        return beta[j]**2*(1-intra_cor**2)
    elif y_method == 'nonlin' or y_method == 'nonlin2':
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

palette = {'Sobol-CPI(100)': 'purple', 'Sobol-CPI(1)': 'blue', 'LOCO-W':'green', "LOCO-HD": "red"}

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

palette = {'Sobol-CPI(100)': 'purple', 'Sobol-CPI(1)': 'blue', 'LOCO-W':'green', 'PFI':'orange', "LOCO-HD": "red"}

sns.set_style("white")
fig, ax = plt.subplots(2, 3, figsize=(18, 10), gridspec_kw={'hspace': 0.1, 'wspace': 0.3})



# Plot for imp_V0 (top-left subplot)
sns.set(rc={'figure.figsize':(4,4)})
sns.lineplot(data=df, x='intra_cor', y='imp_V0', hue='method', palette=palette, ax=ax[0, 0])  # Top-left subplot

th_cv= theoretical_curve(y_method, 0, np.array(intra_cor), beta=[2, 1])
ax[0, 0].plot(intra_cor, th_cv, label=r"Theoretical",linestyle='--', linewidth=1, color="black")


# Format top-left subplot
#ax[0, 0].set_xscale('log')
ax[0, 0].tick_params(axis='x', labelsize=15)  # Adjust x-axis tick label font size
ax[0, 0].tick_params(axis='y', labelsize=15) 
ax[0, 0].set_xlabel(r'')
ax[0, 0].set_ylabel(f'Importance of $X_0$', fontsize=20)
ax[0, 0].legend().remove()


sns.set(rc={'figure.figsize':(4,4)})
sns.lineplot(data=df, x='intra_cor', y='imp_V1', hue='method', palette=palette, ax=ax[0, 1])  # Top-left subplot

th_cv= theoretical_curve(y_method, 1, np.array(intra_cor), beta=[2, 1])
ax[0, 1].plot(intra_cor, th_cv, label=r"Theoretical",linestyle='--', linewidth=1, color="black")


# Format top-left subplot
#ax[0, 0].set_xscale('log')
ax[0, 1].tick_params(axis='x', labelsize=15)  # Adjust x-axis tick label font size
ax[0, 1].tick_params(axis='y', labelsize=15) 
ax[0, 1].set_xlabel(r'')
ax[0, 1].set_ylabel(f'Importance of $X_1$', fontsize=20)
ax[0, 1].legend().remove()


sns.lineplot(data=df, x='intra_cor', y='imp_V6', hue='method', palette=palette, ax=ax[0, 2])  # Top-left subplot

th_cv= theoretical_curve(y_method, 6, np.array(intra_cor), beta=[2, 1])
ax[0, 2].plot(intra_cor, th_cv, label=r"Theoretical",linestyle='--', linewidth=1, color="black")


# Format top-left subplot
#ax[0, 0].set_xscale('log')
ax[0, 2].tick_params(axis='x', labelsize=15)  # Adjust x-axis tick label font size
ax[0, 2].tick_params(axis='y', labelsize=15) 
ax[0, 2].set_xlabel(r'')
ax[0, 2].set_ylabel(f'Importance of $X_6$', fontsize=20)
ax[0, 2].legend().remove()



sns.lineplot(data=df, x='intra_cor', y='AUC', hue='method', palette=palette, ax=ax[1, 0])  # Bottom-left subplot


# Format bottom-left subplot
#ax[1, 0].set_xscale('log')
ax[1, 0].tick_params(axis='x', labelsize=15)  # Adjust x-axis tick label font size
ax[1, 0].tick_params(axis='y', labelsize=15) 
ax[1, 0].set_xlabel(r'')
ax[1, 0].set_ylabel(f'AUC', fontsize=20)
ax[1, 0].legend().remove()

# Plot for imp_V8 (bottom-right subplot)
sns.lineplot(data=df, x='intra_cor', y='null_imp', hue='method', palette=palette, ax=ax[1, 1])  # Bottom-right subplot


# Format bottom-right subplot
#ax[1, 1].set_xscale('log')
ax[1, 1].tick_params(axis='x', labelsize=15)  # Adjust x-axis tick label font size
ax[1, 1].tick_params(axis='y', labelsize=15) 
ax[1, 1].set_xlabel(r'')
ax[1, 1].set_ylabel(f'Bias null covariates', fontsize=20)
ax[1, 1].legend().remove()

# Plot for imp_V8 (bottom-right subplot)
sns.lineplot(data=df, x='intra_cor', y='tr_time', hue='method', palette=palette, ax=ax[1, 2])  # Bottom-right subplot


# Format bottom-right subplot
ax[1, 2].set_yscale('log')
ax[1, 2].tick_params(axis='x', labelsize=15)  # Adjust x-axis tick label font size
ax[1, 2].tick_params(axis='y', labelsize=15) 
ax[1, 2].set_xlabel(r'')
ax[1, 2].set_ylabel(f'Time', fontsize=20)
ax[1, 2].legend().remove()


# Adjust subplot layout for better spacing
plt.subplots_adjust(hspace=0.4, wspace=0.3)
fig.text(0.5, 0.04, 'Correlation', ha='center', fontsize=20)



if super_learner:
    plt.savefig(f"final_figures/appendix/correlation_{y_method}_p{p}_n{n}_super.pdf", bbox_inches="tight")
else:
    plt.savefig(f"final_figures/appendix/correlation_{y_method}_p{p}_n{n}.pdf", bbox_inches="tight")

# Display the plots



y_method="poly"

snr = 4
p = 50
n = 1000
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

palette = {'Sobol-CPI(100)': 'purple', 'Sobol-CPI(1)': 'blue', 'LOCO-W':'green', "LOCO-HD": "red"}

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

palette = {'Sobol-CPI(100)': 'purple', 'Sobol-CPI(1)': 'blue', 'LOCO-W':'green', 'PFI':'orange', "LOCO-HD": "red"}
sns.set_style("white")

fig, ax = plt.subplots(1, 3, figsize=(18, 5), gridspec_kw={'hspace': 0.1, 'wspace': 0.3})




sns.lineplot(data=df, x='intra_cor', y='AUC', hue='method', palette=palette, ax=ax[0])  # Bottom-left subplot


# Format bottom-left subplot
#ax[1, 0].set_xscale('log')
ax[ 0].tick_params(axis='x', labelsize=15)  # Adjust x-axis tick label font size
ax[ 0].tick_params(axis='y', labelsize=15) 
ax[0].set_xlabel(r'')
ax[ 0].set_ylabel(f'AUC', fontsize=20)
ax[0].legend().remove()

# Plot for imp_V8 (bottom-right subplot)
sns.lineplot(data=df, x='intra_cor', y='null_imp', hue='method', palette=palette, ax=ax[1])  # Bottom-right subplot


# Format bottom-right subplot
#ax[1, 1].set_xscale('log')
ax[1].tick_params(axis='x', labelsize=15)  # Adjust x-axis tick label font size
ax[ 1].tick_params(axis='y', labelsize=15) 
ax[ 1].set_xlabel(r'')
ax[1].set_ylabel(f'Bias null covariates', fontsize=20)
ax[ 1].legend().remove()

# Plot for imp_V8 (bottom-right subplot)
sns.lineplot(data=df, x='intra_cor', y='tr_time', hue='method', palette=palette, ax=ax[2])  # Bottom-right subplot


# Format bottom-right subplot
ax[2].set_yscale('log')
ax[ 2].tick_params(axis='x', labelsize=15)  # Adjust x-axis tick label font size
ax[ 2].tick_params(axis='y', labelsize=15) 
ax[ 2].set_xlabel(r'')
ax[ 2].set_ylabel(f'Time', fontsize=20)
ax[ 2].legend().remove()


# Adjust subplot layout for better spacing
plt.subplots_adjust(hspace=0.4, wspace=0.3)
fig.text(0.5, -0.05, 'Correlation', ha='center', fontsize=20)



if super_learner:
    plt.savefig(f"final_figures/appendix/correlation_{y_method}_p{p}_n{n}_super.pdf", bbox_inches="tight")
else:
    plt.savefig(f"final_figures/appendix/correlation_{y_method}_p{p}_n{n}.pdf", bbox_inches="tight")

# Display the plot



