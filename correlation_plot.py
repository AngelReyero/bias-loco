import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from utils import toep

# snr=4
# p=2
# n=100
# intra_cor=[0,0.05, 0.1, 0.2, 0.3, 0.5, 0.65, 0.85]
# cor_meth='toep'
# y_method='lin'
# beta= np.array([2, 1])
# var_to_plot = [0, 1]

snr=4
p=50
n=10000
intra_cor=[0,0.15, 0.3, 0.5, 0.65, 0.85]
cor_meth='toep'
y_method='nonlin'
beta= np.array([2, 1])
super_learner=False

var_to_plot = [0, 1, 6, 7]


def theoretical_curve(y_method, coef_to_plot, intra_cor, beta=[2, 1]):
    if y_method == 'lin':
        return beta[coef_to_plot]**2*(1-intra_cor**2)
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

palette = {'Sobol-CPI(100)': 'purple', 'Sobol-CPI(1)': 'blue', 'LOCO-W':'green', 'PFI':'orange', "LOCO-HD": "red"}

for j in var_to_plot:
    plt.figure()
    sns.set(rc={'figure.figsize':(4,4)})
    sns.lineplot(data=df,x='intra_cor',y=f'imp_V{j}',hue='method',palette=palette)#,style='Regressor',markers=markers, dashes=dashes)
    th_cv= theoretical_curve(y_method, j, np.array(intra_cor), beta=[2, 1])
    plt.plot(intra_cor, th_cv, label=r"Theoretical",linestyle='--', linewidth=1, color="black")

    #plt.ylim((1e-2,1e3))
    #plt.legend()
    
    if j==0:
        plt.legend(bbox_to_anchor=(-1.20, 0.5), loc='center left', borderaxespad=0., fontsize=17)
    else: 
        plt.legend().set_visible(False)



    plt.subplots_adjust(right=0.75)

    #plt.xscale('log')
    #plt.yscale('log')


    plt.ylabel(f'Importance of $X_{j}$',fontsize=17 )
    plt.xlabel(r'Correlation',fontsize=17 )
    if super_learner:
        plt.savefig(f"visualization/correlation_{y_method}_p{p}_n{n}_var{j}_super.pdf", bbox_inches="tight")
    else:
        plt.savefig(f"visualization/correlation_{y_method}_p{p}_n{n}_var{j}.pdf", bbox_inches="tight")


