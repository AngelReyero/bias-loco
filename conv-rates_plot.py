import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from utils import toep


#linear data
# snr=4
# p=2
# cor=0.6
# n_samples=[30, 50, 100, 250, 500, 1000, 2000]
# cor_meth='toep'
# y_method='lin'
# beta= np.array([2, 1])
# var_to_plot = [0, 1]


p=50
cor=0.6
n_samples=[50, 100, 200, 500, 1000, 2000]
y_method = "nonlin"
beta= np.array([2, 1])
cor_meth='toep'
var_to_plot = [0, 1, 6, 7]


def theoretical_curve(y_method, coef_to_plot, cor,p, beta=[2, 1]):
    if y_method == 'lin':
        return beta[coef_to_plot]**2*(1-cor**2)
    elif y_method == 'nonlin':
        if j >4:
            return 0
        elif j==0:
            return (1-cor**2)/2
        elif j==1:
            mat=toep(p, cor)
            sigma_1=mat[1]
            sigma_1=np.delete(sigma_1, 1)
            inv=np.delete(mat, 1, axis=0)
            inv=np.delete(inv, 1, axis=1)
            inv=np.linalg.inv(inv)
            return (1-np.dot(np.dot(sigma_1,inv), sigma_1.T))*0.5


df = pd.read_csv(f"results_csv/conv_rates_{y_method}_p{p}_cor{cor}.csv")

# Display the first few rows of the DataFrame
print(df.head())

palette = {'Robust-CPI': 'purple', '0.5*CPI': 'blue', 'LOCO-W':'green', 'PFI':'orange', "LOCO-HD": "red"}

for j in var_to_plot:
    plt.figure()
    sns.set(rc={'figure.figsize':(4,4)})
    sns.lineplot(data=df,x='n_samples',y=f'imp_V{j}',hue='method',palette=palette)#,style='Regressor',markers=markers, dashes=dashes)
    th_cv= theoretical_curve(y_method, j, cor,p, beta=[2, 1])
    plt.plot(n_samples, [th_cv for _ in n_samples], label=r"Theoretical",linestyle='--', linewidth=1, color="black")

    #plt.ylim((1e-2,1e3))
    #plt.legend()

    plt.legend(bbox_to_anchor=(-1.20, 0.5), loc='center left', borderaxespad=0., fontsize=15)

    plt.subplots_adjust(right=0.75)

    #plt.xscale('log')
    #plt.yscale('log')


    plt.ylabel(f'Importance of $X_{j}$',fontsize=15 )
    plt.xlabel(r'Number of samples',fontsize=15 )
    plt.savefig(f"visualization/conv_rates_{y_method}_p{p}_cor{cor}_var{j}.pdf", bbox_inches="tight")