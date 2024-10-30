import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from utils import toep

snr=4
dim=[10, 20, 35, 50, 100]
min_p=10
n=1000
cor_meth='toep'
cor=0.6
y_method='nonlin'

var_to_plot = [0, 1, 6, 7]

def theoretical_curve(y_method, coef_to_plot, dim, cor, beta=[2, 1]):
    if y_method == 'lin':
        return beta[coef_to_plot]**2*(1-cor**2)#TO CHANGE
    elif y_method == 'nonlin':
        if j >4:
            return [0 for _ in dim]
        elif j==0:
            return [(1-cor**2)/2 for _ in dim]
        elif j==1:
            theo=[]
            for p in dim:
                mat=toep(p, cor)
                sigma_1=mat[1]
                sigma_1=np.delete(sigma_1, 1)
                inv=np.delete(mat, 1, axis=0)
                inv=np.delete(inv, 1, axis=1)
                inv=np.linalg.inv(inv)
                theo.append((1-np.dot(np.dot(sigma_1,inv), sigma_1.T))*0.5)
            return theo


df = pd.read_csv(f"results_csv/dimension_{y_method}_n{n}_cor{cor}.csv",)

# Display the first few rows of the DataFrame
print(df.head())

palette = {'Robust-CPI': 'purple', '0.5*CPI': 'blue', 'LOCO-W':'green', 'PFI':'orange', "LOCO-HD": "red"}

for j in var_to_plot:
    plt.figure()
    sns.set(rc={'figure.figsize':(4,4)})
    sns.lineplot(data=df,x='d',y=f'imp_V{j}',hue='method',palette=palette)#,style='Regressor',markers=markers, dashes=dashes)
    th_cv= theoretical_curve(y_method, j, np.array(dim), cor, beta=[2, 1])
    plt.plot(dim, th_cv, label=r"Theoretical",linestyle='--', linewidth=1, color="black")

    #plt.ylim((1e-2,1e3))
    #plt.legend()

    plt.legend(bbox_to_anchor=(-1.20, 0.5), loc='center left', borderaxespad=0., fontsize=15)

    plt.subplots_adjust(right=0.75)

    #plt.xscale('log')
    #plt.yscale('log')


    plt.ylabel(f'Importance of $X_{j}$',fontsize=15 )
    plt.xlabel(r'Dimesion',fontsize=15 )
    plt.savefig(f"visualization/dimension_{y_method}_n{n}_cor{cor}_var{j}.pdf", bbox_inches="tight")


