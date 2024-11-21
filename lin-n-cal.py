from hidimstat.cpi import CPI
from hidimstat.loco import LOCO
from hidimstat.permutation_importance import PermutationImportance
from hidimstat.data_simulation import simu_data
import numpy as np
import vimpy
from robust_cpi import robust_CPI
from utils import GenToysDataset
import pandas as pd
from utils import best_mod
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LassoCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import argparse


seed= 0
num_rep=10

snr=2
p=200
n=500
intra_cor=[0, 0.15, 0.3, 0.5, 0.65, 0.85]
cor_meth='toep'
beta= np.array([2, 1])

sparsity=0.2
n_calib=[1, 5, 20, 50, 100]

n_jobs=10


rng = np.random.RandomState(seed)

imp2=np.zeros((len(n_calib),num_rep, len(intra_cor), p))# 5 because there is 5 methods
tr_imp=np.zeros((num_rep, len(intra_cor), p))



for l in range(num_rep):
    print("Experiment: "+str(l))
    for (i,cor) in enumerate(intra_cor):
        print("With cor="+str(cor))
        true_imp=np.zeros(p)
        X, y, _, non_zero_index = simu_data(n, p, rho=cor, sparsity=sparsity, seed=seed)
        true_imp[non_zero_index]=1
        tr_imp[l, i]=true_imp
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
        model=LassoCV(alphas=np.logspace(-3, 3, 10), cv=5, random_state=seed)
        model.fit(X_train, y_train)
        for j, n_cal in enumerate(n_calib):
            rob_cpi= robust_CPI(
                estimator=model,
                imputation_model=LassoCV(alphas=np.logspace(-3, 3, 10), cv=5, random_state=seed),
                n_permutations=1,
                random_state=seed,
                n_jobs=n_jobs,
                n_cal=n_cal)
            rob_cpi.fit(X_train, y_train)
            rob_importance = rob_cpi.score(X_test, y_test)
            imp2[j,l,i]= rob_importance["importance"].reshape((p,))
        

#Save the results
f_res={}
f_res = pd.DataFrame(f_res)
for l in range(num_rep):
    for i, n_cal in enumerate(n_calib):#n_calib
        for j in range(len(intra_cor)):
            f_res1={}
            f_res1["method"] = [f"n_cal{n_cal}"]
            f_res1["intra_cor"]=intra_cor[j]
            for k in range(p):
                f_res1["imp_V"+str(k)]=imp2[i,l, j, k]
                f_res1["tr_V"+str(k)] =tr_imp[l, j, k]
            f_res1=pd.DataFrame(f_res1)
            f_res=pd.concat([f_res, f_res1], ignore_index=True)
f_res.to_csv(
    f"results_csv/lin_n_cal_p{p}_n{n}.csv",
    index=False,
) 
print(f_res.head())
