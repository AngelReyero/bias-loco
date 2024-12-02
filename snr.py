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




p = 200
n = 500
sparsity = 0.2


seed= 0
num_rep=10




cor=0.8
cor_meth='toep'
beta= np.array([2, 1])
snrs=[0.01, 0.1, 0.5, 1, 2, 5, 20]


n_cal=10
n_jobs=10

best_model=None
dict_model=None

rng = np.random.RandomState(seed)

imp2=np.zeros((5,num_rep, len(snrs), p))# 5 because there is 5 methods
tr_imp=np.zeros((num_rep, len(snrs), p))


for l in range(num_rep):
    seed+=1
    print("Experiment: "+str(l))
    for (i,snr) in enumerate(snrs):
        print("With SNR="+str(snr))
        true_imp=np.zeros(p)
        X, y, _, non_zero_index = simu_data(n, p, rho=cor, sparsity=sparsity, seed=seed, snr=snr)
        true_imp[non_zero_index]=1
        tr_imp[l, i]=true_imp
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
        model=LassoCV(alphas=np.logspace(-3, 3, 10), cv=5, random_state=seed)
        model.fit(X_train, y_train)
    
        rob_cpi= robust_CPI(
            estimator=model,
            imputation_model=LassoCV(alphas=np.logspace(-3, 3, 10), cv=5, random_state=seed),
            n_permutations=1,
            random_state=seed,
            n_jobs=n_jobs,
            n_cal=n_cal)
        rob_cpi.fit(X_train, y_train)
        rob_importance = rob_cpi.score(X_test, y_test)
        imp2[4,l,i]= rob_importance["importance"].reshape((p,))
       
        cpi= robust_CPI(
            estimator=model,
            imputation_model=LassoCV(alphas=np.logspace(-3, 3, 10), cv=5, random_state=seed),
            n_permutations=1,
            random_state=seed,
            n_jobs=n_jobs,
            n_cal=1)
        cpi.fit(X_train, y_train)
        cpi_importance = cpi.score(X_test, y_test)
        imp2[0,l,i]= cpi_importance["importance"].reshape((p,))

        pi = PermutationImportance(
            estimator=model,
            n_permutations=1,
            random_state=seed,
            n_jobs=n_jobs,
        )
        pi.fit(X_train, y_train)
        pi_importance = pi.score(X_test, y_test)
        imp2[1,l,i]= pi_importance["importance"].reshape((p,))

       
        #LOCO Williamson
        for j in range(p):
            print("covariate: "+str(j))
            vimp = vimpy.vim(y = y, x = X, s = j, pred_func = LassoCV(alphas=np.logspace(-3, 3, 10), cv=5, random_state=seed), measure_type = "r_squared")
            vimp.get_point_est()
            vimp.get_influence_function()
            vimp.get_se()
            vimp.get_ci()
            vimp.hypothesis_test(alpha = 0.05, delta = 0)
            imp2[2,l,i,j]+=vimp.vimp_*np.var(y)
        #LOCO Hidimstat

        loco = LOCO(
            estimator=model,
            random_state=seed,
            loss=mean_squared_error, 
            n_jobs=n_jobs,
        )
        loco.fit(X_train, y_train)
        loco_importance = loco.score(X_test, y_test)
        imp2[3,l,i]= loco_importance["importance"].reshape((p,))




#Save the results
f_res={}
f_res = pd.DataFrame(f_res)
for l in range(num_rep):
    for i in range(5):#CPI, PFI, LOCO_W, LOCO_AC, Robust-Loco
        for j in range(len(snrs)):
            f_res1={}
            if i==0:
                f_res1["method"] = ["0.5*CPI"]
            elif i==1:
                f_res1["method"]=["PFI"]
            elif i==2: 
                f_res1["method"]=["LOCO-W"]
            elif i==3:
                f_res1["method"]=["LOCO-HD"]
            else:
                f_res1["method"]=["Robust-CPI"]
            f_res1["snr"]=snrs[j]
            for k in range(p):
                f_res1["imp_V"+str(k)]=imp2[i,l, j, k]
                f_res1["tr_V"+str(k)] =tr_imp[l, j, k]
            f_res1=pd.DataFrame(f_res1)
            f_res=pd.concat([f_res, f_res1], ignore_index=True)


f_res.to_csv(
    f"results_csv/snr_p{p}_n{n}.csv",
    index=False,
) 
print(f_res.head())
