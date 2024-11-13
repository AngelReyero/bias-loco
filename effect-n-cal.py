from hidimstat.cpi import CPI
from hidimstat.loco import LOCO
from hidimstat.permutation_importance import PermutationImportance
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

seed= 0
num_rep=10

snr=4
p=50
n=10000
intra_cor=[0,0.05, 0.15, 0.3, 0.5, 0.65, 0.85]
cor_meth='toep'
y_method='nonlin'
beta= np.array([2, 1])
super_learner=True


n_calib=[1, 5, 20, 50, 100, 250]

n_jobs=10

best_model=None
dict_model=None

rng = np.random.RandomState(seed)

imp2=np.zeros((len(n_calib),num_rep, len(intra_cor), p))# 5 because there is 5 methods



for l in range(num_rep):
    print("Experiment: "+str(l))
    for (i,cor) in enumerate(intra_cor):
        print("With correlation="+str(cor))
        X, y, _ = GenToysDataset(n=n, d=p, cor=cor_meth, y_method=y_method, k=2, mu=None, rho_toep=cor)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
    
        model=best_mod(X_train, y_train, seed=seed, regressor=best_model, dict_reg=dict_model, super_learner=super_learner)

        for j, n_cal in enumerate(n_calib):
            rob_cpi= robust_CPI(
                estimator=model,
                imputation_model=LassoCV(alphas=np.logspace(-3, 3, 10), cv=5),
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
            f_res1=pd.DataFrame(f_res1)
            f_res=pd.concat([f_res, f_res1], ignore_index=True)
f_res.to_csv(
    f"results_csv/n_cal_{y_method}_p{p}_n{n}.csv",
    index=False,
) 
print(f_res.head())
