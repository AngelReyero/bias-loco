from hidimstat.cpi import CPI
#from hidimstat.loco import LOCO
from hidimstat.permutation_importance import PermutationImportance
from hidimstat.data_simulation import simu_data
import numpy as np
import vimpy
from r_cpi import r_CPI
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LassoCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from loco import LOCO
from utils import GenToysDataset
import argparse
import time



p = 100
ns = [200, 500, 1000, 5000, 10000, 20000]#[100, 300, 500, 700, 1000, 2000, 5000, 10000]
sparsity = 0.2


seed= 0
num_rep=10




cor=0.8

cor_meth='toep'
beta= np.array([2, 1])
snr=2


n_cal=10
n_cal2 = 100
n_jobs=10

best_model=None
dict_model=None

rng = np.random.RandomState(seed)

imp2=np.zeros((13,num_rep, len(ns), p))# 5 because there is 5 methods
tr_imp=np.zeros((num_rep, len(ns), p))
p_val=np.zeros((13,num_rep, len(ns), p))# 5 because there is 5 methods
tim = np.zeros((13, num_rep, len(ns)))
# 0 LOCO-W, 1-3 CPI (-, sqrt, n), 4-6 R-CPI(-, sqrt, n), 7-9 R-CPI2 (-, sqrt, n), 10-12 LOCO

for l in range(num_rep):
    seed+=1
    print("Experiment: "+str(l))
    for (i,n) in enumerate(ns):
        print("With N="+str(n))
        X, y, true_imp = GenToysDataset(n=n, d=p, cor=cor_meth, y_method="poly", k=2, mu=None, rho_toep=cor,  sparsity=sparsity, seed=seed)
        tr_imp[l, i]=true_imp
        start_time = time.time()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
        ntrees = np.arange(100, 500, 100)
        lr = np.arange(.01, .1, .05)
        param_grid = [{'n_estimators':ntrees, 'learning_rate':lr}]
        ## set up cv objects
        model = GridSearchCV(GradientBoostingRegressor(loss = 'squared_error', max_depth = 3), param_grid = param_grid, cv = 5, n_jobs=n_jobs)
        model.fit(X_train, y_train)
        tr_time = time.time()-start_time
        start_time = time.time()
        rob_cpi= r_CPI(
            estimator=model,
            imputation_model=LassoCV(alphas=np.logspace(-3, 3, 10), cv=5, random_state=seed),
            n_permutations=1,
            random_state=seed,
            n_jobs=n_jobs)
        rob_cpi.fit(X_train, y_train)
        imp_time = time.time()-start_time

        start_time = time.time()
        cpi_importance = rob_cpi.score(X_test, y_test, n_cal=1,  p_val='emp_var')
        tim[1, l, i] = time.time() - start_time + tr_time + imp_time
        imp2[1,l,i]= cpi_importance["importance"].reshape((p,))
        p_val[1,l,i]= cpi_importance["pval"].reshape((p,))

        start_time = time.time()
        cpi_importance = rob_cpi.score(X_test, y_test, n_cal=1,  p_val='corrected_sqrt')
        tim[2, l, i] = time.time() - start_time + tr_time + imp_time
        imp2[2,l,i]= cpi_importance["importance"].reshape((p,))
        p_val[2,l,i]= cpi_importance["pval"].reshape((p,))

        start_time = time.time()
        cpi_importance = rob_cpi.score(X_test, y_test, n_cal=1,  p_val='corrected_n')
        tim[3, l, i] = time.time() - start_time + tr_time + imp_time
        imp2[3,l,i]= cpi_importance["importance"].reshape((p,))
        p_val[3,l,i]= cpi_importance["pval"].reshape((p,))

        start_time = time.time()
        cpi_importance = rob_cpi.score(X_test, y_test, n_cal=n_cal,  p_val='emp_var')
        tim[4, l, i] = time.time() - start_time + tr_time + imp_time
        imp2[4,l,i]= cpi_importance["importance"].reshape((p,))
        p_val[4,l,i]= cpi_importance["pval"].reshape((p,))

        start_time = time.time()
        cpi_importance = rob_cpi.score(X_test, y_test, n_cal=n_cal,  p_val='corrected_sqrt')
        tim[5, l, i] = time.time() - start_time + tr_time + imp_time
        imp2[5,l,i]= cpi_importance["importance"].reshape((p,))
        p_val[5,l,i]= cpi_importance["pval"].reshape((p,))

        start_time = time.time()
        cpi_importance = rob_cpi.score(X_test, y_test, n_cal=n_cal,  p_val='corrected_n')
        tim[6, l, i] = time.time() - start_time + tr_time + imp_time
        imp2[6,l,i]= cpi_importance["importance"].reshape((p,))
        p_val[6,l,i]= cpi_importance["pval"].reshape((p,))


        start_time = time.time()
        cpi_importance = rob_cpi.score(X_test, y_test, n_cal=n_cal2,  p_val='emp_var')
        tim[7, l, i] = time.time() - start_time + tr_time + imp_time
        imp2[7,l,i]= cpi_importance["importance"].reshape((p,))
        p_val[7,l,i]= cpi_importance["pval"].reshape((p,))

        start_time = time.time()
        cpi_importance = rob_cpi.score(X_test, y_test, n_cal=n_cal2,  p_val='corrected_sqrt')
        tim[8, l, i] = time.time() - start_time + tr_time + imp_time
        imp2[8,l,i]= cpi_importance["importance"].reshape((p,))
        p_val[8,l,i]= cpi_importance["pval"].reshape((p,))

        start_time = time.time()
        cpi_importance = rob_cpi.score(X_test, y_test, n_cal=n_cal2,  p_val='corrected_n')
        tim[9, l, i] = time.time() - start_time + tr_time + imp_time
        imp2[9,l,i]= cpi_importance["importance"].reshape((p,))
        p_val[9,l,i]= cpi_importance["pval"].reshape((p,))

        start_time = time.time()
        loco = LOCO(
            estimator=model,
            random_state=seed,
            loss=mean_squared_error, 
            n_jobs=n_jobs,
        )
        loco.fit(X_train, y_train)
        tr_loco_time = time.time()-start_time

        start_time = time.time()
        loco_importance = loco.score(X_test, y_test, p_val='emp_var')
        tim[10, l, i] = time.time() - start_time + tr_time + tr_loco_time
        imp2[10,l,i]= loco_importance["importance"].reshape((p,))
        p_val[10,l,i]= loco_importance["pval"].reshape((p,))

        start_time = time.time()
        loco_importance = loco.score(X_test, y_test, p_val='corrected_sqrt')
        tim[11, l, i] = time.time() - start_time + tr_time + tr_loco_time
        imp2[11,l,i]= loco_importance["importance"].reshape((p,))
        p_val[11,l,i]= loco_importance["pval"].reshape((p,))

        start_time = time.time()
        loco_importance = loco.score(X_test, y_test, p_val='corrected_n')
        tim[12, l, i] = time.time() - start_time + tr_time + tr_loco_time
        imp2[12,l,i]= loco_importance["importance"].reshape((p,))
        p_val[12,l,i]= loco_importance["pval"].reshape((p,))


        start_time = time.time()
        #LOCO Williamson
        for j in range(p):
            print("covariate: "+str(j))
            ntrees = np.arange(100, 500, 100)
            lr = np.arange(.01, .1, .05)
            param_grid = [{'n_estimators':ntrees, 'learning_rate':lr}]
            ## set up cv objects
            vimp = vimpy.vim(y = y, x = X, s = j, pred_func = GridSearchCV(GradientBoostingRegressor(loss = 'squared_error', max_depth = 3), param_grid = param_grid, cv = 5, n_jobs=n_jobs), measure_type = "r_squared")
            vimp.get_point_est()
            vimp.get_influence_function()
            vimp.get_se()
            vimp.get_ci()
            vimp.hypothesis_test(alpha = 0.05, delta = 0)
            imp2[0,l,i,j]+=vimp.vimp_*np.var(y)
            p_val[0, l, i, j]=vimp.p_value_
        tim[0, l, i] = time.time() - start_time 


#Save the results
f_res={}
f_res = pd.DataFrame(f_res)
for l in range(num_rep):
    for i in range(13):
        for j in range(len(ns)):
            f_res1={}
            if i==0:
                f_res1["method"] = ["LOCO-W"]
            elif i==1:
                f_res1["method"]=["CPI"]
            elif i==2: 
                f_res1["method"]=["CPI_sqrt"]
            elif i==3:
                f_res1["method"] = ["CPI_n"]
            elif i==4:
                f_res1["method"]=["R-CPI"]
            elif i==5: 
                f_res1["method"]=["R-CPI_sqrt"]
            elif i==6:
                f_res1["method"] = ["R-CPI_n"]
            elif i==7:
                f_res1["method"]=["R-CPI2"]
            elif i==8: 
                f_res1["method"]=["R-CPI2_sqrt"]
            elif i==9:
                f_res1["method"] = ["R-CPI2_n"]
            elif i==10:
                f_res1["method"]=["LOCO"]
            elif i==11: 
                f_res1["method"]=["LOCO_sqrt"]
            elif i==12:
                f_res1["method"] = ["LOCO_n"]
            f_res1["n"]=ns[j]
            for k in range(p):
                f_res1["imp_V"+str(k)]=imp2[i,l, j, k]
                f_res1["tr_V"+str(k)] =tr_imp[l, j, k]
                f_res1["pval"+str(k)] = p_val[i, l, j, k]
            f_res1['tr_time'] = tim[i, l, j]
            f_res1=pd.DataFrame(f_res1)
            f_res=pd.concat([f_res, f_res1], ignore_index=True)


f_res.to_csv(
    f"p_values/results_csv/poly_n_p{p}_cor{cor}.csv",
    index=False,
) 
print(f_res.head())
