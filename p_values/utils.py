import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_memory
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
import random
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from joblib import Parallel, delayed
from sklearn.linear_model import Lasso
from sklearn.ensemble import StackingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
from hidimstat.data_simulation import simu_data
from sklearn.preprocessing import PolynomialFeatures




def hypertune_predictor(estimator, X, y, param_grid, n_jobs=10):
    grid_search = GridSearchCV(estimator, param_grid=param_grid, cv=2, n_jobs=n_jobs, scoring= 'r2')
    grid_search.fit(X, y)
    best_hyperparameters = grid_search.best_params_

    print("Best Hyperparameters:", best_hyperparameters)
    return grid_search.best_estimator_, grid_search.best_score_





def best_mod(X_train, y_train, seed=2024, n_jobs=10, verbose=False, regressor=None, dict_reg=None, super_learner=False):
    if super_learner:
                # Define base estimators
        estimators = [
            ('rf', RandomForestRegressor(random_state=seed)),
            ('lasso', Lasso()),
            ('svr', SVR()),  # Use SVR instead of LinearSVR
            ('hgb', HistGradientBoostingRegressor(random_state=seed))
        ]

        # Define the parameter grid for RandomizedSearchCV
        param_grid = {
            'rf__n_estimators': randint(50, 500), 
            'rf__max_depth': [3, 6, 10], 
            'lasso__alpha': uniform(0.001, 1.0), 
            'svr__C': uniform(0.1, 100),  # C parameter for SVR
            'svr__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # Different kernels for SVR
            'svr__gamma': ['scale', 'auto'],  # Gamma parameter for SVR
            'hgb__max_iter': randint(100, 1000),  # Max iterations for HistGradientBoostingRegressor
            'hgb__learning_rate': uniform(0.01, 0.3),  # Learning rate for HistGradientBoostingRegressor
            'final_estimator__alpha': uniform(0.1, 10)  # Alpha parameter for Ridge
        }

        # Final estimator: Ridge regression
        final_estimator = Ridge()

        # Initialize StackingRegressor
        stacking_regressor = StackingRegressor(
            estimators=estimators,
            final_estimator=final_estimator
        )

        # Perform RandomizedSearchCV to tune hyperparameters
        random_search = RandomizedSearchCV(
            stacking_regressor,
            param_distributions=param_grid,
            n_iter=50,  # Number of iterations for random search
            cv=5,  # Number of cross-validation folds
            random_state=seed,
            n_jobs=n_jobs
        )

        # Fit the model
        random_search.fit(X_train, y_train)

        # Best model
        best_model = random_search.best_estimator_
        score = random_search.best_score_
        if verbose: 
            return best_model, score
        else:
            return best_model
    if regressor is not None:
        model, score=hypertune_predictor(regressor, X_train, y_train, dict_reg, n_jobs=n_jobs)
        if verbose: 
            return model, score
        else:
            return model
    # modelMLP=MLPRegressor(random_state=seed)


    # mlp_param_grid = {
    #     'hidden_layer_sizes': [(50,), (100,), (50, 50)],  # Simplified to 3 options
    #     'activation': ['relu', 'tanh'],  # Focus on most common activations
    #     'solver': ['adam', 'sgd'],  # Keep 2 popular solvers
    #     'alpha': [0.0001, 0.001, 0.01],  # Narrow alpha range
    #     'learning_rate': ['constant', 'adaptive'],  # Focus on the most common learning rates
    #     'learning_rate_init': [0.001, 0.01],  # Two learning rate initialization values
    #     'batch_size': ['auto', 32],  # Focus on default 'auto' and a smaller value
    #     'momentum': [0.9, 0.95]  
    # }

    # modelMLP, MLP_score= hypertune_predictor(modelMLP, X_train, y_train, mlp_param_grid, n_jobs=n_jobs)

    # print("MLP score: "+str(MLP_score))
    modelRF=RandomForestRegressor(random_state=seed)

    rf_param_grid = {
        'n_estimators': [100, 200], 
        'max_depth': [None, 10, 30],  
        'min_samples_split': [2, 10], 
        'min_samples_leaf': [1, 4],  
        'max_features': ['log2', 'sqrt'], 
        'bootstrap': [True] 
    }



    modelRF, RF_score=hypertune_predictor(modelRF, X_train, y_train, rf_param_grid, n_jobs=n_jobs)

    print("RF score: "+str(RF_score))
    modelGB= GradientBoostingRegressor(random_state=seed)

    gb_param_grid = {
        'n_estimators': [100, 300],  
        'learning_rate': [0.01, 0.1], 
        'max_depth': [3, 7], 
        'min_samples_split': [2, 10],  
        'min_samples_leaf': [1, 4],
        'subsample': [0.8, 1.0], 
        'loss': ['squared_error', 'huber']  
    }


    modelGB, GB_score=hypertune_predictor(modelGB, X_train, y_train, gb_param_grid, n_jobs=n_jobs)
    print("GB score: "+str(GB_score))
    modelxgb = XGBRegressor(random_state=seed)

    xgb_param_grid = {
        'n_estimators': [100, 300],  
        'learning_rate': [0.01, 0.1],  
        'max_depth': [3, 7], 
        'min_child_weight': [1, 5],  
        'subsample': [0.8, 1.0],  
        'colsample_bytree': [0.8, 1.0], 
        'gamma': [0, 0.1]  
    }



    modelxgb, xgb_score=hypertune_predictor(modelxgb, X_train, y_train, xgb_param_grid, n_jobs=n_jobs)

    print("XGB score: "+str(xgb_score))

    modelLasso = Lasso(random_state=seed)

# Define the hyperparameter grid for Lasso
    lasso_param_grid = {
        'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],  # Regularization strength
        'max_iter': [1000, 5000, 10000],                  # Maximum number of iterations
        'tol': [1e-4, 1e-3, 1e-2],                       # Tolerance for stopping criteria
    }

# Hypertune the Lasso model
    modelLasso, Lasso_score = hypertune_predictor(modelLasso, X_train, y_train, lasso_param_grid, n_jobs=n_jobs)

    print("Lasso score: " + str(Lasso_score))
    # results = Parallel(n_jobs=n_jobs)(delayed(hypertune_predictor)(model, X_train, y_train, grid) for (model, grid) in [(modelMLP, mlp_param_grid), (modelRF, rf_param_grid), (modelGB, gb_param_grid), (modelxgb, xgb_param_grid), (modelLasso, lasso_param_grid)])
    # best_score=-10000
    # for model, score in results:
    #     if score>best_score:
    #         best_score=score
    #         best_model=model
    models=[modelRF, modelGB, modelxgb, modelLasso]
    scores=[RF_score, GB_score, xgb_score, Lasso_score]
    max_index = scores.index(max(scores))
    print(f'Best model:{max_index} with score {scores[max_index]}')
    #print(f"Best score is:{best_score}")
    if verbose:
        return models[max_index], scores[max_index]
    return models[max_index]#best_model


# covariance matrice 
def ind(i,j,k):
    # separates &,n into k blocks
    return int(i//k==j//k)
# One Toeplitz matrix  
def toep (d, rho=0.6):
  return np.array([[ (rho)**abs(i-j) for i in range(d)]for j in range(d)])

def GenToysDataset(n=1000, d=10, cor='toep', y_method="nonlin", k=2, mu=None, rho_toep=0.6, sparsity=0.1, seed=0, snr=2):
    true_imp=np.zeros(d)
    if y_method=="williamson":
        X1 = np.random.uniform(-1, 1, n)
        X2 = np.random.uniform(-1, 1, n)
        X = np.column_stack((X1, X2))
        epsilon = np.random.normal(0, 1, n)
        y = (25/9) * X1**2+ epsilon
        return X, y, np.array([1, 0])
    if y_method=="hidimstats":
        X, y, _, non_zero_index = simu_data(n, d, rho=rho_toep, sparsity=sparsity, seed=seed, snr=snr)
        true_imp[non_zero_index]=1
        return X, y, true_imp
    X = np.zeros((n,d))
    y = np.zeros(n)
    if mu is None:
        mu=np.zeros(d)
    if cor =='iso': 
        # Generate a simple MCAR distribution, with isotrope observation 
        X= np.random.normal(size=(n,d))
    elif cor =='cor': 
        # Generate a simple MCAR distribution, with anisotropic observations and Sigma=U
        U= np.array([[ind(i,j,k) for j in range(d)] for i in range(d)])/np.sqrt(k)
        X= np.random.normal(size=(n,d))@U+mu
    elif cor =='toep': 
        # Generate un simpler MCAR distribution, with anisotropic observations and Sigma=Toepliz
        X= np.random.multivariate_normal(mu,toep(d, rho_toep),size=n)
    else :
        print("WARNING: key word")
    if y_method == "nonlin":
        y=X[:,0]*X[:,1]*(X[:,2]>0)+2*X[:,3]*X[:,4]*(0>X[:,2])
        true_imp[0:5]=1
    elif y_method == "nonlin2":
        y=X[:,0]*X[:,1]*(X[:,2]>0)+2*X[:,3]*X[:,4]*(0>X[:,2])+X[:, 5]*X[:,6]/2-X[:,7]**2+X[:,9]*(X[:, 8]>0)
        true_imp[0:10]=1
    elif y_method == "lin":
        y=2*X[:,0]+X[:,1]
        true_imp[0:2]=1
    elif y_method =="poly":
        rng = np.random.RandomState(seed)
        non_zero_index = rng.choice(d, int(sparsity*d), replace=False)

        poly_transformer = PolynomialFeatures(
            degree=3, interaction_only=True
        )  
        features = poly_transformer.fit_transform(X[:, non_zero_index])

        # coefficient associated to each feature, can be either -1, or 1 with equal probability
        coef_features = np.random.choice([-1, 1], features.shape[1])
        y = np.dot(features, coef_features)
        true_imp[non_zero_index]=1
    else :
        print("WARNING: key word")
    return X, y, true_imp



def bootstrap_var(imp_list, n_groups=30, size_group=50):
    estim_imp=[]
    for _ in range(n_groups):
        bootstrap_sample = random.choices(imp_list, k=size_group)
        estim_imp.append(np.mean(bootstrap_sample))
    return np.var(estim_imp)
    
