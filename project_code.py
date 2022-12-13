# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 20:06:05 2019

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 18:03:34 2019

@author: user
"""


import xgboost
import numpy as np

import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import seaborn as sns

from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor


from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor

#, StackingRegressor
from sklearn.utils import shuffle
import lightgbm


import csv

methodMSE='uniform_average'
methodR2='uniform_average'

def MultiOutputKFold(number_splits, modelCV, x, y, shufle):
    R2score = []
    cv = KFold(n_splits = number_splits, random_state = 42, shuffle = shufle)
    for train_index, test_index in cv.split(x.values):
#        print("Train Index: ", train_index, "\n")
#        print("Test Index: ", test_index)
        X_train, X_test, Y_train, Y_test = x.values[train_index], x.values[test_index], y12[train_index], y12[test_index]
        classifierCV = MultiOutputRegressor(modelCV, n_jobs=-1)
        classiCV = classifierCV.fit(X_train, Y_train)
        prediction = classiCV.predict(X_test)
        R2score.append(r2_score(Y_test, prediction, multioutput = 'uniform_average'))
        
    return R2score

# %%

df = pd.read_csv('train.csv', sep = ',')
df_test = pd.read_csv('test.csv', sep = ',')

dff=df
col=dff.columns
#dff=dff.drop([col[0], col[1]], axis=1)
dff=dff.drop([col[0], col[1],'batch_size_test','batch_size_val','criterion','optimizer','batch_size_train'], axis=1)


dff_test = df_test
col=dff_test.columns
#dff_test=dff_test.drop([col[0], col[1]], axis=1)
dff_test=dff_test.drop([col[0], col[1],'batch_size_test','batch_size_val','criterion','optimizer','batch_size_train'], axis=1)



# %%

## TRAIN DATA PREPROCESSING #########################################
##################################################################

#
#lbl = preprocessing.LabelEncoder()
#arch_and_hp_tr=lbl.fit_transform(df['arch_and_hp'].astype(str))
#init_params_mu_tr=lbl.fit_transform(df['init_params_mu'].astype(str))
#init_params_std_tr=lbl.fit_transform(df['init_params_std'].astype(str))
#init_params_l2_tr=lbl.fit_transform(df['init_params_l2'].astype(str))
#
#
#
#
#arch_and_hp_sc=arch_and_hp_tr/(np.max(arch_and_hp_tr)-np.min(arch_and_hp_tr))
#init_params_mu_sc=init_params_mu_tr/(np.max(init_params_mu_tr)-np.min(init_params_mu_tr))
#init_params_std_sc=init_params_std_tr/(np.max(init_params_std_tr)-np.min(init_params_std_tr))
#init_params_l2_sc=init_params_l2_tr/(np.max(init_params_l2_tr)-np.min(init_params_l2_tr))
#
#
#vals = df[['epochs','number_parameters']].values #returns a numpy array
#min_max_scaler = preprocessing.MinMaxScaler()
#vals_scaled = min_max_scaler.fit_transform(vals)
#epochs_sc = pd.DataFrame(vals_scaled)[0]
#number_params_sc = pd.DataFrame(vals_scaled)[1]
#
#
#dff.arch_and_hp = arch_and_hp_sc
#dff.init_params_mu = init_params_mu_sc
#dff.init_params_std = init_params_std_sc
#dff.init_params_l2 = init_params_l2_sc
#dff.epochs = epochs_sc
#dff.number_parameters = number_params_sc
#
#
### TEST DATA PREPROCESSING ##############################################
#######################################################################
#
#lbl = preprocessing.LabelEncoder()
#arch_and_hp_tst=lbl.fit_transform(df_test['arch_and_hp'].astype(str))
#init_params_mu_tst=lbl.fit_transform(df_test['init_params_mu'].astype(str))
#init_params_std_tst=lbl.fit_transform(df_test['init_params_std'].astype(str))
#init_params_l2_tst=lbl.fit_transform(df_test['init_params_l2'].astype(str))
#
#
#arch_and_hp_sc_tst=arch_and_hp_tst/(np.max(arch_and_hp_tst)-np.min(arch_and_hp_tst))
#init_params_mu_sc_tst=init_params_mu_tst/(np.max(init_params_mu_tst)-np.min(init_params_mu_tst))
#init_params_std_sc_tst=init_params_std_tst/(np.max(init_params_std_tst)-np.min(init_params_std_tst))
#init_params_l2_sc_tst=init_params_l2_tst/(np.max(init_params_l2_tst)-np.min(init_params_l2_tst))
#
#
#vals_test = df_test[['epochs','number_parameters']].values #returns a numpy array
#vals_test_scaled = min_max_scaler.fit_transform(vals_test)
#epochs_sc_tst = pd.DataFrame(vals_test_scaled)[0]
#number_params_sc_tst = pd.DataFrame(vals_test_scaled)[1]
#
#
#dff_test.arch_and_hp = arch_and_hp_sc_tst
#dff_test.init_params_mu = init_params_mu_sc_tst
#dff_test.init_params_std = init_params_std_sc_tst
#dff_test.init_params_l2 = init_params_l2_sc_tst
#dff_test.epochs = epochs_sc_tst
#dff_test.number_parameters = number_params_sc_tst



# %% EVERYTHING NORMALIZED

## TRAIN DATA PREPROCESSING #########################################
##################################################################


lbl = preprocessing.LabelEncoder()
#arch_and_hp_tr=lbl.fit_transform(df['arch_and_hp'].astype(str))
#init_params_mu_tr=lbl.fit_transform(df['init_params_mu'].astype(str))
#init_params_std_tr=lbl.fit_transform(df['init_params_std'].astype(str))
#init_params_l2_tr=lbl.fit_transform(df['init_params_l2'].astype(str))

dff.arch_and_hp = lbl.fit_transform(df['arch_and_hp'].astype(str))
dff.init_params_mu = lbl.fit_transform(df['init_params_mu'].astype(str))
dff.init_params_std = lbl.fit_transform(df['init_params_std'].astype(str))
dff.init_params_l2 = lbl.fit_transform(df['init_params_l2'].astype(str))




#arch_and_hp_sc=arch_and_hp_tr/(np.max(arch_and_hp_tr)-np.min(arch_and_hp_tr))
#init_params_mu_sc=init_params_mu_tr/(np.max(init_params_mu_tr)-np.min(init_params_mu_tr))
#init_params_std_sc=init_params_std_tr/(np.max(init_params_std_tr)-np.min(init_params_std_tr))
#init_params_l2_sc=init_params_l2_tr/(np.max(init_params_l2_tr)-np.min(init_params_l2_tr))
#
#
#vals = df[['epochs','number_parameters']].values #returns a numpy array
#min_max_scaler = preprocessing.MinMaxScaler()
#vals_scaled = min_max_scaler.fit_transform(vals)
#epochs_sc = pd.DataFrame(vals_scaled)[0]
#number_params_sc = pd.DataFrame(vals_scaled)[1]


#dff.arch_and_hp = arch_and_hp_sc
#dff.init_params_mu = init_params_mu_sc
#dff.init_params_std = init_params_std_sc
#dff.init_params_l2 = init_params_l2_sc
#dff.epochs = epochs_sc
#dff.number_parameters = number_params_sc


## TEST DATA PREPROCESSING ##############################################
######################################################################

lbl = preprocessing.LabelEncoder()
#arch_and_hp_tst=lbl.fit_transform(df_test['arch_and_hp'].astype(str))
#init_params_mu_tst=lbl.fit_transform(df_test['init_params_mu'].astype(str))
#init_params_std_tst=lbl.fit_transform(df_test['init_params_std'].astype(str))
#init_params_l2_tst=lbl.fit_transform(df_test['init_params_l2'].astype(str))

dff_test.arch_and_hp = lbl.fit_transform(df_test['arch_and_hp'].astype(str))
dff_test.init_params_mu = lbl.fit_transform(df_test['init_params_mu'].astype(str))
dff_test.init_params_std = lbl.fit_transform(df_test['init_params_std'].astype(str))
dff_test.init_params_l2 = lbl.fit_transform(df_test['init_params_l2'].astype(str))


#arch_and_hp_sc_tst=arch_and_hp_tst/(np.max(arch_and_hp_tst)-np.min(arch_and_hp_tst))
#init_params_mu_sc_tst=init_params_mu_tst/(np.max(init_params_mu_tst)-np.min(init_params_mu_tst))
#init_params_std_sc_tst=init_params_std_tst/(np.max(init_params_std_tst)-np.min(init_params_std_tst))
#init_params_l2_sc_tst=init_params_l2_tst/(np.max(init_params_l2_tst)-np.min(init_params_l2_tst))


#vals_test = df_test[['epochs','number_parameters']].values #returns a numpy array
#vals_test_scaled = min_max_scaler.fit_transform(vals_test)
#epochs_sc_tst = pd.DataFrame(vals_test_scaled)[0]
#number_params_sc_tst = pd.DataFrame(vals_test_scaled)[1]


#dff_test.arch_and_hp = arch_and_hp_sc_tst
#dff_test.init_params_mu = init_params_mu_sc_tst
#dff_test.init_params_std = init_params_std_sc_tst
#dff_test.init_params_l2 = init_params_l2_sc_tst
#dff_test.epochs = epochs_sc_tst
#dff_test.number_parameters = number_params_sc_tst

# %% FEATURE ENGINEERING

#final_val_accs = df.iloc[:,df.columns.get_loc('val_accs_49')]
#final_train_accs = df.iloc[:,df.columns.get_loc('train_accs_49')]
#
#
#dff.insert(7,"val_accs",final_val_accs, True)
#dff.insert(8,"train_accs",final_train_accs, True)
#
#
#val_loss_index = dff.columns.get_loc('val_loss')
#val_accs_index = dff.columns.get_loc('val_accs')
#
#
#final_val_loss_test = df_test.iloc[:,df_test.columns.get_loc('val_losses_49')]
#final_train_loss_test = df_test.iloc[:,df_test.columns.get_loc('train_losses_49')]
#
#
#final_val_accs_test = df_test.iloc[:,df_test.columns.get_loc('val_accs_49')]
#final_train_accs_test = df_test.iloc[:,df_test.columns.get_loc('train_accs_49')]
#
#dff_test.insert(val_loss_index-1, "val_loss",final_val_loss_test, True)
#dff_test.insert(val_loss_index, "train_loss",final_train_loss_test, True)
#
#
#dff_test.insert(val_accs_index-2,"val_accs",final_val_accs_test, True)
#dff_test.insert(val_accs_index-1,"train_accs",final_train_accs_test, True)


# for val_accs, train_accs only
#dff_test.insert(val_accs_index-5 ,"val_accs",final_val_accs_test, True)
#dff_test.insert(val_accs_index-4,"train_accs",final_train_accs_test, True)

#%% ######## ASSIGN OUTPUT VARIABLES AND PREDICTORS

y1=df.loc[:,'val_error']
y2=df.loc[:,'train_error']
y12 = np.c_[y1,y2]
#y12=y_tst
#x=dff.drop(['val_error', 'train_error'], axis=1)
x=dff.drop(['val_error','val_loss', 'train_error','train_loss'], axis=1)
#x=x_tst
#x, x_tst, y12, y_tst = train_test_split(x, y12, test_size=0.2, random_state=42)
x_test = dff_test

#scaler = preprocessing.StandardScaler()
#scaler.fit(x)
#x = scaler.transform(x)
#x_tst = scaler.transform(x_tst)
# #x_test = scaler.transform(x_test)



scaler = preprocessing.MinMaxScaler()
scaler.fit(x)

x = scaler.transform(x)
#x_tst = scaler.transform(x_tst)   # uncomment if using train/test split
x_test = scaler.transform(x_test)


# %%
#y1=df.loc[:,'val_error']
#y2=df.loc[:,'train_error']
#y12 = np.c_[y1,y2]
##y12=y_tst
##x=dff.drop(['val_error', 'train_error'], axis=1)
#x=dff.drop(['val_error','val_loss', 'train_error','train_loss'], axis=1)
##x=x_tst
##x, x_tst, y12, y_tst = train_test_split(x, y12, test_size=0.6, random_state=42)
##x_valid, x_tst, y_valid, y_tst = train_test_split(x, y12, test_size=0.2, random_state=42)
#x_test = dff_test
#
##scaler = preprocessing.StandardScaler()
##scaler.fit(x)
##x = scaler.transform(x)
##x_tst = scaler.transform(x_tst)
## #x_test = scaler.transform(x_test)
#
#
#
#scaler = preprocessing.MinMaxScaler()
#scaler.fit(x)
#
#x = scaler.transform(x)
##x_tst = scaler.transform(x_tst)   # uncomment if using train/test split
##x_valid = scaler.transform(x_valid)   # uncomment if using train/test split
#x_test = scaler.transform(x_test)


# %%
######### VARIOUS RANDOM FORESTS ON CLEANED #############################
###########################################################################

### TUNED VIA RANDOM SEARCH CLEANED

#bestRFparams_cleaned={'bootstrap': True,
# 'max_depth': 142,
# 'max_features': 'auto',
# 'min_samples_leaf': 1,
# 'min_samples_split': 2,
# 'n_estimators': 1133}
#
#forestcl = RandomForestRegressor(**bestRFparams_cleaned)
#classifiercl = MultiOutputRegressor(forestcl, n_jobs=-1)
#classicl=classifiercl.fit(x, y12)
#
#pred_treecl=classicl.predict(x_tst)
##print(mean_squared_error(y_tst, pred_treecl, multioutput=methodMSE))
#print(r2_score(y_tst, pred_treecl, multioutput=methodR2))
#
#preds_oof_treecl = cross_val_predict(classifiercl, x, y12, cv = 5)
#
#
#### TUNED VIA RANDOM SEARCH
#
#bestRFparams={'bootstrap': True,
# 'max_depth': 100,
# 'max_features': 'auto',
# 'min_samples_leaf': 1,
# 'min_samples_split': 2,
# 'n_estimators': 1400}
#
#forest = RandomForestRegressor(**bestRFparams)
#classifier = MultiOutputRegressor(forest, n_jobs=-1)
#classi=classifier.fit(x, y12)
#
#pred_tree=classi.predict(x_tst)
##print(mean_squared_error(y_tst, pred_tree, multioutput='raw_values'))
#print(r2_score(y_tst, pred_tree, multioutput='uniform_average'))
#
#
#preds_oof_tree = cross_val_predict(classifier, x, y12, cv = 5)
#
#
#### TUNED VIA RANDOM SEARCH AND GRID SEARCH
#
#bestRFparams_random_grid = {'bootstrap': True,
# 'max_depth': 120,
# 'max_features': 'auto',
# 'min_samples_leaf': 1,
# 'min_samples_split': 2,
# 'n_estimators': 1400}
#
#forest1 = RandomForestRegressor(**bestRFparams_random_grid)
#classifier1 = MultiOutputRegressor(forest1, n_jobs=-1)
#classi1=classifier1.fit(x, y12)
#
#pred_tree1=classi1.predict(x_tst)
##print(mean_squared_error(y_tst, pred_tree1, multioutput='raw_values'))
#print("r2",r2_score(y_tst, pred_tree1, multioutput='uniform_average'))
#
#
#preds_oof_tree1 = cross_val_predict(classifier1, x, y12, cv = 5)
#


# %%
### UNTUNED

RFparams={'bootstrap': True,
 'max_features': 0.7,
 'min_samples_leaf': 4,
 'min_samples_split': 2,
 'n_estimators': 100}

#avg_r2 = np.mean(MultiOutputKFold(5, RandomForestRegressor(**RFparams, random_state=1), x, y12, True))


forestU = RandomForestRegressor(**RFparams, random_state=1)
classifierU = MultiOutputRegressor(forestU, n_jobs=-1)
classiU=classifierU.fit(x, y12)

pred_treeU=classiU.predict(x_tst)
#print(mean_squared_error(y_tst, pred_treeU, multioutput=methodMSE))
print("r2",r2_score(y_tst, pred_treeU, multioutput=methodR2))


#preds_oof_treeU = cross_val_predict(classifierU, x, y12, cv = 5)


#
#feat_impts = [] 
#for clf in classiU.estimators_:
#    feat_impts.append(clf.feature_importances_)
#
#importance = np.mean(feat_impts, axis=0)
###############################################################################
###############################################################################



# %%

##################################### XGBOOST #############################
###########################################################################

#param_reg = {'objective':'reg:squarederror', 'colsample_bytree' : 0.3, 'learning_rate' : 0.1, 
#          'max_depth' : 5, 'alpha' : 10, 'eta' : 0.0000005, 'n_estimators' : 10}
param_reg={'booster':'gbtree','objective':'reg:squarederror',   'learning_rate':0.04, 'sample_type':'uniform', 
           'colsample_bytree':1, 'subsample':1, 'n_estimators':1000, 
           'reg_alpha':0.2, 'max_depth':4, 'gamma':0.0, 'alpha':0}

xg_reg = xgb.XGBRegressor(**param_reg)
xgb_reg_multioutput = MultiOutputRegressor(xg_reg).fit(x, y12)


pred_reg = xgb_reg_multioutput.predict(x_tst)
#print(mean_squared_error(y_tst, pred_reg, multioutput = methodMSE))
print("r2",r2_score(y_tst, pred_reg, multioutput = methodR2))

###############################################################################
###############################################################################

#preds_oof_xgboost = cross_val_predict(xgb_reg_multioutput, x, y12, cv = 5)


# %%

##################################### XGBOOST #############################
###########################################################################


param_reg1={'objective':'reg:squarederror',   'learning_rate':0.04, 
           'colsample_bytree':1, 'subsample':1, 'n_estimators':100, 
           'reg_alpha':0.2, 'max_depth':4, 'gamma':0.0, 'alpha':0}

xg_reg1 = xgb.XGBRegressor(**param_reg1)
xgb_reg_multioutput1 = MultiOutputRegressor(xg_reg1).fit(x, y12)


pred_reg1 = xgb_reg_multioutput1.predict(x_tst)
print("r2",r2_score(y_tst, pred_reg1, multioutput = methodR2))

###############################################################################
###############################################################################

# %%

##################################### LIGHTGBM #############################
###########################################################################

param_lightreg={'boosting':'gbdt','objective':'cross_entropy', 'learning_rate':0.1, 'max_bin':255, 
           'colsample_bytree':0.8, 'subsample':1, 'n_estimators':800, 
           'reg_alpha':0.3, 'max_depth':4, 'gamma':0.0, 'min_data_in_leaf' : 25,'num_leaves ':24}

# For better accuracy: large max_bin (may be slower), small learning_rate with large num_iterations
# large num_leaves(may cause over-fitting), bigger training data, dart, categorical feature directly

# less over-fitting: small max_bin, small num_leaves, min_data_in_leaf and min_sum_hessian_in_leaf
# bagging by set bagging_fraction and bagging_freq, feature sub-sampling by set feature_fraction
# bigger training data, lambda_l1, lambda_l2 and min_gain_to_split to regularization
# max_depth to avoid growing deep tree
    
# Ideally, the value of num_leaves should be less than or equal to 2^(max_depth)

lightg = MultiOutputRegressor(lightgbm.LGBMRegressor(**param_lightreg), n_jobs=-1)
lightg.fit(x, y12) 
lightg_pred = lightg.predict(x_tst)

#print(mean_squared_error(y_tst, lightg_pred, multioutput = methodMSE))
print("r2",r2_score(y_tst, lightg_pred, multioutput = methodR2))

#preds_oof_lightg = cross_val_predict(lightg, x, y12, cv = 5)



###############################################################################
###############################################################################

# %%

##################################### LIGHTGBM #############################
###########################################################################

param_lightreg1={'boosting':'gbdt','objective':'regression', 'learning_rate':0.1, 'max_bin':255, 
           'colsample_bytree':0.8, 'subsample':1, 'n_estimators':2000, 
           'reg_alpha':0.3, 'max_depth':4, 'gamma':0.0, 'min_data_in_leaf' : 25,'num_leaves ':24}

lightg1 = MultiOutputRegressor(lightgbm.LGBMRegressor(**param_lightreg1), n_jobs=-1)
lightg1.fit(x, y12) 
lightg_pred1 = lightg1.predict(x_tst)

#print(mean_squared_error(y_tst, lightg_pred, multioutput = methodMSE))
print("r2",r2_score(y_tst, lightg_pred1, multioutput = methodR2))



###############################################################################
###############################################################################


# %% Neural Networks 300,200,100  , r2 = 0.93659

mlp = MLPRegressor(hidden_layer_sizes=(300,200,100),  activation='relu', solver='lbfgs', alpha=0.0002, batch_size='auto',
               learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
               random_state=0, tol=0.0001, verbose=False, warm_start=False, momentum=0.9,
               nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
               epsilon=1e-08)

mlp_model = mlp.fit(x, y12)
mlp_model_pred = mlp_model.predict(x_tst)

print("r2",r2_score(y_tst, mlp_model_pred, multioutput = methodR2))
print(mean_squared_error(y_tst, mlp_model_pred, multioutput = methodMSE))

#(200,200,100,300,200)  
#(150,300,500,750) 0.932
#(150,300,500,350) 0.933


# %% Neural Networks 300,200,100,  r2=0.935207

mlp1 = MLPRegressor(hidden_layer_sizes=(300,200,100,300,200),  activation='relu', solver='lbfgs', alpha=0.0002, batch_size='auto',
               learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
               random_state=0, tol=0.0001, verbose=False, warm_start=False, momentum=0.9,
               nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
               epsilon=1e-08)

mlp_model1 = mlp1.fit(x, y12)
mlp_model_pred1 = mlp_model1.predict(x_tst)

print("r2",r2_score(y_tst, mlp_model_pred1, multioutput = methodR2))
print(mean_squared_error(y_tst, mlp_model_pred1, multioutput = methodMSE))

# %% Neural Networks  (300,200,100,300,200,300,500,600) r2=0.9359

mlp2 = MLPRegressor(hidden_layer_sizes=(300,200,100,300,200,300,500,600),  activation='relu', solver='lbfgs', alpha=0.0002, batch_size='auto',
               learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
               random_state=0, tol=0.0001, verbose=False, warm_start=False, momentum=0.9,
               nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
               epsilon=1e-08)

mlp_model2 = mlp2.fit(x, y12)
mlp_model_pred2 = mlp_model2.predict(x_tst)

print("r2",r2_score(y_tst, mlp_model_pred2, multioutput = methodR2))
print(mean_squared_error(y_tst, mlp_model_pred2, multioutput = methodMSE))



# %% Neural Networks 300,200,100
#
#mlp2 = MLPRegressor(hidden_layer_sizes=(300,200,50,50),  activation='relu', solver='lbfgs', alpha=0.0002, batch_size='auto',
#               learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
#               random_state=0, tol=0.0001, verbose=False, warm_start=False, momentum=0.9,
#               nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
#               epsilon=1e-08)
#
#mlp_model2 = mlp2.fit(x, y12)
#mlp_model_pred2 = mlp_model2.predict(x_tst)
#
#print("r2",r2_score(y_tst, mlp_model_pred2, multioutput = methodR2))
#print(mean_squared_error(y_tst, mlp_model_pred2, multioutput = methodMSE))

# %%

#mlp1 = MultiOutputRegressor(MLPRegressor(hidden_layer_sizes=(10, 10, 10), max_iter=1000))
#mlp_model1 = mlp1.fit(x, y12)
#mlp_model_pred1 = mlp_model1.predict(x_tst)
#print("r2",r2_score(y_tst, mlp_model_pred1, multioutput = methodR2))


# %%

########## ENSEMBLE PREDICTIONS SPLIT TEST DATA #############################

#stack_preds = np.c_[pred_reg,lightg_pred,pred_treeU]
#meta_model = MultiOutputRegressor(LinearRegression())
#meta_model_preds = meta_model.fit(stack_preds, y_tst)
#print("r2",r2_score(y_tst, meta_model_preds, multioutput = methodR2))

#estimators = [('l', RidgeCV()), ('ee', LinearSVR(random_state=42))]
#
#reg = StackingRegressor(
#     estimators=estimators,
#     final_estimator=RandomForestRegressor(n_estimators=10,
#                                           random_state=42))
#
#reg.fit(x, y12).score(x_tst, y_tst)


#ensemble_pred = (pred_reg + lightg_pred +  pred_treecl) / 3
#ensemble_pred = 0.4*pred_reg + 0.2*lightg_pred +  0.4*pred_treeU 
#ensemble_pred = (pred_reg + lightg_pred + pred_tree + pred_tree1 + pred_treeU + pred_treecl) / 6
#print(mean_squared_error(y_tst, ensemble_pred, multioutput = methodMSE))

ensemble_pred = 0.3 *pred_reg   + 0.25*pred_treeU + 0.45* mlp_model_pred
print("r2",r2_score(y_tst, ensemble_pred, multioutput = methodR2))
print("r2",mean_squared_error(y_tst, ensemble_pred, multioutput = methodMSE))

ensemble_pred = (pred_reg   + pred_treeU + mlp_model_pred + mlp_model_pred1+mlp_model_pred2)/5
print("r2",r2_score(y_tst, ensemble_pred, multioutput = methodR2))


# Best 0.940158
ensemble_pred = 0.17  *pred_reg   + 0.22 *pred_treeU + 0.48* mlp_model_pred + 0.13*mlp_model_pred1
print("r2",r2_score(y_tst, ensemble_pred, multioutput = methodR2))
print("r2",mean_squared_error(y_tst, ensemble_pred, multioutput = methodMSE))


ensemble_pred =0.16  *pred_reg   + 0.2 * pred_treeU + 0.4 * mlp_model_pred + 0.24*mlp_model_pred2
print("r2",r2_score(y_tst, ensemble_pred, multioutput = methodR2))
print("r2",mean_squared_error(y_tst, ensemble_pred, multioutput = methodMSE))

# %% Stacking


# %% 

############################################################## STACKING

C = np.c_[preds_oof_treecl, preds_oof_tree, preds_oof_tree1, preds_oof_treeU, preds_oof_xgboost, preds_oof_lightg]
D = np.mean(pred_treecl, pred_tree, pred_tree1, pred_treeU, pred_reg, lightg_pred) # with the base models predict on x_tst whose labels are y_tst


params_meta_model = {'bootstrap': True,
 'max_depth': 120,
 'max_features': '0.7',
 'min_samples_leaf': 4,
 'min_samples_split': 2,
 'n_estimators': 1400}

meta_model = MultiOutputRegressor(ExtraTreesRegressor(**params_meta_model), n_jobs=-1)
meta_model.fit(C, y12)
meta_model_preds = meta_model.predict(D) 

#meta_model = MultiOutputRegressor(LinearRegression(fit_intercept = True), n_jobs=-1=-1)
#meta_model.fit(C, y12)
#meta_model_preds = meta_model.predict(D)

print("r2",r2_score(y_tst, meta_model_preds, multioutput = methodR2))

###############################################################################
###############################################################################


# %%
########## PREDICTIONS ON REAL TEST DATA ##################################

#pred_test_tree = classi.predict(x_test)
#pred_test_tree1 = classi1.predict(x_test)
#pred_test_treecl = classicl.predict(x_test)
pred_test_treeU = classiU.predict(x_test)
pred_test_reg =  xgb_reg_multioutput.predict(x_test)
#lightg_test_pred = lightg.predict(x_test)
mlp_model_test_pred = mlp_model.predict(x_test)
#mlp_model_test_pred1 = mlp_model1.predict(x_test)
mlp_model_test_pred2 = mlp_model2.predict(x_test)


#pred_test_ensemble=(pred_test_tree + pred_test_tree1 + pred_test_treeU + lightg_pred)/4
#pred_test_ensemble_1 = 0.4*pred_test_reg + 0.4*lightg_test_pred +  0.2*pred_test_treeU

#pred_test_ensemble_1 = 0.4*pred_test_reg + 0.2*lightg_test_pred +  0.4*pred_test_treeU # ayto xoris ta extra loss kai accs

#pred_test_ensemble_1 = 0.25*pred_test_reg + 0.25*lightg_test_pred  + 0.5*pred_test_treecl

## STACKING
#stack_preds = np.c_[pred_treeU,pred_reg,lightg_pred]
#stack_preds = np.c_[pred_treeU, pred_tree, pred_tree1, pred_reg, pred_treecl, lightg_pred]
#meta_model = MultiOutputRegressor(LinearRegression())
#meta_model_fit = meta_model.fit(stack_preds, y_tst)
#
#pred_test_ensemble_1 = meta_model.predict(x_test)

#pred_test_ensemble = (pred_test_reg + lightg_test_pred  + pred_test_treeU + mlp_model_pred)/4

# second best
pred_test_ensemble = 0.3 *pred_test_reg  + 0.25*pred_test_treeU + 0.45* mlp_model_test_pred

# best
pred_test_ensemble=0.17  *pred_test_reg   + 0.22 *pred_test_treeU + 0.48* mlp_model_test_pred + 0.13*mlp_model_test_pred1


pred_test_ensemble =0.16  *pred_test_reg   + 0.2 * pred_test_treeU + 0.4 * mlp_model_test_pred + 0.24*mlp_model_test_pred2



csv_preds=pred_test_ensemble.flatten() # final predictions

csv_preds[csv_preds<0]
csv_preds[csv_preds<0]=-csv_preds[csv_preds<0]




# %%
########### CREATE THE ROW NAMES FOR CSV FILE ###########################

row_names=[]
for i in range(476):
    row_names.append('test_'+str(i)+'_val_error')
    row_names.append('test_'+str(i)+'_train_error')

######### SAVE RESULTS IN CSV FORMAT ####################################

with open('preds_ensemble_data_cl.csv', mode='w+', newline='') as csv_file:
    fieldnames = ['id', 'Predicted']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    writer.writeheader()
    for i in range(len(csv_preds)):
        writer.writerow({'id': row_names[i], 'Predicted': csv_preds[i]})  
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        # %%
############# HYPER PARAMETER TUNING FOR RANDOM FOREST (AZURE)################
##############################################################################
#
############### RANDOM SEARCH ################
#
## Number of trees in random forest
#n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
### Number of features to consider at every split
#max_features = ['auto', 'sqrt']
### Maximum number of levels in tree
#max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
#max_depth.append(None)
### Minimum number of samples required to split a node
#min_samples_split = [2, 5, 10]
### Minimum number of samples required at each leaf node
#min_samples_leaf = [1, 2, 4]
### Method of selecting samples for training each tree
#bootstrap = [True, False]# Create the random grid
#random_grid = {'n_estimators': n_estimators,
#               'max_features': max_features,
#               'max_depth': max_depth,
#               'min_samples_split': min_samples_split,
#               'min_samples_leaf': min_samples_leaf,
#               'bootstrap': bootstrap}
#
#
### Use the random grid to search for best hyperparameters
### First create the base model to tune
#rf = RandomForestRegressor()
### Random search of parameters, using 3 fold cross validation, 
### search across 100 different combinations, and use all available cores
#rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=1, random_state=42, n_jobs = -1)# Fit the random search model
#rf_random.fit(x, y12)
##
#rf_random.best_params_
#
#bestRFparams={'bootstrap': True,
# 'max_depth': 100,
# 'max_features': 'auto',
# 'min_samples_leaf': 1,
# 'min_samples_split': 2,
# 'n_estimators': 1400}
#
#bestRFparams_cleaned={'bootstrap': True,
# 'max_depth': 142,
# 'max_features': 'auto',
# 'min_samples_leaf': 1,
# 'min_samples_split': 2,
# 'n_estimators': 1133}
#
########### GRID SEARCH ##########################
#
#param_grid = {
#    'bootstrap': [True],
#    'max_depth': [80, 100, 120],
#    'max_features': ['auto'],
#    'min_samples_leaf': [1, 2, 4],
#    'min_samples_split': [2, 4],
#    'n_estimators': [1200, 1400 , 1600]
#}
#
#rf = RandomForestRegressor()# Instantiate the grid search model
#grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
#                          cv = 3, n_jobs = -1, verbose = 2)
#grid_search.fit(x, y12)
#
#
#
#bestRFparams_random_grid = {'bootstrap': True,
# 'max_depth': 120,
# 'max_features': 'auto',
# 'min_samples_leaf': 1,
# 'min_samples_split': 2,
# 'n_estimators': 1400}
#
#
#
################################################################################
################################################################################
        
        
        
# %%

#def EnsembleMultiOutputKFold(number_splits, modelCV, x, y, shuffle):
#    R2score = []
#    number_models = len(modelCV)
#    cv = KFold(n_splits = number_splits, random_state = 42, shuffle = shuffle)
#    for i in range(number_models):
#        for train_index, test_index in cv.split(x.values):
#            #        print("Train Index: ", train_index, "\n")
#            #        print("Test Index: ", test_index)
#            X_train, X_test, Y_train, Y_test = x.values[train_index], x.values[test_index], y12[train_index], y12[test_index]
#            classifierCV = MultiOutputRegressor(modelCV[i], n_jobs=-1)
#            classiCV = classifierCV.fit(X_train, Y_train)
#            prediction = classiCV.predict(X_test)
#    R2score.append(r2_score(Y_test, prediction, multioutput='uniform_average'))
#        
#    return R2score
        
        
        
# %%  ##oop for generating many prediction from a single model with different parameters
      ## which can be used for stacking
    
#lightg_pred = 0
#N = 1
#for i in range(N):
#    lightg = MultiOutputRegressor(lightgbm.LGBMRegressor(random_state=i*101), n_jobs=-1)
#    lightg.fit(x, y12)
#    lightg_pred += lightg.predict(x_tst)
#    
#lightg_pred /= N