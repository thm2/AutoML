# -*- coding: utf-8 -*-



import numpy as np
import pandas as pd

import xgboost as xgb
from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

import csv

methodR2='uniform_average'


# %%

df = pd.read_csv('train.csv', sep = ',')
df_test = pd.read_csv('test.csv', sep = ',')

dff=df
col=dff.columns
dff=dff.drop([col[0], col[1],'batch_size_test','batch_size_val','criterion','optimizer','batch_size_train'], axis=1)


dff_test = df_test
col=dff_test.columns
dff_test=dff_test.drop([col[0], col[1],'batch_size_test','batch_size_val','criterion','optimizer','batch_size_train'], axis=1)


# %% EVERYTHING NORMALIZED

## TRAIN DATA PREPROCESSING #########################################


lbl = preprocessing.LabelEncoder()

dff.arch_and_hp = lbl.fit_transform(df['arch_and_hp'].astype(str))
dff.init_params_mu = lbl.fit_transform(df['init_params_mu'].astype(str))
dff.init_params_std = lbl.fit_transform(df['init_params_std'].astype(str))
dff.init_params_l2 = lbl.fit_transform(df['init_params_l2'].astype(str))



dff_test.arch_and_hp = lbl.fit_transform(df_test['arch_and_hp'].astype(str))
dff_test.init_params_mu = lbl.fit_transform(df_test['init_params_mu'].astype(str))
dff_test.init_params_std = lbl.fit_transform(df_test['init_params_std'].astype(str))
dff_test.init_params_l2 = lbl.fit_transform(df_test['init_params_l2'].astype(str))


#%% ######## ASSIGN OUTPUT VARIABLES AND PREDICTORS

y1=df.loc[:,'val_error']
y2=df.loc[:,'train_error']
y12 = np.c_[y1,y2]
x=dff.drop(['val_error','val_loss', 'train_error','train_loss'], axis=1)
#x, x_tst, y12, y_tst = train_test_split(x, y12, test_size=0.2, random_state=42) # uncomment for train/test split
x_test = dff_test



scaler = preprocessing.MinMaxScaler()
scaler.fit(x)

x = scaler.transform(x)
#x_tst = scaler.transform(x_tst)   # uncomment if using train/test split
x_test = scaler.transform(x_test)


# %%
################### RANDOM FOREST ###########################################

RFparams={'bootstrap': True,
 'max_features': 0.7,
 'min_samples_leaf': 4,
 'min_samples_split': 2,
 'n_estimators': 100}

forestU = RandomForestRegressor(**RFparams, random_state=1)
classifierU = MultiOutputRegressor(forestU, n_jobs=-1)
classiU=classifierU.fit(x, y12)

#pred_treeU=classiU.predict(x_tst)
#print("r2",r2_score(y_tst, pred_treeU, multioutput=methodR2))

# %%

##################################### XGBOOST #############################

param_reg={'booster':'gbtree','objective':'reg:squarederror',   'learning_rate':0.04, 'sample_type':'uniform', 
           'colsample_bytree':1, 'subsample':1, 'n_estimators':1000, 
           'reg_alpha':0.2, 'max_depth':4, 'gamma':0.0, 'alpha':0}

xg_reg = xgb.XGBRegressor(**param_reg)
xgb_reg_multioutput = MultiOutputRegressor(xg_reg).fit(x, y12)


#pred_reg = xgb_reg_multioutput.predict(x_tst)
#print("r2",r2_score(y_tst, pred_reg, multioutput = methodR2))


# %%

########################  Neural Networks  

mlp = MLPRegressor(hidden_layer_sizes=(300,200,100),  activation='relu', solver='lbfgs', alpha=0.0002, batch_size='auto',
               learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
               random_state=0, tol=0.0001, verbose=False, warm_start=False, momentum=0.9,
               nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
               epsilon=1e-08)

mlp_model = mlp.fit(x, y12)

#mlp_model_pred = mlp_model.predict(x_tst)
#print("r2",r2_score(y_tst, mlp_model_pred, multioutput = methodR2))


# %% 
####################  Neural Networks  

mlp2 = MLPRegressor(hidden_layer_sizes=(300,200,100,300,200,300,500,600),  activation='relu', solver='lbfgs', alpha=0.0002, batch_size='auto',
               learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
               random_state=0, tol=0.0001, verbose=False, warm_start=False, momentum=0.9,
               nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
               epsilon=1e-08)

mlp_model2 = mlp2.fit(x, y12)

#mlp_model_pred2 = mlp_model2.predict(x_tst)
#print("r2",r2_score(y_tst, mlp_model_pred2, multioutput = methodR2))

# %%
############################## ENSEMBLE for train/test split
 
#ensemble_pred =0.16  *pred_reg   + 0.2 * pred_treeU + 0.4 * mlp_model_pred + 0.24*mlp_model_pred2
#print("r2",r2_score(y_tst, ensemble_pred, multioutput = methodR2))

# %%
########## PREDICTIONS ON REAL TEST DATA ##################################

pred_test_treeU = classiU.predict(x_test)
pred_test_reg =  xgb_reg_multioutput.predict(x_test)
mlp_model_test_pred = mlp_model.predict(x_test)
mlp_model_test_pred2 = mlp_model2.predict(x_test)



pred_test_ensemble =0.16  *pred_test_reg   + 0.2 * pred_test_treeU + 0.4 * mlp_model_test_pred + 0.24*mlp_model_test_pred2
csv_preds=pred_test_ensemble.flatten() # final predictions

csv_preds[csv_preds<0]
csv_preds[csv_preds<0]=-csv_preds[csv_preds<0]




# %%
########### CREATE ROW NAMES FOR CSV  ###########################

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