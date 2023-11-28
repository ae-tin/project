import argparse, glob, os, torch, warnings, time
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from pytorch_tabnet.tab_model import TabNetClassifier
from xgboost import XGBClassifier
from tqdm import tqdm
from sklearn.metrics import f1_score ,roc_auc_score, recall_score
import time
#from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

import lightgbm as lgb

parser = argparse.ArgumentParser(description = "ensemble_trainer")

## Training and evaluation path/lists, save path
parser.add_argument('--csv_path0', type=str, default='/Data2/folder_ti/data/1st_1011_train.csv', help='1st classification file')
parser.add_argument('--csv_path1', type=str, default='/Data2/folder_ti/data/1st_1011_test.csv', help='1st classification test file')
parser.add_argument('--csv_path2', type=str, default='/Data2/folder_ti/data/tr_2nd_class_after_bank_cluster5.csv', help='2nd classifition file')
parser.add_argument('--csv_path3', type=str, default='/Data2/folder_ti/data/te_2nd_class_after_bank_cluster5.csv', help='2nd classifition test file')
parser.add_argument('--csv_path4', type=str, default='/Data2/folder_ti/data/submission/submission.csv', help='submission file')
parser.add_argument('--save_path', type=str, default="exps/ensemble/ensemble(tab_xgb)_1st_2nd_grid_score.txt", help='Path to recording score')

## Model and Loss settings

print ('Available devices ', torch.cuda.device_count())
GPU_NUM = 4 # 원하는 GPU 번호 입력
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)
print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())

## Initialization
warnings.simplefilter("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')
args = parser.parse_args()

print('*'*50)
print('1st classication preprocessing,,,,,,')
print('*'*50)

train = pd.read_csv(args.csv_path0 ,index_col=0)
test = pd.read_csv(args.csv_path1 ,index_col=0)

def trans(x):
    if x == '노대출':
        return 0
    elif x == '대출':
        return 1
train['is_applied'] = train['is_applied'].apply(trans)

cat_col = ['purpose','income_type','rehabilitation','gender','employment_type','houseown_type']

cat_trans = dict()
for cat in cat_col :
    cat_trans[cat] = dict()
    type_ = list(set(train[cat].tolist()))
    for i, ty in enumerate(type_) :
        cat_trans[cat][ty] = i
        
cat_trans1 = dict()
for cat in cat_col :
    cat_trans1[cat] = dict()
    type_ = list(set(test[cat].tolist()))
    for i, ty in enumerate(type_) :
        cat_trans1[cat][ty] = i
        
        
for cat in cat_col :
    for k in tqdm(cat_trans[cat].keys()) :
        train[cat][train[cat]==k] = cat_trans[cat][k] 
        
for cat in cat_col :
    for k in tqdm(cat_trans1[cat].keys()) :
        test[cat][test[cat]==k] = cat_trans1[cat][k] 

types = train.dtypes
target = 'is_applied'
unused_col = ['set','user_id','application_id']
cat_col = ['purpose','income_type','rehabilitation','gender','employment_type','houseown_type']
categorical_dims =  {}
for col in train.columns:
    if col in cat_col :
        print(col, train[col].nunique())
        l_enc = LabelEncoder()
        train[col] = l_enc.fit_transform(train[col].values)
        categorical_dims[col] = len(l_enc.classes_)
    elif col not in unused_col+[target]:
        train.fillna(train.loc[:, col].mean(), inplace=True)


# Categorical Embedding을 위해 Categorical 변수의 차원과 idxs를 담음.
features = [ col for col in train.columns if col not in unused_col+[target]] 
cat_idxs = [ i for i, f in enumerate(features) if f in cat_col]
cat_dims = [ categorical_dims[f] for i, f in enumerate(features) if f in cat_col]


train = train.astype({'income_type':'float32','purpose':'float32','employment_type':'float32','houseown_type':'float32'})
test = test.astype({'income_type':'float32','purpose':'float32','employment_type':'float32','houseown_type':'float32'})

print('****************** preprocessing end! ******************')



score_file = open(args.save_path, "a+")



params_tab = {'n_d' : np.random.choice([4]),
         'n_steps' :np.random.choice([5]),        # tree에 depth
         'gamma': np.random.choice([1.5]),
         'momentum':np.random.choice([0.5]),
         'mask_type':np.random.choice(['sparsemax'])
         }

params_xgb = {'max_depth' : np.random.choice([6]),
              'subsample' : np.random.choice([.8]),
             'colsample_bytree' :np.random.choice([.8]),        # tree에 depth
             'n_estimators': np.random.choice([300])
             }

clf_xgb = XGBClassifier(**params_xgb,
                        learning_rate=0.2, 
                       objective='binary:logistic',
                       eval_metric = 'error@0.33',
                       tree_method='gpu_hist',
                       gpu_id=GPU_NUM)



clf_tab = TabNetClassifier(**params_tab, 
                       n_independent = 2, 
                       n_shared = 2, 
                       lambda_sparse = 1e-4, 
                       cat_idxs = cat_idxs, 
                       cat_dims = cat_dims,
                       cat_emb_dim = 1,
                       clip_value = 2,
                      optimizer_fn = torch.optim.Adam,
                       optimizer_params = {'lr':3e-2},
                       scheduler_params = {"gamma": 0.95,
                                         "step_size": 20},
                       scheduler_fn = torch.optim.lr_scheduler.StepLR,
                       epsilon = 1e-15,
                       device_name = device
                      )




X_train = train.loc[:,features].values
y_train = train.loc[:,target].values

X_test = test.loc[:,features].values
y_test = test.loc[:,target].values
                                                                                          

print('*'*20,' XGBoost 1st classification is stared! ','*'*20)
clf_xgb.fit(X_train, y_train)
print('*'*20,' XGBoost 1st classification is done! ','*'*20)


print('*'*20,' TabNet 1st classification is stared! ','*'*20)
clf_tab.fit(X_train=X_train, y_train=y_train,
    max_epochs=60, patience=20,
    batch_size=10240, virtual_batch_size=1280,
    num_workers=0,
    weights=1,
    drop_last=False
)



print('*'*20,' TabNet 1st classification is done! ','*'*20)


print('*'*20,' 1st classification is done! ','*'*20)


def thres_(threshold, array) :
    pred = np.ceil(array[:,1]-threshold)
    return pred
def thres_2(threshold, array) :
    pred = np.ceil(array-threshold)
    return pred
                                                       
                                
print('*'*50)
print('1st Classication Soft Voting Ensemble is Proceeding ,,,,,,,,')
print('*'*50)


pred_prob_valid_xgb = clf_xgb.predict_proba(X_test)

pred_prob_valid_tab = clf_tab.predict_proba(X_test)


def proba_weighted_mean_best_pred(thres,x,y) :
    wm = (thres*x[:,1] + (100-thres)*y[:,1])/100
                                
    f1_best_thres_valid_ = 0.36

    return thres_2(f1_best_thres_valid_,wm)                                
                                
                                

pre_valid_soft = proba_weighted_mean_best_pred(0.03*100, pred_prob_valid_tab, pred_prob_valid_xgb)

                                

print('*'*50)
print('1st Classication Soft Voting Ensemble is Done!')
print('*'*50)

train_1st = train.loc[:,['application_id']+features]
valid_1st = test.loc[:,['application_id']+features]

train_1st_id_set = list(set(train_1st.application_id.to_list()))
valid_1st_id_set = list(set(valid_1st.application_id.to_list()))

valid_1st['pred'] = pre_valid_soft

valid_1st_pred = valid_1st.loc[:,['application_id','pred']]

test_1st_id_set = list(set(valid_1st[valid_1st['pred']==1].application_id.to_list()))
test_1st_else_id_set = list(set(valid_1st[valid_1st['pred']==0].application_id.to_list()))
               
    
    
   
    
    
    
print('*'*50)
print('2nd Classication Data Prepare ,,,,,,,')
print('*'*50)


second = pd.read_csv(args.csv_path2)
second_test = pd.read_csv(args.csv_path3)
    
second.drop(['purpose','employment_type','houseown_type','user_id','gender','company_year','loanapply_insert_time','insert_time'],axis=1,inplace=True)  #
second = pd.merge(second,train.loc[:,['application_id','purpose','employment_type','houseown_type']],on='application_id',how='left')
    
    
second_test.drop(['purpose','employment_type','houseown_type','user_id','gender','company_year','loanapply_insert_time','insert_time'],axis=1,inplace=True)  #
second_test = pd.merge(second_test,test.loc[:,['application_id','purpose','employment_type','houseown_type']],on='application_id',how='left')

first_only_col = ['application_id','past.is_applied','m_past.is_applied','seq','min.loan_rate','product_n','n.limit_over_desire','loan_limit_min.rate',
                 'diff2','limit_over_desire_product','yd']

second = pd.merge(second,train.loc[:,first_only_col],on='application_id', how='left')
second_test = pd.merge(second_test,test.loc[:,first_only_col],on='application_id', how='left')


train_2nd = second[second['application_id'].isin(train_1st_id_set)]
valid_2nd = second_test[second_test['application_id'].isin(valid_1st_id_set)]       
    
test_2nd = second_test[second_test['application_id'].isin(test_1st_id_set)]
test_2nd_else = second_test[second_test['application_id'].isin(test_1st_else_id_set)]

assert valid_2nd.shape[0] == (test_2nd.shape[0]+test_2nd_else.shape[0]), '2nd Classification Valid vs Test,Test else 길이가 맞지 않습니다.'
    
    


cat_col = ['bank_id','purpose','income_type','employment_type','houseown_type','rehabilitation','rate_rank','bank_cluster','cluster_min_rate']


cat_trans = dict()
for cat in cat_col :
    cat_trans[cat] = dict()
    type_ = list(set(test_2nd[cat].tolist()))
    for i, ty in enumerate(type_) :
        cat_trans[cat][ty] = i
        

for cat in cat_col :
    for k in tqdm(cat_trans[cat].keys()) :
        train_2nd[cat][train_2nd[cat]==k] = cat_trans[cat][k]   
for cat in cat_col :
    for k in tqdm(cat_trans[cat].keys()) :
        valid_2nd[cat][valid_2nd[cat]==k] = cat_trans[cat][k]   
for cat in cat_col :
    for k in tqdm(cat_trans[cat].keys()) :
        test_2nd[cat][test_2nd[cat]==k] = cat_trans[cat][k]   
for cat in cat_col :
    for k in tqdm(cat_trans[cat].keys()) :
        test_2nd_else[cat][test_2nd_else[cat]==k] = cat_trans[cat][k] 



types = valid_2nd.dtypes
target = 'is_applied'
unused_col = ['set','application_id','pred','pred2','product_id']  
cat_col = ['bank_id','purpose','income_type','employment_type','houseown_type','rehabilitation','rate_rank','bank_cluster','cluster_min_rate']
categorical_dims =  {}
for col in valid_2nd.columns:
    if col in cat_col :
        print(col, valid_2nd[col].nunique())
        l_enc = LabelEncoder()
        valid_2nd[col] = l_enc.fit_transform(valid_2nd[col].values)
        categorical_dims[col] = len(l_enc.classes_)
    elif col not in unused_col+[target]:
        valid_2nd.fillna(valid_2nd.loc[:, col].mean(), inplace=True)


features = [ col for col in valid_2nd.columns if col not in unused_col+[target]] 
cat_idxs = [ i for i, f in enumerate(features) if f in cat_col]
cat_dims = [ categorical_dims[f] for i, f in enumerate(features) if f in cat_col]
train_2nd = train_2nd.astype({'income_type':'float32','purpose':'float32','employment_type':'float32','houseown_type':'float32'})
valid_2nd = valid_2nd.astype({'income_type':'float32','purpose':'float32','employment_type':'float32','houseown_type':'float32'})
test_2nd = test_2nd.astype({'income_type':'float32','purpose':'float32','employment_type':'float32','houseown_type':'float32'})
test_2nd_else = test_2nd_else.astype({'income_type':'float32','purpose':'float32','employment_type':'float32','houseown_type':'float32'})



X_train = train_2nd.loc[:,features].values
y_train = train_2nd.loc[:,target].values

X_valid = test_2nd.loc[:,features].values

X_valid_else = test_2nd_else.loc[:,features].values


print('*'*20,'2nd Classication Data Prepared!','*'*20)


print('*'*50)
print('2nd Classication Start!')
print('*'*50)

params_tab = {'n_d' : np.random.choice([4]),
         'n_steps' :np.random.choice([5]),      
         'gamma': np.random.choice([1.5]),
         'momentum':np.random.choice([0.5]),
         'mask_type':np.random.choice(['sparsemax'])
         }

params_xgb = {'max_depth' : np.random.choice([6]),
              'subsample' : np.random.choice([.8]),
             'colsample_bytree' :np.random.choice([.8]),  
             'n_estimators': np.random.choice([300])
             }

clf_xgb = XGBClassifier(**params_xgb,
                        learning_rate=0.2, 
                       objective='binary:logistic',
                       eval_metric = 'error@0.33',
                       tree_method='gpu_hist',
                       gpu_id=GPU_NUM)



clf_tab = TabNetClassifier(**params_tab, 
                       n_independent = 2, 
                       n_shared = 2, 
                       lambda_sparse = 1e-4, 
                       cat_idxs = cat_idxs, 
                       cat_dims = cat_dims,
                       cat_emb_dim = 1,
                       clip_value = 2,
                      optimizer_fn = torch.optim.Adam,
                       optimizer_params = {'lr':3e-2},
                       scheduler_params = {"gamma": 0.95,
                                         "step_size": 20},
                       scheduler_fn = torch.optim.lr_scheduler.StepLR,
                       epsilon = 1e-15,
                       device_name = device
                      )




print('*'*20,' XGBoost 2nd Classification is Stared ,,,,,,,,,, ','*'*20)
clf_xgb.fit(X_train, y_train)
print('*'*20,' XGBoost 2nd Classification is Done! ','*'*20)


print('*'*20,' TabNet 2nd Classification is Stared ,,,,,,,,,, ','*'*20)
clf_tab.fit(X_train=X_train, y_train=y_train,
    max_epochs=50, patience=20,
    batch_size=10240, virtual_batch_size=1280,
    num_workers=0,
    weights=1,
    drop_last=False
)
print('*'*20,' TabNet 2nd Classification is Done! ','*'*20)


print('*'*50)
print('2nd Classication Soft Voting Ensemble is Proceeding ,,,,,,,,')
print('*'*50)

pred_prob_valid_xgb = clf_xgb.predict_proba(X_valid)

pred_prob_valid_tab = clf_tab.predict_proba(X_valid)

pred_prob_valid_xgb_else = clf_xgb.predict_proba(X_valid_else)

pred_prob_valid_tab_else = clf_tab.predict_proba(X_valid_else)
    
def proba_weighted_mean_best_pred(thres,x,y) :
    wm = (thres*x[:,1] + (100-thres)*y[:,1])/100
                                
    f1_best_thres_valid_ = 0.46

    return thres_2(f1_best_thres_valid_,wm)                  

test_2nd['pred'] = proba_weighted_mean_best_pred(0.05*100, pred_prob_valid_tab, pred_prob_valid_xgb)


print('*'*50)
print('2nd Classication Soft Voting Ensemble is Done!')
print('*'*50)

print('*'*50)
print('Making Submission File ,,,,,,,,,')
print('*'*50)
    
test_2nd_else['pred'] = 0
final_2nd = pd.concat([test_2nd,test_2nd_else], ignore_index=True)

final_test = final_2nd.loc[:,['application_id','product_id','pred']]
submission_ = pd.read_csv(args.csv_path4)
submission = submission_.drop(['is_applied'],axis=1,inplace=False)
submission = pd.merge(submission,final_test,on=['application_id','product_id'],how='left')
submission.rename(columns={'pred':'is_applied'},inplace=True)
submission.to_csv('/Data2/folder_ti/data/submission/submission/데이터분석분야_퓨처스부문_빅월드콘팀_평가데이터.csv')
    

print('*'*50)
print('Making Submission File is Done!!!!')
print('Data Saved At /Data2/folder_ti/data/submission/submission/데이터분석분야_퓨처스부문_빅월드콘팀_평가데이터.csv')
print('*'*50)