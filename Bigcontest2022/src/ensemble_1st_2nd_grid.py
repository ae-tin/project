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
parser.add_argument('--csv_path',  type=str,   default='/Data2/folder_ti/data/1st_1011_train.csv', help='1st classification file')
parser.add_argument('--csv_path2',  type=str,   default='/Data2/folder_ti/data/tr_2nd_class_after_bank_cluster5.csv', help='2nd classifition file')
parser.add_argument('--save_path',  type=str,   default="exps/ensemble/ensemble(tab_xgb)_1st_2nd_grid_score_final4.txt", help='Path to recording score')

## Model and Loss settings

print ('Available devices ', torch.cuda.device_count())
GPU_NUM = 2 # 원하는 GPU 번호 입력
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

train = pd.read_csv(args.csv_path ,index_col=0)

def trans(x):
    if x == '노대출':
        return 0
    elif x == '대출':
        return 1
train['is_applied'] = train['is_applied'].apply(trans)

data_0 = train[train.is_applied == 0]
data_1 = train[train.is_applied == 1]

if "set" not in data_0.columns:
    data_0["set"] = np.random.choice([0,1,2,3,4], p =[.2, .2, .2, .2, .2], size=(data_0.shape[0],))
if "set" not in data_1.columns:
    data_1["set"] = np.random.choice([0,1,2,3,4], p =[.2, .2, .2, .2, .2], size=(data_1.shape[0],))

train = pd.concat([data_0,data_1], ignore_index=True)
train = train.sample(frac=1).reset_index(drop=True)

cat_col = ['purpose','income_type','rehabilitation','gender','employment_type','houseown_type']

cat_trans = dict()
for cat in cat_col :
    if cat in ['gender','rehabilitation']:
        continue
    cat_trans[cat] = dict()
    type_ = list(set(train[cat].tolist()))
    for i, ty in enumerate(type_) :
        cat_trans[cat][ty] = i

for cat in cat_col :
    if cat in ['gender','rehabilitation']:
        continue
    for k in tqdm(cat_trans[cat].keys()) :
        train[cat][train[cat]==k] = cat_trans[cat][k]   

#train_indices = train[train.set=="train"].index
#valid_indices = train[train.set=="valid"].index
#test_indices = train[train.set=="test"].index

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



train_indices = train[train.set!=i].index
valid_indices = train[train.set==i].index

X_train = train[features].values[train_indices]
y_train = train[target].values[train_indices]

X_valid = train[features].values[valid_indices]
y_valid = train[target].values[valid_indices]


print('*'*20,' XGBoost 1st classification is stared! ','*'*20)
clf_xgb.fit(X_train, y_train)
print('*'*20,' XGBoost 1st classification is done! ','*'*20)


print('*'*20,' TabNet 1st classification is stared! ','*'*20)
clf_tab.fit(X_train=X_train, y_train=y_train,eval_set=[(X_valid, y_valid)],
    max_epochs=60, patience=20,
    batch_size=10240, virtual_batch_size=1280,
    num_workers=0,
    weights=1,
    drop_last=False
)
pred_valid_xgb = clf_xgb.predict(X_valid)
pred_prob_valid_xgb = clf_xgb.predict_proba(X_valid)

pred_valid_tab = clf_tab.predict(X_valid)
pred_prob_valid_tab = clf_tab.predict_proba(X_valid)

print('*'*20,' TabNet 1st classification is done! ','*'*20)


print('*'*20,' 1st classification is done! ','*'*20)



#f1_valid = f1_score(y_valid, pred_valid) # y_true : 정답 값, y_pred : 예측 값 
#roc_auc = roc_auc_score(y_valid, pred_valid)

first_mode = ['hard','soft']

def thres_(threshold, array) :
    pred = np.ceil(array[:,1]-threshold)
    return pred
def thres_2(threshold, array) :
    pred = np.ceil(array-threshold)
    return pred

f1_score_valid_tab = []
f1_score_valid_xgb = []

print('*'*50)
print('1st Classication Hard Voting Ensemble is Proceeding,,,,,,')
print('*'*50)

#### determine best threshold to decide 0 or 1
for i in tqdm(range(1,100)) :
    pre_valid_tab = thres_(i/100,pred_prob_valid_tab)
    pre_valid_xgb = thres_(i/100,pred_prob_valid_xgb)

    f1_valid_tab = f1_score(y_valid, pre_valid_tab)
    f1_valid_xgb = f1_score(y_valid, pre_valid_xgb)
    
    f1_score_valid_tab.append(f1_valid_tab)
    f1_score_valid_xgb.append(f1_valid_xgb)


f1_best_thres_valid_tab = f1_score_valid_tab.index(max(f1_score_valid_tab))/100
f1_best_thres_valid_xgb = f1_score_valid_xgb.index(max(f1_score_valid_xgb))/100

f1_best_score_tab = max(f1_score_valid_tab)
f1_best_score_xgb = max(f1_score_valid_xgb)

recall_score_tab = recall_score(y_valid, thres_(f1_best_thres_valid_tab,pred_prob_valid_tab))
recall_score_xgb = recall_score(y_valid, thres_(f1_best_thres_valid_xgb,pred_prob_valid_xgb))

score_file.write("TabNet : 1st classificaiton result, f1_score_best : %2.3f , f1_thres_best : %2.3f, recall_score : %2.3f \n"%(f1_best_score_tab, f1_best_thres_valid_tab,recall_score_tab))
score_file.flush()
score_file.write("XGBoost : 1st classificaiton result, f1_score_best : %2.3f , f1_thres_best : %2.3f, recall_score : %2.3f \n"%(f1_best_score_xgb, f1_best_thres_valid_xgb, recall_score_xgb))
score_file.flush()


tab_f1_pred = thres_(f1_best_thres_valid_tab,pred_prob_valid_tab)
xgb_f1_pred = thres_(f1_best_thres_valid_xgb,pred_prob_valid_xgb)


def weighted_mean_best_pred(thres,x,y) :
    wm = (thres*x + (100-thres)*y)/100
                                
    f1_score_valid_ = []

    for i in range(1,100) :
        pre = thres_2(i/100,wm)

        f1_valid_ = f1_score(y_valid, pre)

        f1_score_valid_.append(f1_valid_)

    f1_best_thres_valid_ = f1_score_valid_.index(max(f1_score_valid_))/100

    f1_best_score_ = max(f1_score_valid_)

    return thres_2(f1_best_thres_valid_,wm) , f1_best_thres_valid_

                                
                                
f1_score_hard = []
b_thr = []

#### determine best threshold to decide 0 or 1
for i in tqdm(range(5,100,5)) :
    pre_valid_f1, thr = weighted_mean_best_pred(i, tab_f1_pred, xgb_f1_pred)

    f1_f1 = f1_score(y_valid, pre_valid_f1)
    
    f1_score_hard.append(f1_f1)
    b_thr.append(thr)

f1_best_thres_f1 = f1_score_hard.index(max(f1_score_hard))/100

f1_best_score_f1 = max(f1_score_hard)

f1_best_thres_in_thres = b_thr[f1_score_hard.index(max(f1_score_hard))]

recall_score_hard = recall_score(y_valid, weighted_mean_best_pred(f1_best_thres_f1*100, tab_f1_pred, xgb_f1_pred)[0])

score_file.write("Hard Voting Ensemble F1 Score for 1st Classification --- f1_score_best(tab_f1,xgb_f1) : %2.3f , weight_thres_best : %2.3f, thres_in_thres : %2.3f, recall_score : %2.3f \n"%(f1_best_score_f1, f1_best_thres_f1,f1_best_thres_in_thres,recall_score_hard))
score_file.flush()


print('*'*50)
print('1st Classication Hard Voting Ensemble is Done!')
print('*'*50)


print('*'*50)
print('1st Classication Soft Voting Ensemble is Proceeding ,,,,,,,,')
print('*'*50)


pred_prob_valid_xgb = clf_xgb.predict_proba(X_valid)

pred_prob_valid_tab = clf_tab.predict_proba(X_valid)


def proba_weighted_mean_best_pred(thres,x,y) :
    wm = (thres*x[:,1] + (100-thres)*y[:,1])/100
                                
    f1_score_valid_ = []

    for i in range(1,100) :
        pre = thres_2(i/100,wm)

        f1_valid_ = f1_score(y_valid, pre)

        f1_score_valid_.append(f1_valid_)

    f1_best_thres_valid_ = f1_score_valid_.index(max(f1_score_valid_))/100

    f1_best_score_ = max(f1_score_valid_)

    return thres_2(f1_best_thres_valid_,wm) ,f1_best_thres_valid_                               
                                
                                
                                
f1_score_soft = []
b_thr = []

for i in tqdm(range(5,100,5)) :

    pre_valid_soft, thr = proba_weighted_mean_best_pred(i, pred_prob_valid_tab, pred_prob_valid_xgb)

    f1 = f1_score(y_valid, pre_valid_soft)
    
    f1_score_soft.append(f1)
    b_thr.append(thr)

f1_best_thres_soft = f1_score_soft.index(max(f1_score_soft))/100

f1_best_score_soft = max(f1_score_soft)

f1_best_thres_in_thres = b_thr[f1_score_soft.index(max(f1_score_soft))]
                                
recall_score_soft = recall_score(y_valid, proba_weighted_mean_best_pred(f1_best_thres_soft*100, pred_prob_valid_tab, pred_prob_valid_xgb)[0])


score_file.write("Soft Voting Ensemble F1 Score for 1st Classification --- f1_score_best(tab_prob,xgb_prob) : %2.3f , weight_thres_best : %2.3f, thres_in_thres : %2.3f, recall_score : %2.3f \n"%(f1_best_score_soft, f1_best_thres_soft,f1_best_thres_in_thres,recall_score_soft))
score_file.flush()


print('*'*50)
print('1st Classication Soft Voting Ensemble is Done!')
print('*'*50)

    
what_f1 = [f1_best_score_f1,f1_best_score_soft]
hard_or_soft_f1 = what_f1.index(max(what_f1))
diff_score = abs(f1_best_score_f1-f1_best_score_soft)

what_recall = [recall_score_hard,recall_score_soft]
hard_or_soft_recall = what_recall.index(max(what_recall))

if diff_score < 0.02 :
    hard_or_soft = hard_or_soft_recall
    if hard_or_soft == 0 :
        print('*'*50,'hard_has_better_recall','*'*50)
    else :
        print('*'*50,'soft_has_better_recall','*'*50)
else :
    hard_or_soft = hard_or_soft_f1


train_1st = train.loc[train_indices,['application_id']+features]
valid_1st = train.loc[valid_indices,['application_id']+features]

train_1st_id_set = list(set(train_1st.application_id.to_list()))
valid_1st_id_set = list(set(valid_1st.application_id.to_list()))

if hard_or_soft == 0 :
    valid_1st['pred'] = weighted_mean_best_pred(f1_best_thres_f1*100, tab_f1_pred, xgb_f1_pred)[0]
elif hard_or_soft == 1 :
    valid_1st['pred'] = proba_weighted_mean_best_pred(f1_best_thres_soft*100, pred_prob_valid_tab, pred_prob_valid_xgb)[0]

valid_1st_pred = valid_1st.loc[:,['application_id','pred']]

test_1st_id_set = list(set(valid_1st[valid_1st['pred']==1].application_id.to_list()))
test_1st_else_id_set = list(set(valid_1st[valid_1st['pred']==0].application_id.to_list()))
               
    
    
    
    
    
    
    
    
'''
print ('Available devices ', torch.cuda.device_count())
GPU_NUM = 3 # 원하는 GPU 번호 입력
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)
print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
'''    
    
    
    
    
    
    
    
    
    
    
    
print('*'*50)
print('2nd Classication Data Prepare ,,,,,,,')
print('*'*50)


second = pd.read_csv(args.csv_path2)
second.drop(['purpose','employment_type','houseown_type','user_id','gender','company_year','loanapply_insert_time','insert_time','product_id'],axis=1,inplace=True)  #
second = pd.merge(second,train.loc[:,['application_id','purpose','employment_type','houseown_type']],on='application_id',how='left')

first_only_col = ['application_id','past.is_applied','m_past.is_applied','seq','min.loan_rate','product_n','n.limit_over_desire','loan_limit_min.rate',
                 'diff2','limit_over_desire_product','yd']

second = pd.merge(second,train.loc[:,first_only_col],on='application_id', how='left')


train_2nd = second[second['application_id'].isin(train_1st_id_set)]
valid_2nd = second[second['application_id'].isin(valid_1st_id_set)]
test_2nd = second[second['application_id'].isin(test_1st_id_set)]
test_2nd_else = second[second['application_id'].isin(test_1st_else_id_set)]

assert second.shape[0] == (train_2nd.shape[0]+valid_2nd.shape[0]), '2nd Classification All Data vs Train, Valid 길이가 맞지 않습니다.'
assert valid_2nd.shape[0] == (test_2nd.shape[0]+test_2nd_else.shape[0]), '2nd Classification Valid vs Test,Test else 길이가 맞지 않습니다.'


cat_col = ['bank_id','purpose','income_type','employment_type','houseown_type','rehabilitation','rate_rank','bank_cluster','cluster_min_rate']


cat_trans = dict()
for cat in cat_col :
    cat_trans[cat] = dict()
    type_ = list(set(train_2nd[cat].tolist()))
    for i, ty in enumerate(type_) :
        cat_trans[cat][ty] = i
cat_trans1 = dict()
for cat in cat_col :
    cat_trans1[cat] = dict()
    type_ = list(set(valid_2nd[cat].tolist()))
    for i, ty in enumerate(type_) :
        cat_trans1[cat][ty] = i
cat_trans2 = dict()
for cat in cat_col :
    cat_trans2[cat] = dict()
    type_ = list(set(test_2nd[cat].tolist()))
    for i, ty in enumerate(type_) :
        cat_trans2[cat][ty] = i        
cat_trans3 = dict()
for cat in cat_col :
    cat_trans3[cat] = dict()
    type_ = list(set(test_2nd_else[cat].tolist()))
    for i, ty in enumerate(type_) :
        cat_trans3[cat][ty] = i
        
        
        
for cat in cat_col :
    for k in tqdm(cat_trans[cat].keys()) :
        train_2nd[cat][train_2nd[cat]==k] = cat_trans[cat][k]   
for cat in cat_col :
    for k in tqdm(cat_trans1[cat].keys()) :
        valid_2nd[cat][valid_2nd[cat]==k] = cat_trans1[cat][k]   
for cat in cat_col :
    for k in tqdm(cat_trans2[cat].keys()) :
        test_2nd[cat][test_2nd[cat]==k] = cat_trans2[cat][k]   
for cat in cat_col :
    for k in tqdm(cat_trans3[cat].keys()) :
        test_2nd_else[cat][test_2nd_else[cat]==k] = cat_trans3[cat][k] 



types = train_2nd.dtypes
target = 'is_applied'
#unused_col = ['set','user_id','application_id','loanapply_insert_time','insert_time','gender','yearly_income',
#             'company_year','existing_loan_cnt','product_id','pred']
unused_col = ['set','application_id','pred','pred2']  # ,'loanapply_insert_time','insert_time','product_id'
cat_col = ['bank_id','purpose','income_type','employment_type','houseown_type','rehabilitation','rate_rank','bank_cluster','cluster_min_rate']
categorical_dims =  {}
for col in train_2nd.columns:
    if col in cat_col :
        print(col, train_2nd[col].nunique())
        l_enc = LabelEncoder()
        train_2nd[col] = l_enc.fit_transform(train_2nd[col].values)
        categorical_dims[col] = len(l_enc.classes_)
    elif col not in unused_col+[target]:
        train_2nd.fillna(train_2nd.loc[:, col].mean(), inplace=True)


# Categorical Embedding을 위해 Categorical 변수의 차원과 idxs를 담음.
features = [ col for col in valid_2nd.columns if col not in unused_col+[target]] 
cat_idxs = [ i for i, f in enumerate(features) if f in cat_col]
cat_dims = [ categorical_dims[f] for i, f in enumerate(features) if f in cat_col]
#train_shuf = train_shuf.astype({'income_type':'int64','employment_type':'int64','houseown_type':'int64','purpose':'int64'})

train_2nd = train_2nd.astype({'income_type':'float32','purpose':'float32','employment_type':'float32','houseown_type':'float32'})
valid_2nd = valid_2nd.astype({'income_type':'float32','purpose':'float32','employment_type':'float32','houseown_type':'float32'})
test_2nd = test_2nd.astype({'income_type':'float32','purpose':'float32','employment_type':'float32','houseown_type':'float32'})
test_2nd_else = test_2nd_else.astype({'income_type':'float32','purpose':'float32','employment_type':'float32','houseown_type':'float32'})



X_train = train_2nd.loc[:,features].values
y_train = train_2nd.loc[:,target].values

X_valid = test_2nd.loc[:,features].values
y_valid = test_2nd.loc[:,target].values

X_valid_else = test_2nd_else.loc[:,features].values
y_valid_else = test_2nd_else.loc[:,target].values


print('*'*20,'2nd Classication Data Prepared!','*'*20)


print('*'*50)
print('2nd Classication Start!')
print('*'*50)

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




print('*'*20,' XGBoost 2nd Classification is Stared ,,,,,,,,,, ','*'*20)
clf_xgb.fit(X_train, y_train)
print('*'*20,' XGBoost 2nd Classification is Done! ','*'*20)


print('*'*20,' TabNet 2nd Classification is Stared ,,,,,,,,,, ','*'*20)
clf_tab.fit(X_train=X_train, y_train=y_train,eval_set=[(X_valid, y_valid)],
    max_epochs=50, patience=20,
    batch_size=10240, virtual_batch_size=1280,
    num_workers=0,
    weights=1,
    drop_last=False
)
print('*'*20,' TabNet 2nd Classification is Done! ','*'*20)

pred_prob_valid_xgb = clf_xgb.predict_proba(X_valid)

pred_prob_valid_tab = clf_tab.predict_proba(X_valid)

pred_prob_valid_xgb_else = clf_xgb.predict_proba(X_valid_else)

pred_prob_valid_tab_else = clf_tab.predict_proba(X_valid_else)







f1_score_valid_tab = []
f1_score_valid_xgb = []

print('*'*50)
print('2nd Classication Hard Voting Ensemble is Proceeding,,,,,,')
print('*'*50)

#### determine best threshold to decide 0 or 1
for i in tqdm(range(1,100)) :
    pre_valid_tab = thres_(i/100,pred_prob_valid_tab)
    pre_valid_xgb = thres_(i/100,pred_prob_valid_xgb)

    f1_valid_tab = f1_score(y_valid, pre_valid_tab)
    f1_valid_xgb = f1_score(y_valid, pre_valid_xgb)
    
    f1_score_valid_tab.append(f1_valid_tab)
    f1_score_valid_xgb.append(f1_valid_xgb)


f1_best_thres_valid_tab = f1_score_valid_tab.index(max(f1_score_valid_tab))/100
f1_best_thres_valid_xgb = f1_score_valid_xgb.index(max(f1_score_valid_xgb))/100

f1_best_score_tab = max(f1_score_valid_tab)
f1_best_score_xgb = max(f1_score_valid_xgb)

recall_score_tab = recall_score(y_valid, thres_(f1_best_thres_valid_tab,pred_prob_valid_tab))
recall_score_xgb = recall_score(y_valid, thres_(f1_best_thres_valid_xgb,pred_prob_valid_xgb))

score_file.write("TabNet : 2nd classificaiton result, f1_score_best : %2.3f , f1_thres_best : %2.3f, recall_score : %2.3f \n"%(f1_best_score_tab, f1_best_thres_valid_tab,recall_score_tab))
score_file.flush()
score_file.write("XGBoost : 2nd classificaiton result, f1_score_best : %2.3f , f1_thres_best : %2.3f, recall_score : %2.3f \n"%(f1_best_score_xgb, f1_best_thres_valid_xgb, recall_score_xgb))
score_file.flush()


tab_f1_pred = thres_(f1_best_thres_valid_tab,pred_prob_valid_tab)
xgb_f1_pred = thres_(f1_best_thres_valid_xgb,pred_prob_valid_xgb)

tab_f1_pred_else = thres_(f1_best_thres_valid_tab,pred_prob_valid_tab_else)
xgb_f1_pred_else = thres_(f1_best_thres_valid_xgb,pred_prob_valid_xgb_else)


def weighted_mean_best_pred(thres,x,y) :
    assert len(x)==len(y), 'data 길이가 맞지 않습니다.'
    wm = (thres*x + (100-thres)*y)/100
                                
    f1_score_valid_ = []

    for i in range(1,100) :
        pre = thres_2(i/100,wm)
        
        f1_valid_ = f1_score(y_valid, pre)

        f1_score_valid_.append(f1_valid_)

    f1_best_thres_valid_ = f1_score_valid_.index(max(f1_score_valid_))/100

    f1_best_score_ = max(f1_score_valid_)

    return thres_2(f1_best_thres_valid_,wm), f1_best_thres_valid_

def weighted_mean_best_pred_else(thres,x,y) :
    assert len(x)==len(y), 'data 길이가 맞지 않습니다.'
    wm = (thres*x + (100-thres)*y)/100
                                
    f1_score_valid_ = []

    for i in range(1,100) :
        pre = thres_2(i/100,wm)
        
        f1_valid_ = f1_score(y_valid_else, pre)

        f1_score_valid_.append(f1_valid_)

    f1_best_thres_valid_ = f1_score_valid_.index(max(f1_score_valid_))/100

    f1_best_score_ = max(f1_score_valid_)

    return thres_2(f1_best_thres_valid_,wm)

                                
                                
f1_score_hard = []
b_thr = []

#### determine best threshold to decide 0 or 1
for i in tqdm(range(5,100,5)) :
    pre_valid_f1, thr = weighted_mean_best_pred(i, tab_f1_pred, xgb_f1_pred)

    f1_f1 = f1_score(y_valid, pre_valid_f1)
    
    f1_score_hard.append(f1_f1)
    b_thr.append(thr)

f1_best_thres_f1 = f1_score_hard.index(max(f1_score_hard))/100

f1_best_score_f1 = max(f1_score_hard)

f1_best_thres_in_thres = b_thr[f1_score_hard.index(max(f1_score_hard))]

recall_score_hard = recall_score(y_valid, weighted_mean_best_pred(f1_best_thres_f1*100, tab_f1_pred, xgb_f1_pred)[0])

score_file.write("Hard Voting Ensemble F1 Score for 2nd Classification --- f1_score_best(tab_f1,xgb_f1) : %2.3f , weight_thres_best : %2.3f, thres_in_thres : %2.3f, recall_score : %2.3f \n"%(f1_best_score_f1, f1_best_thres_f1,f1_best_thres_in_thres,recall_score_hard))
score_file.flush()


print('*'*50)
print('2nd Classication Hard Voting Ensemble is Done!')
print('*'*50)


print('*'*50)
print('2nd Classication Soft Voting Ensemble is Proceeding ,,,,,,,,')
print('*'*50)


def proba_weighted_mean_best_pred(thres,x,y) :
    wm = (thres*x[:,1] + (100-thres)*y[:,1])/100
                                
    f1_score_valid_ = []

    for i in range(1,100) :
        pre = thres_2(i/100,wm)

        f1_valid_ = f1_score(y_valid, pre)

        f1_score_valid_.append(f1_valid_)

    f1_best_thres_valid_ = f1_score_valid_.index(max(f1_score_valid_))/100

    f1_best_score_ = max(f1_score_valid_)

    return thres_2(f1_best_thres_valid_,wm), f1_best_thres_valid_                            
                                
def proba_weighted_mean_best_pred_else(thres,x,y) :
    wm = (thres*x[:,1] + (100-thres)*y[:,1])/100
                                
    f1_score_valid_ = []

    for i in range(1,100) :
        pre = thres_2(i/100,wm)

        f1_valid_ = f1_score(y_valid_else, pre)

        f1_score_valid_.append(f1_valid_)

    f1_best_thres_valid_ = f1_score_valid_.index(max(f1_score_valid_))/100

    f1_best_score_ = max(f1_score_valid_)

    return thres_2(f1_best_thres_valid_,wm)     
                                
f1_score_soft = []
b_thr = []

for i in tqdm(range(5,100,5)) :

    pre_valid_soft, thr = proba_weighted_mean_best_pred(i, pred_prob_valid_tab, pred_prob_valid_xgb)

    f1 = f1_score(y_valid, pre_valid_soft)
    
    f1_score_soft.append(f1)
    b_thr.append(thr)

f1_best_thres_soft = f1_score_soft.index(max(f1_score_soft))/100

f1_best_score_soft = max(f1_score_soft)

f1_best_thres_in_thres = b_thr[f1_score_soft.index(max(f1_score_soft))]
                                
recall_score_soft = recall_score(y_valid, proba_weighted_mean_best_pred(f1_best_thres_soft*100, pred_prob_valid_tab, pred_prob_valid_xgb)[0])


score_file.write("Soft Voting Ensemble F1 Score for 2nd Classification --- f1_score_best(tab_prob,xgb_prob) : %2.3f , weight_thres_best : %2.3f, thres_in_thres : %2.3f, recall_score : %2.3f \n"%(f1_best_score_soft, f1_best_thres_soft,f1_best_thres_in_thres,recall_score_soft))
score_file.flush()


print('*'*50)
print('2nd Classication Soft Voting Ensemble is Done!')
print('*'*50)

what_f1 = [f1_best_score_f1,f1_best_score_soft]
hard_or_soft_f1 = what_f1.index(max(what_f1))
diff_score = abs(f1_best_score_f1-f1_best_score_soft)

what_recall = [recall_score_hard,recall_score_soft]
hard_or_soft_recall = what_recall.index(max(what_recall))

if diff_score < 0.02 :
    hard_or_soft = hard_or_soft_recall
    if hard_or_soft == 0 :
        print('*'*50,'Hard_has_better_recall','*'*50)
    else :
        print('*'*50,'Soft_has_better_recall','*'*50)
else :
    hard_or_soft = hard_or_soft_f1
    print('*'*50,'There is some difference in Hard and Soft','*'*50)
    
if hard_or_soft == 0 :
    test_2nd['pred'] = weighted_mean_best_pred(f1_best_thres_f1*100, tab_f1_pred, xgb_f1_pred)[0]
elif hard_or_soft == 1 :
    test_2nd['pred'] = proba_weighted_mean_best_pred(f1_best_thres_soft*100, pred_prob_valid_tab, pred_prob_valid_xgb)[0]



    
####### 2nd only  ########
if hard_or_soft == 0 :
    test_2nd_else['pred'] = weighted_mean_best_pred_else(f1_best_thres_f1*100, tab_f1_pred_else, xgb_f1_pred_else)
elif hard_or_soft == 1 :
    test_2nd_else['pred'] = proba_weighted_mean_best_pred_else(f1_best_thres_soft*100, pred_prob_valid_tab_else, pred_prob_valid_xgb_else)

final_2nd = pd.concat([test_2nd,test_2nd_else], ignore_index=True)

second_only = f1_score(final_2nd.is_applied.values, final_2nd.pred.values)


####### 1st & 2nd  ########

test_2nd_else['pred'] = 0
final_2nd = pd.concat([test_2nd,test_2nd_else], ignore_index=True)

first_second = f1_score(final_2nd.is_applied.values, final_2nd.pred.values)


print('2nd_only_2nd_classification_f1_score : ',second_only)

## 1,2 차 모두

print('1st_2nd_both_2nd_classification_f1_score : ',first_second)
    



score_file.write("Final 2nd Classification --- 2nd Phase only F1 Score : %2.3f , 1st & 2nd Phase F1 Score : %2.3f \n"%(second_only, first_second))
score_file.flush()







