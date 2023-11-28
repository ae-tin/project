import argparse, glob, os, torch, warnings, time
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from pytorch_tabnet.tab_model import TabNetClassifier
from tqdm import tqdm
from sklearn.metrics import f1_score ,roc_auc_score
import time
#from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

import lightgbm as lgb

def init_args(args):
    args.score_save_path    = os.path.join(args.save_path, 'score.txt')
    args.model_save_path    = os.path.join(args.save_path, 'model')
    os.makedirs(args.model_save_path, exist_ok = True)
    return args

parser = argparse.ArgumentParser(description = "Tabnet_trainer")

## Training and evaluation path/lists, save path
parser.add_argument('--csv_path',  type=str,   default='/Data2/folder_ti/data/classification1_train.csv', help='Path to save the score.txt and models')
parser.add_argument('--save_path',  type=str,   default="exps/first_classification/tabnet_grid_score.txt", help='Path to save the score.txt and models')

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
print('preprocessing,,,,,,')
print('*'*50)

train = pd.read_csv(args.csv_path ,index_col=0)
train = train.astype({'gender':'object'})

data_0 = train[train.is_applied == 0]
data_1 = train[train.is_applied == 1]

if "set" not in data_0.columns:
    data_0["set"] = np.random.choice([0,1,2,3,4], p =[.2, .2, .2, .2, .2], size=(data_0.shape[0],))
if "set" not in data_1.columns:
    data_1["set"] = np.random.choice([0,1,2,3,4], p =[.2, .2, .2, .2, .2], size=(data_1.shape[0],))

train = pd.concat([data_0,data_1], ignore_index=True)

cat_col = ['purpose','gender','income_type','employment_type','houseown_type']

cat_trans = dict()
for cat in cat_col :
    if cat == 'gender':
        continue
    cat_trans[cat] = dict()
    type_ = list(set(train[cat].tolist()))
    for i, ty in enumerate(type_) :
        cat_trans[cat][ty] = i

for cat in cat_col :
    if cat == 'gender' :
        continue
    for k in tqdm(cat_trans[cat].keys()) :
        train[cat][train[cat]==k] = cat_trans[cat][k]   

#train_indices = train[train.set=="train"].index
#valid_indices = train[train.set=="valid"].index
#test_indices = train[train.set=="test"].index

nunique = train.nunique()
types = train.dtypes
target = 'is_applied'
unused_col = ['application_id','last.loan_insert.time','user_id','set']
cat_col = ['purpose','gender','income_type','employment_type','houseown_type']
categorical_dims =  {}
for col in train.columns:
    if types[col] == 'object' or col in cat_col :
        if col in unused_col :
            continue
        print(col, train[col].nunique())
        l_enc = LabelEncoder()
        train[col] = l_enc.fit_transform(train[col].values)
        categorical_dims[col] = len(l_enc.classes_)
    else:
        train.fillna(train.loc[:, col].mean(), inplace=True)


# Categorical Embedding을 위해 Categorical 변수의 차원과 idxs를 담음.
features = [ col for col in train.columns if col not in unused_col+[target]] 
cat_idxs = [ i for i, f in enumerate(features) if f in cat_col]
cat_dims = [ categorical_dims[f] for i, f in enumerate(features) if f in cat_col]
#train_shuf = train_shuf.astype({'income_type':'int64','employment_type':'int64','houseown_type':'int64','purpose':'int64'})


print('****************** preprocessing end! ******************')

def thres_(threshold, array) :
    pred = np.ceil(array[:,1]-threshold)
    return pred


"""
        Defines main part of the TabNet network without the embedding layers.
        Parameters
        ----------
        input_dim : int
            Number of features
        output_dim : int or list of int for multi task classification
            Dimension of network output
            examples : one for regression, 2 for binary classification etc...
        n_d : int
            Dimension of the prediction  layer (usually between 4 and 64)
        n_a : int
            Dimension of the attention  layer (usually between 4 and 64)
        n_steps : int
            Number of successive steps in the network (usually between 3 and 10)
        gamma : float
            Float above 1, scaling factor for attention updates (usually between 1.0 to 2.0)
        n_independent : int
            Number of independent GLU layer in each GLU block (default 2)
        n_shared : int
            Number of independent GLU layer in each GLU block (default 2)
        epsilon : float
            Avoid log(0), this should be kept very low
        virtual_batch_size : int
            Batch size for Ghost Batch Normalization
        momentum : float
            Float value between 0 and 1 which will be used for momentum in all batch norm
        mask_type : str
            Either "sparsemax" or "entmax" : this is the masking function to use
        cat_idxs : list of int
            Index of each categorical column in the dataset
        cat_dims : list of int
            Number of categories in each categorical column
        cat_emb_dim : int or list of int
            Size of the embedding of categorical features
            if int, all categorical features will have same embedding size
            if list of int, every corresponding feature will have specific size
"""

score_file = open(args.save_path, "a+")



for j in tqdm(range(200)):

    params = {'n_d' : np.random.choice([4,16,64]),
              'n_a' : np.random.choice([4,16,64]),
             'n_steps' :np.random.choice([3,5,7,10]),        # tree에 depth
             'gamma': np.random.choice([1.3,1.5,1.7]),
             'momentum':np.random.choice([0.3,0.5]),
        #     'cat_emb_dim':np.random.choice([1,16,32,64]),
             'mask_type':np.random.choice(['sparsemax','entmax'])
             }

    clf = TabNetClassifier(**params, 
                           n_independent = 2, 
                           n_shared = 2, 
                           lambda_sparse = 1e-4, 
                           cat_idxs = cat_idxs, 
                           cat_dims = cat_dims,
                           cat_emb_dim = 1,
                           clip_value = 2,
                          optimizer_fn = torch.optim.Adam,
                           optimizer_params = {'lr':1e-1},
                           scheduler_params = {"gamma": 0.95,
                                             "step_size": 20},
                           scheduler_fn = torch.optim.lr_scheduler.StepLR,
                           epsilon = 1e-15,
                           device_name = device
                          )

    cv_score = []  
    cv_thres = []
    for i in range(2):

        train_indices = train[train.set!=i].index
        valid_indices = train[train.set==i].index

        X_train = train[features].values[train_indices]
        y_train = train[target].values[train_indices]

        X_valid = train[features].values[valid_indices]
        y_valid = train[target].values[valid_indices]


        clf.fit(X_train=X_train, y_train=y_train,eval_set=[(X_valid, y_valid)],
            max_epochs=25, patience=20,
            batch_size=10240, virtual_batch_size=1280,
            num_workers=0,
            weights=1,
            drop_last=False
        )

        pred_valid = clf.predict(X_valid)


        f1_valid = f1_score(y_valid, pred_valid) # y_true : 정답 값, y_pred : 예측 값 
        roc_auc = roc_auc_score(y_valid, pred_valid)

        print("Valid F1 score: {}".format(f1_valid))
        print("Valid ROC AUC Score: {}".format(roc_auc))

        pred_prob_valid = clf.predict_proba(X_valid)



        score_valid = [0]
        cv_best_thres_valid = 0
        cv_best_score = 0
        for i in range(1,100) :
            pre_valid = thres_(i/100,pred_prob_valid)

            f1_valid = f1_score(y_valid, pre_valid)

            score_valid.append(f1_valid)

        cv_best_thres_valid = score_valid.index(max(score_valid))/100
        cv_best_score = max(score_valid)
        cv_score.append(cv_best_score)
        cv_thres.append(cv_best_thres_valid)
    cv_mean_score = np.mean(cv_score)

    score_file.write("params : %s , cv_mean(best_f1) %2.2f , best_threshold : %s\n"%(params,cv_mean_score, cv_thres))
    score_file.flush()
    
               

