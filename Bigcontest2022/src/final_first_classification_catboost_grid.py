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
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

import lightgbm as lgb

from catboost import CatBoostClassifier


def init_args(args):
    args.score_save_path    = os.path.join(args.save_path, 'score.txt')
    args.model_save_path    = os.path.join(args.save_path, 'model')
    os.makedirs(args.model_save_path, exist_ok = True)
    return args

parser = argparse.ArgumentParser(description = "CatBoost_trainer")

## Training and evaluation path/lists, save path
parser.add_argument('--csv_path',  type=str,   default='/Data2/folder_ti/data/1st_1011_train.csv', help='Path to save the score.txt and models')
parser.add_argument('--save_path',  type=str,   default="exps/first_classification/final_catboost_grid_score.txt", help='Path to save the score.txt and models')

## Model and Loss settings

print ('Available devices ', torch.cuda.device_count())
GPU_NUM = 1 # 원하는 GPU 번호 입력
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
train = train.astype({'gender':'object','rehabilitation':'object'})
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

train2 = train
train2 = pd.get_dummies(train2, columns = cat_col )
features2 = [ col for col in train2.columns if col not in unused_col+[target]]


print('****************** preprocessing end! ******************')

def thres_(threshold, array) :
    pred = np.ceil(array[:,1]-threshold)
    return pred


"""
        Defines main part of the TabNet network without the embedding layers.
        Parameters
        ----------
       
        learning_rate : int
            (usually between 0.01 ~ 0.2)
     #   max_depth : int  default = 6
            if N, 2^N leaf node generated
        min_child_weight : int    default = 1
            Number of successive steps in the network (usually between 3 and 10)
            
     #   subsample은 data자체를 샘플링하는 것이고 colsample_* 파라미터는 feature를 샘플링하는 것이다.
        colsample_* 파라미터는 feature가 너무 많거나 소수의 feature에 지나치게 의존적일 때 사용하면 좋음.
        
        
     #   colsample_bytree : (0,1]
            각각의 트리(스탭)마다 사용할 칼럼(Feature)의 비율 
        colsample_bylevel :  (0,1]
            각각의 트리 depth 마다 사용할 칼럼(Feature)의 비율
        colsample_bynode : (0,1]
            각각의 노드 depth 마다 사용할 칼럼(Feature)의 비율
        n_estimators : int
            maximum number of decision tree; high value can be lead to overfitting
            
        object : reg:squarederror / reg:squaredlogerror ( 오차제곱, 오차로그제곱 ). 
                binary:logistic / multi:softmax / multi:softprob / rank:pairwise / rank:ndcg / rank:map
                
        base_score : y 범위의 중간?
            초기 편향치(bias)이다
        eval_metric : rmse / rmsle / mae / error / error@t / merror(다항분류버전) / auc / aucpr 
            error@t : 이항 분류(binary class)에서 error는 0.5 이상을 1 미만을 0이라고 판단하고 error@t는 t 이상을 1 미만을 0이라고 판단한다.
            aucpr : auc뒤에 붙은 pr은 preicison recall을 뜻한다. f-score나 precision, recall에 민감할 때 사용한다.(error나 rmse가 더 잘 나오는 경우도 많으니 다 해봐야 함.)
            
    #    num_boost_round : [0,inf)
            이건 epoch랑 같은 거인 듯 
            몇 회의 step을 반복? 너무 높은 값을 설정하면 오버 피팅이 생기고 모델 사이즈가 커진다. 
            
        early_stopping_rounds : [0,inf)
            조기 종료 조건이다.
            eval_metric이 결과가 early_stopping_rounds 횟수 동안 개선되지 않으면 num_boost_round에 도달하기 전에 종료한다.
            
        
"""

score_file = open(args.save_path, "a+")



for j in tqdm(range(200)):
    
    datatype = np.random.choice(['label_encoding','one_hot_encoding'])

    params = {'learning_rate': np.random.choice([0.01, 0.03, 0.05, 0.1, 0.3]),
          'depth': [4, 5, 6, 8, 10],
          'loss_function': 'CrossEntropy',
          'l2_leaf_reg': np.random.choice([0.01, 0.1, 0.5, 1, 3]),
          'leaf_estimation_iterations': [10,30,50],
          'subsample' : np.random.choice([0.8,1]),
          'eval_metric': 'F1',
          'iterations': np.random.choice([10,50,100,200,300,500,700,1000]),
          'logging_level':'Silent',
   #       'class_weights' : [0.1, 4]
          'random_seed': [42]
         }
        
        
    clf = CatBoostClassifier(**params, task_type='GPU')
    params['datatype'] = datatype

    cv_score = []  
    cv_thres = []
    print('*'*40)
    print('params : ',params)
    print('*'*40)
    for i in range(2):
        
        if datatype == 'label_encoding':
            train_indices = train[train.set!=i].index
            valid_indices = train[train.set==i].index

            X_train = train[features].values[train_indices]
            y_train = train[target].values[train_indices]

            X_valid = train[features].values[valid_indices]
            y_valid = train[target].values[valid_indices]
            
        else : 
            train_indices = train2[train2.set!=i].index
            valid_indices = train2[train2.set==i].index

            X_train = train2[features2].values[train_indices]
            y_train = train2[target].values[train_indices]

            X_valid = train2[features2].values[valid_indices]
            y_valid = train2[target].values[valid_indices]

            
        start = time.time()
        clf.fit(X_train, y_train,cat_features = cat_idxs, eval_set = [(X_train, y_train),(X_valid,y_valid)], early_stopping_rounds = 30)
        pred_valid = clf.predict(X_valid)


        f1_valid = f1_score(y_valid, pred_valid) # y_true : 정답 값, y_pred : 예측 값 
        roc_auc = roc_auc_score(y_valid, pred_valid)

        print("Valid F1 score: {}".format(f1_valid))
        print("Valid ROC AUC Score: {}".format(roc_auc))

        pred_prob_valid = clf.predict_proba(X_valid)
        end = time.time()


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

    score_file.write("params : %s , cv_mean(best_f1) %2.2f , best_threshold : %s, train_time : %2.2f\n"%(params,cv_mean_score, cv_thres,end-start))
    score_file.flush()
    
               

