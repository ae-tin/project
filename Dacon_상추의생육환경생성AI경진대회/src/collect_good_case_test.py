
import torch, math
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm

from sklearn.metrics import mean_squared_error
import argparse, os, warnings, time
import os
from pytorch_tabnet.tab_model import TabNetRegressor

from sklearn.preprocessing import StandardScaler

saved_filepath = '/Data2/dacon/model_ti/submission2/model/prediction/tabnet_prediction.zip'
train_x_path = '/Data2/dacon/model_ti/submission2/data/train/train_input/'
train_y_path = '/Data2/dacon/model_ti/submission2/data/train/train_target/'
test_x_path = '/Data2/dacon/model_ti/submission2/data/test/test_input/'
test_y_path = '/Data2/dacon/model_ti/submission2/data/test/test_target/'
test_dataset_pred_path = '/Data2/dacon/model_ti/submission2/output/output_testset_pred/test_dataset_pred.csv'
best_case_test_path = '/Data2/dacon/model_ti/submission2/output/output_testset_pred/best_case_test.csv'

tabnet_params = dict(n_d=24, n_a=24, n_steps=1, gamma=1.3,
                     lambda_sparse=0, optimizer_fn=torch.optim.Adam,
                     optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
                     mask_type='entmax',
                     scheduler_params=dict(mode="min",
                                           patience=5,
                                           min_lr=1e-5,
                                           factor=0.9,),
                     scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
                     verbose=10,
                     )
model = TabNetRegressor(**tabnet_params)
model.load_model(saved_filepath)



test_x_list = glob(test_x_path+'*.csv')
test_y_list = glob(test_y_path+'*.csv')

test_x_list = sorted(test_x_list)
test_y_list = sorted(test_y_list)

train_x_list = glob(train_x_path+'*.csv')
train_y_list = glob(train_y_path+'*.csv')

train_x_list = sorted(train_x_list)
train_y_list = sorted(train_y_list)



all_x_list = train_x_list+test_x_list
all_y_list = train_y_list+test_y_list



output_name = ['TEST_01.csv','TEST_02.csv','TEST_03.csv','TEST_04.csv','TEST_05.csv']

dataset = pd.DataFrame()
for j,(x,y) in tqdm(enumerate(zip(test_x_list,test_y_list))) :

    data_x = pd.read_csv(x)
    data_y = pd.read_csv(y)
    data_x['DAT'] = data_x['DAT']+1
    time = data_x['obs_time'].to_list()
    time_to_int = []
    for t in time : 
        h = int(t.split(':')[0])
        m = int(t.split(':')[1])
        if m == 59 :
            h +=1
        time_to_int.append(h)
    data_x['obs_time'] = time_to_int



    # 일 누적
    not_cum_feature = [i for i in data_x.columns.values if '관측' in i]
    for fea in not_cum_feature :
        arr = np.array(data_x[fea].to_list())
        day_cum = np.zeros(arr.shape[0])
        for i in range(28):
            day_cum[(i)*24:(i+1)*24] = arr[(i)*24:(i+1)*24].cumsum()

        data_x['일간누적'+fea] = day_cum

    # 총 누적
    cum_feature = [i for i in data_x.columns.values if '일간' in i]
    for fea in cum_feature :
        arr = np.array(data_x[fea].to_list())

        data_x['총누적'+fea] = arr.cumsum()  

    feature = [i for i in data_x.columns.values if i not in ['DAT','obs_time','predicted_weight_g','weight_diff']]

    scaler = StandardScaler()
    input_std = scaler.fit_transform(data_x[['ec관측치','시간당분무량']])
    ec = input_std[:,0]
    qnsan = input_std[:,1]
    data_x['ec_분무량_minus'] = ec-qnsan


    # 간단한 통계량 추가
    for fea in feature : 
        arr = np.array(data_x[fea].to_list())
        data_x[fea+'_mean'] = arr.mean().repeat(arr.shape[0])
        data_x[fea+'_std'] = arr.std().repeat(arr.shape[0])


    y_pred = model.predict(data_x.values)
    y_pred = y_pred.reshape(-1)
    prediction = []
    for i in range(28) :
        prediction.append(sum(y_pred[i*24:(i+1)*24])/24)
#    print(prediction)
    data_y['predicted_weight_g'] = prediction  
    data = pd.merge(data_x, data_y, how='left', on='DAT')
    dataset = pd.concat([dataset,data])
dataset.reset_index(drop=True,inplace=True)
dataset.to_csv(test_dataset_pred_path ,index=None)


test_last_weight = np.array(dataset[(dataset['DAT'] == 28) &(dataset['obs_time'] == 23)]['predicted_weight_g'].to_list())

print('best_test_case_index : ',test_last_weight.argmax())
best_test_idx = test_last_weight.argmax()

gen_col =[0,1,2,3,4,5,6,8,10,12]    # 누적 변수, 총광량 변수 제외 
best_case_x = dataset.iloc[28*24*best_test_idx : 28*24*(best_test_idx+1) ,gen_col]
best_case_x.reset_index(drop=True,inplace=True)



best_case_x.to_csv(best_case_test_path, index=False)