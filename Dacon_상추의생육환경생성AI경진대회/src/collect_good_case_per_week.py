
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
test_dataset_pred_path = '/Data2/dacon/model_ti/submission2/output/output_testset_pred/week_dataset_pred.csv'
best_case_per_week_path = '/Data2/dacon/model_ti/submission2/output/output_testset_pred/best_case_per_week.csv'

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

dataset_pred = pd.DataFrame()
for j,(x,y) in tqdm(enumerate(zip(all_x_list,all_y_list))) :

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
    dataset_pred = pd.concat([dataset_pred,data])
dataset_pred.to_csv(test_dataset_pred_path ,index=None)

data_pred = pd.read_csv(test_dataset_pred_path)

w1_init_weight = np.array(data_pred[(data_pred['DAT'] == 1) &(data_pred['obs_time'] == 0)]['predicted_weight_g'].to_list())
w1_last_weight = np.array(data_pred[(data_pred['DAT'] == 7) &(data_pred['obs_time'] == 23)]['predicted_weight_g'].to_list())
w2_init_weight = np.array(data_pred[(data_pred['DAT'] == 8) &(data_pred['obs_time'] == 0)]['predicted_weight_g'].to_list())
w2_last_weight = np.array(data_pred[(data_pred['DAT'] == 14) &(data_pred['obs_time'] == 23)]['predicted_weight_g'].to_list())
w3_init_weight = np.array(data_pred[(data_pred['DAT'] == 15) &(data_pred['obs_time'] == 0)]['predicted_weight_g'].to_list())
w3_last_weight = np.array(data_pred[(data_pred['DAT'] == 21) &(data_pred['obs_time'] == 23)]['predicted_weight_g'].to_list())
w4_init_weight = np.array(data_pred[(data_pred['DAT'] == 22) &(data_pred['obs_time'] == 0)]['predicted_weight_g'].to_list())
w4_last_weight = np.array(data_pred[(data_pred['DAT'] == 28) &(data_pred['obs_time'] == 23)]['predicted_weight_g'].to_list())

print('best_week1_case_index : ',np.array(w1_last_weight-w1_init_weight).argmax())
print('best_week2_case_index : ',np.array(w2_last_weight-w2_init_weight).argmax())
print('best_week3_case_index : ',np.array(w3_last_weight-w3_init_weight).argmax())
print('best_week4_case_index : ',np.array(w4_last_weight-w4_init_weight).argmax())
w1_idx = np.array(w1_last_weight-w1_init_weight).argmax()
w2_idx = np.array(w2_last_weight-w2_init_weight).argmax()
w3_idx = np.array(w3_last_weight-w3_init_weight).argmax()
w4_idx = np.array(w4_last_weight-w4_init_weight).argmax()
weeks_idx = [w1_idx,w2_idx,w3_idx,w4_idx]

gen_col =[0,1,2,3,4,5,6,8,10,12]    # 누적 변수, 총광량 변수 제외 
dataset_week_concat = pd.DataFrame()
for j,case in enumerate(weeks_idx) : 
    week_data = data_pred.iloc[28*24*case + j*7*24 : 28*24*case + (j+1)*7*24,gen_col]
    dataset_week_concat = pd.concat([dataset_week_concat,week_data])
dataset_week_concat.reset_index(drop=True,inplace=True)


# 백색광량은 값이 0이므로 generate에서 제외

col = [0,1,2,3,4,5,6,8,9]
dataset_week_concat1 = dataset_week_concat.iloc[:,col]

dataset_week_concat1.to_csv(best_case_per_week_path, index=False)