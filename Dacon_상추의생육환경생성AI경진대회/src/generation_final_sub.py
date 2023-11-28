#!/usr/bin/env python
# coding: utf-8

# In[81]:



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

train_gen_path = '/Data2/dacon/model_ti/submission2/output/output_generation_train/gen_dataset_tmp_1000.csv'
test_gen_path = '/Data2/dacon/model_ti/submission2/output/output_generation_test/gen_dataset_tmp_1000.csv'  # 검증용
week_gen_path = '/Data2/dacon/model_ti/submission2/output/output_generation_week/gen_dataset_tmp_1000.csv'

from_train_case_generation_sub_path = '/Data2/dacon/model_ti/submission2/output/from_train_case_generation_output_sub/'
from_test_case_generation_sub_path = '/Data2/dacon/model_ti/submission2/output/from_test_case_generation_output/'   # 검증용
from_week_case_generation_sub_path = '/Data2/dacon/model_ti/submission2/output/from_week_case_generation_output_sub/'


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


train_x_list = glob(train_x_path+'*.csv')
train_y_list = glob(train_y_path+'*.csv')

train_x_list = sorted(train_x_list)
train_y_list = sorted(train_y_list)

output_x_name = 'generation.csv'
output_y_name = 'predicted_weight_g.csv'


# - 시간당 내부 평균온도: 4도~40도
# - 시간당 내부 평균습도: 0% ~ 100%
# - 시간당 내부 평균 co2 농도 : 0ppm ~ 1200 ppm
# - 시간당 평균 EC : 0 ~ 8
# - 시간당 분무량 : 0 ~ 3000 / 일간 누적 분무량 0 ~ 72,000
# - 시간당 백색광량 : 0 ~ 120,000 / 일간 누적 백색광량 0 ~ 2,880,000
# - 시간당 적색광량 : 0 ~ 120,000 / 일간 누적 적색광량 0 ~ 2,880,000 
# - 시간당 청색광량 : 0 ~ 120,000 / 일간 누적 청색광량 0 ~ 2,880,000
# - 시간당 총광량 : 0 ~ 120,000 / 일간 누적 총광량 0 ~ 2,880,000


### train case ###
data_x = pd.read_csv(train_gen_path)
data_y = pd.read_csv(train_y_list[0])

tem = data_x['내부온도관측치'].to_list()
for i, value in enumerate(tem) :
    if value < 4 :
        tem[i] = 4
    if value > 40 :
        tem[i] = 40
data_x['내부온도관측치'] = tem

hum = data_x['내부습도관측치'].to_list()
for i, value in enumerate(hum) :
    if value < 0 :
        hum[i] = 0
    if value > 100 :
        hum[i] = 100
data_x['내부습도관측치'] = hum
        
co2 = data_x['co2관측치'].to_list()
for i, value in enumerate(co2) :
    if value < 0 :
        co2[i] = 0
    if value > 1200 :
        co2[i] = 1200
data_x['co2관측치'] = co2

ec = data_x['ec관측치'].to_list()
for i, value in enumerate(ec) :
    if value < 0 :
        ec[i] = 0
    if value > 8 :
        ec[i] = 8   
data_x['ec관측치'] = ec
        
chic = data_x['시간당분무량'].to_list()
for i, value in enumerate(chic) :
    if value < 0 :
        chic[i] = 0
    if value > 3000 :
        chic[i] = 3000 
data_x['시간당분무량'] = chic
        
white = data_x['시간당백색광량'].to_list()
for i, value in enumerate(white) :
    if value < 0 :
        white[i] = 0
    if value > 120000 :
        white[i] = 120000 
data_x['시간당백색광량'] = white
        
red = data_x['시간당적색광량'].to_list()
for i, value in enumerate(red) :
    if value < 0 :
        red[i] = 0
    if value > 120000 :
        red[i] = 120000 
data_x['시간당적색광량'] = red
        
blue = data_x['시간당청색광량'].to_list()
for i, value in enumerate(blue) :
    if value < 0 :
        blue[i] = 0
    if value > 120000 :
        blue[i] = 120000
data_x['시간당청색광량'] = blue

dat = np.array(list(range(28))).repeat(24)+1
day_1 = data_x.iloc[:24,:]
data_x = pd.concat([day_1,data_x])
data_x['DAT'] = dat




arr = np.array(data_x['시간당분무량'].to_list())
new_arr = np.zeros(arr.shape[0])
for i in range(28) :
    new_arr[i*24:(i+1)*24] = arr[i*24:(i+1)*24].cumsum()

data_x[('일간누적분무량').replace('시간당','일간누적')] = new_arr

light_col = [i for i in data_x.columns.values if '광량' in i ]
light_sum = np.zeros(data_x.shape[0])
for col in light_col :
    arr = np.array(data_x[col].to_list())
    new_arr = np.zeros(arr.shape[0])
    for i in range(28) :
        new_arr[i*24:(i+1)*24] = arr[i*24:(i+1)*24].cumsum()
    
    data_x[col.replace('시간당','일간누적')] = new_arr
    light_sum = light_sum + arr

data_x['시간당총광량'] = light_sum
light_cumsum = np.zeros(data_x.shape[0])
for i in range(28):
    light_cumsum[i*24:(i+1)*24] = light_sum[i*24:(i+1)*24].cumsum()
data_x['일간누적총광량'] = light_cumsum




data_x = data_x[['DAT', 'obs_time', '내부온도관측치', '내부습도관측치','co2관측치','ec관측치','시간당분무량','일간누적분무량','시간당백색광량','일간누적백색광량','시간당적색광량','일간누적적색광량','시간당청색광량','일간누적청색광량','시간당총광량','일간누적총광량']]




data_x_sub = data_x
data_x_sub['DAT'] = data_x_sub['DAT']-1
data_x_sub.to_csv(from_train_case_generation_sub_path+output_x_name,index=False)



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
data_y['predicted_weight_g'] = prediction  
data_y.to_csv(from_train_case_generation_sub_path+output_y_name,index=False)

    
    

    
    
    
### test case ###
data_x = pd.read_csv(test_gen_path)
data_y = pd.read_csv(train_y_list[0])

tem = data_x['내부온도관측치'].to_list()
for i, value in enumerate(tem) :
    if value < 4 :
        tem[i] = 4
    if value > 40 :
        tem[i] = 40
data_x['내부온도관측치'] = tem

hum = data_x['내부습도관측치'].to_list()
for i, value in enumerate(hum) :
    if value < 0 :
        hum[i] = 0
    if value > 100 :
        hum[i] = 100
data_x['내부습도관측치'] = hum
        
co2 = data_x['co2관측치'].to_list()
for i, value in enumerate(co2) :
    if value < 0 :
        co2[i] = 0
    if value > 1200 :
        co2[i] = 1200
data_x['co2관측치'] = co2

ec = data_x['ec관측치'].to_list()
for i, value in enumerate(ec) :
    if value < 0 :
        ec[i] = 0
    if value > 8 :
        ec[i] = 8   
data_x['ec관측치'] = ec
        
chic = data_x['시간당분무량'].to_list()
for i, value in enumerate(chic) :
    if value < 0 :
        chic[i] = 0
    if value > 3000 :
        chic[i] = 3000 
data_x['시간당분무량'] = chic
        
#white = data_x['시간당백색광량'].to_list()
#for i, value in enumerate(white) :
#    if value < 0 :
#        white[i] = 0
#    if value > 120000 :
#        white[i] = 120000 
data_x['시간당백색광량'] = 0
        
red = data_x['시간당적색광량'].to_list()
for i, value in enumerate(red) :
    if value < 0 :
        red[i] = 0
    if value > 120000 :
        red[i] = 120000 
data_x['시간당적색광량'] = red
        
blue = data_x['시간당청색광량'].to_list()
for i, value in enumerate(blue) :
    if value < 0 :
        blue[i] = 0
    if value > 120000 :
        blue[i] = 120000
data_x['시간당청색광량'] = blue

dat = np.array(list(range(28))).repeat(24)+1
day_1 = data_x.iloc[:24,:]
data_x = pd.concat([day_1,data_x])
data_x['DAT'] = dat

arr = np.array(data_x['시간당분무량'].to_list())
new_arr = np.zeros(arr.shape[0])
for i in range(28) :
    new_arr[i*24:(i+1)*24] = arr[i*24:(i+1)*24].cumsum()

data_x[('일간누적분무량').replace('시간당','일간누적')] = new_arr

light_col = [i for i in data_x.columns.values if '광량' in i ]
light_sum = np.zeros(data_x.shape[0])
for col in light_col :
    arr = np.array(data_x[col].to_list())
    new_arr = np.zeros(arr.shape[0])
    for i in range(28) :
        new_arr[i*24:(i+1)*24] = arr[i*24:(i+1)*24].cumsum()
    
    data_x[col.replace('시간당','일간누적')] = new_arr
    light_sum = light_sum + arr

data_x['시간당총광량'] = light_sum
light_cumsum = np.zeros(data_x.shape[0])
for i in range(28):
    light_cumsum[i*24:(i+1)*24] = light_sum[i*24:(i+1)*24].cumsum()
data_x['일간누적총광량'] = light_cumsum

data_x = data_x[['DAT', 'obs_time', '내부온도관측치', '내부습도관측치','co2관측치','ec관측치','시간당분무량','일간누적분무량','시간당백색광량','일간누적백색광량','시간당적색광량','일간누적적색광량','시간당청색광량','일간누적청색광량','시간당총광량','일간누적총광량']]

data_x_sub = data_x
data_x_sub['DAT'] = data_x_sub['DAT']-1
data_x_sub.to_csv(from_test_case_generation_sub_path+output_x_name,index=False)

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
data_y['predicted_weight_g'] = prediction  
data_y.to_csv(from_test_case_generation_sub_path+output_y_name,index=False)    
    
    
    
    
    
    
    
    
    
    



### week case ###
data_x = pd.read_csv(week_gen_path)
data_y = pd.read_csv(train_y_list[0])

tem = data_x['내부온도관측치'].to_list()
for i, value in enumerate(tem) :
    if value < 4 :
        tem[i] = 4
    if value > 40 :
        tem[i] = 40
data_x['내부온도관측치'] = tem

hum = data_x['내부습도관측치'].to_list()
for i, value in enumerate(hum) :
    if value < 0 :
        hum[i] = 0
    if value > 100 :
        hum[i] = 100
data_x['내부습도관측치'] = hum
        
co2 = data_x['co2관측치'].to_list()
for i, value in enumerate(co2) :
    if value < 0 :
        co2[i] = 0
    if value > 1200 :
        co2[i] = 1200
data_x['co2관측치'] = co2

ec = data_x['ec관측치'].to_list()
for i, value in enumerate(ec) :
    if value < 0 :
        ec[i] = 0
    if value > 8 :
        ec[i] = 8   
data_x['ec관측치'] = ec
        
chic = data_x['시간당분무량'].to_list()
for i, value in enumerate(chic) :
    if value < 0 :
        chic[i] = 0
    if value > 3000 :
        chic[i] = 3000 
data_x['시간당분무량'] = chic
        
#white = data_x['시간당백색광량'].to_list()
#for i, value in enumerate(white) :
#    if value < 0 :
#        white[i] = 0
#    if value > 120000 :
#        white[i] = 120000 
data_x['시간당백색광량'] = 0
        
red = data_x['시간당적색광량'].to_list()
for i, value in enumerate(red) :
    if value < 0 :
        red[i] = 0
    if value > 120000 :
        red[i] = 120000 
data_x['시간당적색광량'] = red
        
blue = data_x['시간당청색광량'].to_list()
for i, value in enumerate(blue) :
    if value < 0 :
        blue[i] = 0
    if value > 120000 :
        blue[i] = 120000
data_x['시간당청색광량'] = blue

dat = np.array(list(range(28))).repeat(24)+1
day_1 = data_x.iloc[:24,:]
data_x = pd.concat([day_1,data_x])
data_x['DAT'] = dat

arr = np.array(data_x['시간당분무량'].to_list())
new_arr = np.zeros(arr.shape[0])
for i in range(28) :
    new_arr[i*24:(i+1)*24] = arr[i*24:(i+1)*24].cumsum()

data_x[('일간누적분무량').replace('시간당','일간누적')] = new_arr

light_col = [i for i in data_x.columns.values if '광량' in i ]
light_sum = np.zeros(data_x.shape[0])
for col in light_col :
    arr = np.array(data_x[col].to_list())
    new_arr = np.zeros(arr.shape[0])
    for i in range(28) :
        new_arr[i*24:(i+1)*24] = arr[i*24:(i+1)*24].cumsum()
    
    data_x[col.replace('시간당','일간누적')] = new_arr
    light_sum = light_sum + arr

data_x['시간당총광량'] = light_sum
light_cumsum = np.zeros(data_x.shape[0])
for i in range(28):
    light_cumsum[i*24:(i+1)*24] = light_sum[i*24:(i+1)*24].cumsum()
data_x['일간누적총광량'] = light_cumsum

data_x = data_x[['DAT', 'obs_time', '내부온도관측치', '내부습도관측치','co2관측치','ec관측치','시간당분무량','일간누적분무량','시간당백색광량','일간누적백색광량','시간당적색광량','일간누적적색광량','시간당청색광량','일간누적청색광량','시간당총광량','일간누적총광량']]

data_x_sub = data_x
data_x_sub['DAT'] = data_x_sub['DAT']-1
data_x_sub.to_csv(from_week_case_generation_sub_path+output_x_name,index=False)

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
data_y['predicted_weight_g'] = prediction  
data_y.to_csv(from_week_case_generation_sub_path+output_y_name,index=False)

    
    
    
    