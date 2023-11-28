'''
AAMsoftmax loss function copied from voxceleb_trainer: https://github.com/clovaai/voxceleb_trainer/blob/master/loss/aamsoftmax.py
'''

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

def train(train_x_path, train_y_path,save_path):
    print ('Available devices ', torch.cuda.device_count())
    GPU_NUM = 1 # 원하는 GPU 번호 입력
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Device:', device)
    print('Current cuda device:', torch.cuda.current_device())


    train_x_list = glob(train_x_path+'*.csv')
    train_y_list = glob(train_y_path+'*.csv')

    train_x_list = sorted(train_x_list)
    train_y_list = sorted(train_y_list)

    print('Data preprocessing,,,,')
    dataset_scale = pd.DataFrame()
    for i,(x,y) in tqdm(enumerate(zip(train_x_list,train_y_list))) :
        data_x = pd.read_csv(x)
        data_y = pd.read_csv(y)
        y_diff = data_y['predicted_weight_g'].diff().to_list()[1:]
        y_diff.append(y_diff[26])
        data_y['weight_diff'] = y_diff
        data_x['DAT'] = data_x['DAT']+1

        # 전처리
        if i == 5 : 
            a = np.array(data_x.iloc[:,8].to_list()) # 시간당백색광량
            a1 = np.array(data_x.iloc[:,9].to_list()) # 일간누적백색광량
            b = np.array(data_x.iloc[:,10].to_list()) # 시간당적색광량
            b1 = np.array(data_x.iloc[:,11].to_list()) # 일간누적적색광량
            c = np.array(data_x.iloc[:,12].to_list()) # 시간당청색광량
            c1 = np.array(data_x.iloc[:,13].to_list()) # 일간누적청색광량
            d = np.array(data_x.iloc[:,14].to_list()) # 시간당총광량
            d1 = np.array(data_x.iloc[:,15].to_list()) # 일간누적총광량
            a_ = a[:24]
            b_ = b[:24]
            c_ = c[:24]
            d_ = d[:24]
            for j in range(24):
                if a_[j] <0 :
                    a_[j] = (a_[j-1]+a_[j+1])/2
                if b_[j] <0 :
                    b_[j] = (b_[j-1]+b_[j+1])/2
                if c_[j] <0 :
                    c_[j] = (c_[j-1]+c_[j+1])/2
                if d_[j] <0 :
                    d_[j] = (d_[j-1]+d_[j+1])/2
            a[:24] = a_
            a1[:24] = a_.cumsum()
            b[:24] = b_
            b1[:24] = b_.cumsum()
            c[:24] = c_
            c1[:24] = c_.cumsum()
            d[:24] = d_
            d1[:24] = d_.cumsum()
            data_x['시간당백색광량'] = a
            data_x['일간누적백색광량'] = a1
            data_x['시간당적색광량'] = b
            data_x['일간누적적색광량'] = b1
            data_x['시간당청색광량'] = c
            data_x['일간누적청색광량'] = c1
            data_x['시간당총광량'] = d
            data_x['일간누적총광량'] = d1

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


        data = pd.merge(data_x, data_y, how='left', on='DAT')
        dataset_scale = pd.concat([dataset_scale,data])



    dataset_scale.reset_index(drop=True,inplace=True)

    time = dataset_scale['obs_time'].to_list()
    time_to_int = []
    for t in tqdm(time) : 
        h = int(t.split(':')[0])
        m = int(t.split(':')[1])
        if m == 59 :
            h +=1
        time_to_int.append(h)
    dataset_scale['obs_time'] = time_to_int
    print('Data Preprocessing Done.')
    
    
    print('Start Training,,,,')
    predictor_scale = [i for i in dataset_scale.columns.values if i not in ['predicted_weight_g','weight_diff']] #
    y_scale_d = dataset_scale.columns.to_list()[-1]  # diff 값을 predict
    y_scale_w = dataset_scale.columns.to_list()[-2]

   
    MAX_EPOCH=300
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
    regressor_scale = TabNetRegressor(**tabnet_params)
    regressor_scale.fit(X_train=dataset_scale.loc[:,predictor_scale].values, y_train=np.array(dataset_scale.loc[:,y_scale_w].values).reshape(-1,1),
              patience=300, max_epochs=MAX_EPOCH,
              eval_metric=['rmse'])
    print('Training Done.')
    regressor_scale.save_model(save_path)
    print('Save Model.')
    return regressor_scale

def inference(model,test_x_path,test_y_path,test_output_path):

    print('Start Inference,,,,')

    test_x_list = glob(test_x_path+'*.csv')
    test_y_list = glob(test_y_path+'*.csv')

    test_x_list = sorted(test_x_list)
    test_y_list = sorted(test_y_list)

    output_name = ['TEST_01.csv','TEST_02.csv','TEST_03.csv','TEST_04.csv','TEST_05.csv']
    
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
        print(prediction)
        data_y['predicted_weight_g'] = prediction  
        data_y.reset_index(drop=True, inplace=True)
        data_y.to_csv(test_output_path + output_name[j],index=None)

    print('Inference Done.')

    
if __name__ == "__main__":

    warnings.filterwarnings('ignore')
    warnings.simplefilter("ignore")
    
    train_x_path = '/Data2/dacon/model_ti/submission2/data/train/train_input/'
    train_y_path = '/Data2/dacon/model_ti/submission2/data/train/train_target/'
    save_path = '/Data2/dacon/model_ti/submission2/model/prediction/tabnet_prediction'
    
    model = train(train_x_path,train_y_path,save_path)
    
    test_x_path = '/Data2/dacon/model_ti/submission2/data/test/test_input/'
    test_y_path = '/Data2/dacon/model_ti/submission2/data/test/test_target/'
    test_output_path = '/Data2/dacon/model_ti/submission2/output/output_tabnet_sub/'
    
    inference(model,test_x_path,test_y_path,test_output_path)
    print('ALL DONE.')
    
    
    
    
    
    
