#from tensorflow import keras
#from tensorflow.keras import layers

# -*- coding: utf-8 -*-
import argparse
import os
import time
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import sys
import random
from tqdm import tqdm
from tools import utils
from torch import Tensor
import torch.nn.functional as F
from jiwer import wer
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
import torch


def get_box(lambda_value, nf, nt):
    cut_rat = np.sqrt(1.0 - lambda_value)

    cut_w = int(nf * cut_rat)  # rw
    cut_h = int(nt * cut_rat)  # rh

    cut_x = int(np.random.uniform(low=0, high=nf))  # rx
    cut_y = int(np.random.uniform(low=0, high=nt))  # ry

    boundaryx1 = np.minimum(np.maximum(cut_x - cut_w // 2, 0), nf) #tf.clip_by_value(cut_x - cut_w // 2, 0, IMG_SIZE_x)
    boundaryy1 = np.minimum(np.maximum(cut_y - cut_h // 2, 0), nt) #tf.clip_by_value(cut_y - cut_h // 2, 0, IMG_SIZE_y)
    bbx2 = np.minimum(np.maximum(cut_x + cut_w // 2, 0), nf) #tf.clip_by_value(cut_x + cut_w // 2, 0, IMG_SIZE_x)
    bby2 = np.minimum(np.maximum(cut_y + cut_h // 2, 0), nt) #tf.clip_by_value(cut_y + cut_h // 2, 0, IMG_SIZE_y)

    target_h = bby2 - boundaryy1
    if target_h == 0:
        target_h += 1

    target_w = bbx2 - boundaryx1
    if target_w == 0:
        target_w += 1

    return boundaryx1, boundaryy1, target_h, target_w  

class Aug(nn.Module):
    
    def __init__(self,mixup = True,alpha = 0.7,cutout = 0.8):
        super(Aug, self).__init__()
        self.mixup = mixup
        self.alpha = alpha 
        self.cutout = cutout

    def forward(self,x,y,x_,y_) :  
    
        if len(x.shape) == 4: 
            b, c, h, w = x.shape
            l = np.random.beta(self.alpha, self.alpha, b)
            X_l = l.reshape(b, 1, 1, 1)
            y_l = l.reshape(b, 1)
        elif len(x.shape) == 3:
            b, h, w = self.X_train.shape
            l = np.random.beta(self.alpha, self.alpha, b)
            X_l = l.reshape(b, 1, 1)
            y_l = l.reshape(b, 1)
        elif len(x.shape) == 2:
            b, h = x.shape
            l = np.random.beta(self.alpha, self.alpha, b)
            X_l = l.reshape(b, 1)
            y_l = l.reshape(b, 1)

        X1 = x
        X2 = x_
        if self.mixup :
            Xn = X1 * X_l + X2 * (1 - X_l)
        else :
            Xn = X1
        if len(x.shape) == 4: 
            if self.cutout :
                lambda1 = np.random.beta(self.cutout, self.cutout, size = b)   ## beta_param default : 0.7  STC페이퍼 추천은 0.6~0.8
                for i in range(b) :
                    boundaryx1, boundaryy1, target_h, target_w = get_box(lambda1[i], h, w)
                    Xn[i, boundaryx1:(boundaryx1+target_h), boundaryy1:(boundaryy1+target_w),: ] = 0



        y1 = y
        y2 = y_
        y = y1 * y_l + y2 * (1 - y_l)

        return Xn, y
    
    
    
    
     
    
    
    
    
    

class LCNN(nn.Module):

    def __init__(self,mode):

        super(LCNN, self).__init__()
        '''input : (B,1,F,T)'''
        
        self.mode = mode
        
        self.conv2d_1  = nn.Conv2d(1, 32, kernel_size=(5,5), stride=(1,1), padding=(2,2))
        self.conv2d_2  = nn.Conv2d(1, 32, kernel_size=(5,5), stride=(1,1), padding=(2,2))
        
        self.max2d_1 = nn.MaxPool2d(kernel_size =(2,2), stride=(2,2),padding = 0)
        
        self.conv2d_3  = nn.Conv2d(32, 32, kernel_size=(1,1), stride=(1,1))
        self.conv2d_4  = nn.Conv2d(32, 32, kernel_size=(1,1), stride=(1,1))
        
        self.bn2d_1 = nn.BatchNorm2d(32)
        
        self.conv2d_5  = nn.Conv2d(32, 48, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.conv2d_6  = nn.Conv2d(32, 48, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        
        self.max2d_2 = nn.MaxPool2d(kernel_size =(2,2), stride=(2,2),padding = 0)
        self.bn2d_2 = nn.BatchNorm2d(48)
        
        
        self.conv2d_7  = nn.Conv2d(48, 48, kernel_size=(1,1), stride=(1,1))
        self.conv2d_8  = nn.Conv2d(48, 48, kernel_size=(1,1), stride=(1,1))
        
        self.bn2d_3 = nn.BatchNorm2d(48)
        
        self.conv2d_9  = nn.Conv2d(48, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.conv2d_10  = nn.Conv2d(48, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        
        self.max2d_3 = nn.MaxPool2d(kernel_size =(2,2), stride=(2,2),padding = 0)
        
        self.conv2d_11  = nn.Conv2d(64, 64, kernel_size=(1,1), stride=(1,1))
        self.conv2d_12  = nn.Conv2d(64, 64, kernel_size=(1,1), stride=(1,1))
        
        self.bn2d_4 = nn.BatchNorm2d(64)
        
        self.conv2d_13  = nn.Conv2d(64, 32, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.conv2d_14  = nn.Conv2d(64, 32, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        
        self.bn2d_5 = nn.BatchNorm2d(32)
        
        
        self.conv2d_15  = nn.Conv2d(32, 32, kernel_size=(1,1), stride=(1,1))
        self.conv2d_16  = nn.Conv2d(32, 32, kernel_size=(1,1), stride=(1,1))
        
        self.bn2d_6 = nn.BatchNorm2d(32)
        
        
        self.conv2d_17  = nn.Conv2d(32, 32, kernel_size=(1,1), stride=(1,1))
        self.conv2d_18  = nn.Conv2d(32, 32, kernel_size=(1,1), stride=(1,1))
        
        self.max2d_4 = nn.MaxPool2d(kernel_size =(2,2), stride=(2,2),padding = 0)
        
        self.adaptavg = nn.AdaptiveAvgPool2d(1)
        
        
#        self.fc1 = nn.Linear(64, 128)
        
        if self.mode == 'murmur' :
            self.fc_mur = nn.Sequential(
                                        nn.Linear(32,1),
                                        nn.Sigmoid()
                                        )
        elif self.mode == 'outcome' :
            self.fc_out = nn.Sequential(
                                        nn.Linear(32,2),
                                        nn.Softmax()
                                        )
        
        
    def forward(self, input):
        
        x1 = self.conv2d_1(input)
        x2 = self.conv2d_2(input)
        max1 = torch.maximum(x1,x2)
        x = self.max2d_1(max1)
        
        x3 = self.conv2d_3(x)
        x4 = self.conv2d_4(x)
        max2 = torch.maximum(x3,x4)
        x = self.bn2d_1(max2)
        
        x5 = self.conv2d_5(x)
        x6 = self.conv2d_6(x)
        max3 = torch.maximum(x5,x6)
        x = self.max2d_2(max3)
        x = self.bn2d_2(x)
        
        x7 = self.conv2d_7(x)
        x8 = self.conv2d_8(x)
        max4 = torch.maximum(x7,x8)
        x = self.bn2d_3(max4)
        
        x9 = self.conv2d_9(x)
        x10 = self.conv2d_10(x)
        max5 = torch.maximum(x9,x10)
        x = self.max2d_3(max5)
        
        x11 = self.conv2d_11(x)
        x12 = self.conv2d_12(x)
        max6 = torch.maximum(x11,x12)
        x = self.bn2d_4(max6)
        
        x13 = self.conv2d_13(x)
        x14 = self.conv2d_14(x)
        max7 = torch.maximum(x13,x14)
        x = self.bn2d_5(max7)
        
        x15 = self.conv2d_15(x)
        x16 = self.conv2d_16(x)
        max8 = torch.maximum(x15,x16)
        x = self.bn2d_6(max8)
        
        x17 = self.conv2d_17(x)
        x18 = self.conv2d_18(x)
        max9 = torch.maximum(x17,x18)
        x = self.max2d_4(max9)
        
        x = self.adaptavg(x)
        x = x.view(x.size(0),-1)
        
#        x = self.fc1(x)
        
        
        if self.mode == 'murmur' :
            out = self.fc_mur(x)
        elif self.mode == 'outcome' :
            out = self.fc_out(x)
        
        return out
    
class W2V2_LCNN(nn.Module):

    def __init__(self,mode):

        super(W2V2_LCNN, self).__init__()
        '''input : (B,T)'''
        
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.w2v2 = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        self.w2v2.lm_head = nn.Sequential(
                nn.Identity()
                    )

        
        self.mode = mode
        
        self.conv2d_1  = nn.Conv2d(1, 32, kernel_size=(5,5), stride=(1,1), padding=(2,2))
        self.conv2d_2  = nn.Conv2d(1, 32, kernel_size=(5,5), stride=(1,1), padding=(2,2))
        
        self.max2d_1 = nn.MaxPool2d(kernel_size =(2,2), stride=(2,2),padding = 0)
        
        self.conv2d_3  = nn.Conv2d(32, 32, kernel_size=(1,1), stride=(1,1))
        self.conv2d_4  = nn.Conv2d(32, 32, kernel_size=(1,1), stride=(1,1))
        
        self.bn2d_1 = nn.BatchNorm2d(32)
        
        self.conv2d_5  = nn.Conv2d(32, 48, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.conv2d_6  = nn.Conv2d(32, 48, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        
        self.max2d_2 = nn.MaxPool2d(kernel_size =(2,2), stride=(2,2),padding = 0)
        self.bn2d_2 = nn.BatchNorm2d(48)
        
        
        self.conv2d_7  = nn.Conv2d(48, 48, kernel_size=(1,1), stride=(1,1))
        self.conv2d_8  = nn.Conv2d(48, 48, kernel_size=(1,1), stride=(1,1))
        
        self.bn2d_3 = nn.BatchNorm2d(48)
        
        self.conv2d_9  = nn.Conv2d(48, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.conv2d_10  = nn.Conv2d(48, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        
        self.max2d_3 = nn.MaxPool2d(kernel_size =(2,2), stride=(2,2),padding = 0)
        
        self.conv2d_11  = nn.Conv2d(64, 64, kernel_size=(1,1), stride=(1,1))
        self.conv2d_12  = nn.Conv2d(64, 64, kernel_size=(1,1), stride=(1,1))
        
        self.bn2d_4 = nn.BatchNorm2d(64)
        
        self.conv2d_13  = nn.Conv2d(64, 32, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.conv2d_14  = nn.Conv2d(64, 32, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        
        self.bn2d_5 = nn.BatchNorm2d(32)
        
        
        self.conv2d_15  = nn.Conv2d(32, 32, kernel_size=(1,1), stride=(1,1))
        self.conv2d_16  = nn.Conv2d(32, 32, kernel_size=(1,1), stride=(1,1))
        
        self.bn2d_6 = nn.BatchNorm2d(32)
        
        
        self.conv2d_17  = nn.Conv2d(32, 32, kernel_size=(1,1), stride=(1,1))
        self.conv2d_18  = nn.Conv2d(32, 32, kernel_size=(1,1), stride=(1,1))
        
        self.max2d_4 = nn.MaxPool2d(kernel_size =(2,2), stride=(2,2),padding = 0)
        
        self.adaptavg = nn.AdaptiveAvgPool2d(1)
        
        
#        self.fc1 = nn.Linear(64, 128)
        
        if self.mode == 'murmur' :
            self.fc_mur = nn.Sequential(
                                        nn.Linear(32,1),
                                        nn.Sigmoid()
                                        )
        elif self.mode == 'outcome' :
            self.fc_out = nn.Sequential(
                                        nn.Linear(32,1),
                                        nn.Sigmoid()
                                        )
            
    def forward(self, input):
        
        w2v2_out = self.w2v2(input).logits    # output.shape = (B,249,768)
        w2v2_out = w2v2_out.unsqueeze(1)
        
        x1 = self.conv2d_1(w2v2_out)
        x2 = self.conv2d_2(w2v2_out)
        max1 = torch.maximum(x1,x2)
        x = self.max2d_1(max1)
        
        x3 = self.conv2d_3(x)
        x4 = self.conv2d_4(x)
        max2 = torch.maximum(x3,x4)
        x = self.bn2d_1(max2)
        
        x5 = self.conv2d_5(x)
        x6 = self.conv2d_6(x)
        max3 = torch.maximum(x5,x6)
        x = self.max2d_2(max3)
        x = self.bn2d_2(x)
        
        x7 = self.conv2d_7(x)
        x8 = self.conv2d_8(x)
        max4 = torch.maximum(x7,x8)
        x = self.bn2d_3(max4)
        
        x9 = self.conv2d_9(x)
        x10 = self.conv2d_10(x)
        max5 = torch.maximum(x9,x10)
        x = self.max2d_3(max5)
        
        x11 = self.conv2d_11(x)
        x12 = self.conv2d_12(x)
        max6 = torch.maximum(x11,x12)
        x = self.bn2d_4(max6)
        
        x13 = self.conv2d_13(x)
        x14 = self.conv2d_14(x)
        max7 = torch.maximum(x13,x14)
        x = self.bn2d_5(max7)
        
        x15 = self.conv2d_15(x)
        x16 = self.conv2d_16(x)
        max8 = torch.maximum(x15,x16)
        x = self.bn2d_6(max8)
        
        x17 = self.conv2d_17(x)
        x18 = self.conv2d_18(x)
        max9 = torch.maximum(x17,x18)
        x = self.max2d_4(max9)
        
        x = self.adaptavg(x)
        x = x.view(x.size(0),-1)
        
        
        if self.mode == 'murmur' :
            out = self.fc_mur(x)
        elif self.mode == 'outcome' :
            out = self.fc_out(x)
#        out = {'murmur':out1, 'outcome':out2}
        
        return out
    
class Freq_attention(nn.Module):
    def __init__(self, freq_len, reduction = 8, pool_types = ['mean','max']):
        super(Freq_attention,self).__init__()
        '''input : (B,F,T)'''
        self.freq_len = freq_len
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(freq_len, freq_len // reduction),
            nn.ReLU(),
            nn.Linear(freq_len // reduction, freq_len)
            )
        self.pool_types = pool_types
        
    def forward(self, x):
        freq_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='mean':
#                print('before att : ',x.shape)
                avg_pool = F.avg_pool2d( x, (1, x.size(2)), stride=(1, x.size(2)))
#                print('after att : ',avg_pool.shape)
                freq_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (1, x.size(2)), stride=(1, x.size(2)))
                freq_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (1, x.size(2)), stride=(1, x.size(2)))
                freq_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                lse_pool = logsumexp_2d(x)
                freq_att_raw = self.mlp( lse_pool )
                
            if freq_att_sum == None :
                freq_att_sum = freq_att_raw
            else :
                freq_att_sum = freq_att_sum + freq_att_raw

        scale = torch.sigmoid( freq_att_sum ).unsqueeze(2).expand_as(x)
#        print('att scale : ',scale.shape)
        return x * scale
    