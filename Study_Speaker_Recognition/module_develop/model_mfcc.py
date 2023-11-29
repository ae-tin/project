'''
This is the ECAPA-TDNN model.
This model is modified and combined based on the following three projects:
  1. https://github.com/clovaai/voxceleb_trainer/issues/86
  2. https://github.com/lawlict/ECAPA-TDNN/blob/master/ecapa_tdnn.py
  3. https://github.com/speechbrain/speechbrain/blob/96077e9a1afff89d3f5ff47cab4bca0202770e4f/speechbrain/lobes/models/ECAPA_TDNN.py

'''

import math, torch, torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
import math
import librosa
'''
class Freq_Self_Att(nn.Module):
    def __init__(self) :
        super(Freq_Self_Att,self).__init__()
        self.key = nn.Linear(100,256)
        self.query = nn.Linear(100,256)
        self.scale = math.sqrt(256)
        self.adapt = nn.AdaptiveAvgPool1d(100)    # max 혹은 avg+mean 시도 가능
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self,input):
        
        x = self.adapt(input)
        key = self.key(x)
        query = self.query(x)
        att = self.softmax(torch.diagonal(torch.matmul(key,torch.transpose(query,1,2)),offset=0,dim1=-2,dim2=-1) / self.scale)
        att = att.unsqueeze(2).expand_as(input)
#        print(att.shape)
#        print(att)
        out = input * att
        
        return out
'''

def mfcc(x) :
    ex = librosa.feature.mfcc(S=x[0], n_mfcc=40)
    b_mfcc = np.empty((int(x.shape[0]),int(ex.shape[0]),int(ex.shape[1])))
    for i in range(int(x.shape[0])) :
        mfcc = librosa.feature.mfcc(S=x[i,:,:], n_mfcc=40)
        b_mfcc[i] = mfcc
    return b_mfcc
def cepstrum(x) :
    cepstrum = np.empty((int(x.shape[0]),int(x.shape[1]),int(x.shape[2])))
    for i in range(int(x.shape[0])) :
        for j in range(int(x.shape[1])) :
            cepstrum[i,j,:] = np.fft.ifft(x[i,j,:]).real
    return cepstrum
    
class Freq_Self_Att(nn.Module):
    def __init__(self) :
        super(Freq_Self_Att,self).__init__()
        self.key = nn.Linear(100,256)
        self.query = nn.Linear(100,256)
        self.scale = math.sqrt(256)            # scale을 좀 더 키워야할 거 같은데,,?    혹은 80dim softmax 값이 너무 작기 때문에 좀더 키워준다던지
        self.adapt = nn.AdaptiveAvgPool1d(100)    # max 혹은 avg+mean 시도 가능
        self.softmax = nn.Softmax(dim=2)
   #     self.scale2 = nn.AdaptiveAvgPool2d(1)
        
    def forward(self,input):
        
        x = self.adapt(input)
  #      scale2 = self.scale2(torch.sqrt(abs(input)))
        key = self.key(x)
        query = self.query(x)
        att = self.softmax(torch.matmul(key,torch.transpose(query,1,2)) / (self.scale))    #(80,80) *scale2
    #    print(att[0,0,:])
        for i in range(att.shape[2]):
            if i == 0 :
                cat = torch.sum(att[:,i,:].unsqueeze(2)*input,dim= 1).unsqueeze(1)
            else:
                cat = torch.cat([cat,torch.sum(att[:,i,:].unsqueeze(2)*input,dim= 1).unsqueeze(1)],dim=1)
#        print(att.shape)
#        print(att)
        
        
        return cat
    
class Freq_Self_Att_weight(nn.Module):
    def __init__(self) :
        super(Freq_Self_Att_weight,self).__init__()
        self.key = nn.Linear(100,256)
        self.query = nn.Linear(100,256)
        self.scale = math.sqrt(256)            # scale을 좀 더 키워야할 거 같은데,,?    혹은 80dim softmax 값이 너무 작기 때문에 좀더 키워준다던지
        self.adapt = nn.AdaptiveAvgPool1d(100)    # max 혹은 avg+mean 시도 가능
        self.softmax = nn.Softmax(dim=2)
   #     self.scale2 = nn.AdaptiveAvgPool2d(1)
        
    def forward(self,input):
        
        x = self.adapt(input)
  #      scale2 = self.scale2(torch.sqrt(abs(input)))
        key = self.key(x)
        query = self.query(x)
        att = self.softmax(torch.matmul(key,torch.transpose(query,1,2)) / (self.scale))    #(80,80) *scale2
    #    print(att[0,0,:])
        for i in range(att.shape[2]):
            if i == 0 :
                cat = torch.sum(att[:,i,:].unsqueeze(2)*input,dim= 1).unsqueeze(1)
            else:
                cat = torch.cat([cat,torch.sum(att[:,i,:].unsqueeze(2)*input,dim= 1).unsqueeze(1)],dim=1)
        freq_weight = torch.mean(cat,2,True)
        freq_weight_n = NormalizeData_1dim(freq_weight)+1e-6
        
        return input * freq_weight_n
    
class Freq_bundle_Self_Att_weight(nn.Module):
    def __init__(self) :
        super(Freq_bundle_Self_Att_weight,self).__init__()
        self.key = nn.Linear(100,256)
        self.query = nn.Linear(100,256)
        self.scale = math.sqrt(256)            # scale을 좀 더 키워야할 거 같은데,,?    혹은 80dim softmax 값이 너무 작기 때문에 좀더 키워준다던지
        self.adapt = nn.AdaptiveAvgPool2d((20,100))    # max 혹은 avg+mean 시도 가능
        self.softmax = nn.Softmax(dim=2)
   #     self.scale2 = nn.AdaptiveAvgPool2d(1)
        
    def forward(self,input):
        
        x = self.adapt(input)
  #      scale2 = self.scale2(torch.sqrt(abs(input)))
        key = self.key(x)
        query = self.query(x)
        att = self.softmax(torch.matmul(key,torch.transpose(query,1,2)) / (self.scale))    #(80,80) *scale2
    #    print(att[0,0,:])
        for i in range(att.shape[2]):
            if i == 0 :
                soft = att[:,i,:].unsqueeze(2)
                cat = torch.sum(soft.repeat_interleave(int(input.size(1)/soft.size(1)),dim=1)*input,dim= 1).unsqueeze(1)
            else:
                soft = att[:,i,:].unsqueeze(2)
                cat = torch.cat([cat,torch.sum(soft.repeat_interleave(int(input.size(1)/soft.size(1)),dim=1)*input,dim= 1).unsqueeze(1)],dim=1)
        freq_weight = torch.mean(cat,2,True)
        freq_weight_n = NormalizeData_1dim(freq_weight)+1e-6  
        freq_weight_n = freq_weight_n.repeat_interleave(int(input.size(1)/freq_weight_n.size(1)),dim=1)
        
        return input * freq_weight_n
    
    
class Time_Self_Att(nn.Module):
    def __init__(self) :
        super(Time_Self_Att,self).__init__()
        self.key = nn.Linear(80,256)
        self.query = nn.Linear(80,256)
        self.scale = math.sqrt(256)            # scale을 좀 더 키워야할 거 같은데,,?    혹은 80dim softmax 값이 너무 작기 때문에 좀더 키워준다던지
 #       self.adapt = nn.AdaptiveAvgPool1d(100)    # max 혹은 avg+mean 시도 가능
        self.softmax = nn.Softmax(dim=2)
   #     self.scale2 = nn.AdaptiveAvgPool2d(1)
        
    def forward(self,input):
        
    #    x = self.adapt(input)
  #      scale2 = self.scale2(torch.sqrt(abs(input)))
        key = self.key(input.transpose(1,2))
        query = self.query(input.transpose(1,2))
        att = self.softmax(torch.matmul(key,torch.transpose(query,1,2)) / (self.scale))    #(time,time) *scale2
    #    print(att[0,0,:])
        for i in range(att.shape[2]):
            if i == 0 :
                cat = torch.sum(att[:,i,:].unsqueeze(2)*input.transpose(1,2),dim= 1).unsqueeze(1)
            else:
                cat = torch.cat([cat,torch.sum(att[:,i,:].unsqueeze(2)*input.transpose(1,2),dim= 1).unsqueeze(1)],dim=1)
#        print(att.shape)
#        print(att)
        
        
        return cat.transpose(1,2)

class Time_Self_Att_weight(nn.Module):
    def __init__(self) :
        super(Time_Self_Att_weight,self).__init__()
        self.key = nn.Linear(80,256)
        self.query = nn.Linear(80,256)
        self.scale = math.sqrt(256)            # scale을 좀 더 키워야할 거 같은데,,?    혹은 80dim softmax 값이 너무 작기 때문에 좀더 키워준다던지
 #       self.adapt = nn.AdaptiveAvgPool1d(100)    # max 혹은 avg+mean 시도 가능
        self.softmax = nn.Softmax(dim=2)
   #     self.scale2 = nn.AdaptiveAvgPool2d(1)
        
    def forward(self,input):
        
    #    x = self.adapt(input)
  #      scale2 = self.scale2(torch.sqrt(abs(input)))
        key = self.key(input.transpose(1,2))
        query = self.query(input.transpose(1,2))
        att = self.softmax(torch.matmul(key,torch.transpose(query,1,2)) / (self.scale))    #(time,time) *scale2
    #    print(att[0,0,:])
        for i in range(att.shape[2]):
            if i == 0 :
                cat = torch.sum(att[:,i,:].unsqueeze(2)*input.transpose(1,2),dim= 1).unsqueeze(1)
            else:
                cat = torch.cat([cat,torch.sum(att[:,i,:].unsqueeze(2)*input.transpose(1,2),dim= 1).unsqueeze(1)],dim=1)
        time_weight = torch.mean(cat,2,True).transpose(1,2)
        time_weight_n = NormalizeData_2dim(time_weight)+1e-6
        
        return input*time_weight_n


'''
class Residual_Block(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim):
        super(Residual_Block,self).__init__()
        # Residual Block
        self.residual_block = nn.Sequential(
                nn.Conv2d(in_dim, mid_dim, kernel_size=3, padding=1),
                nn.ReLU,
                nn.Conv2d(mid_dim, out_dim, kernel_size=3, padding=1),
            )            
        self.relu = nn.ReLU()
                  
    def forward(self, x):
        out = self. residual_block(x)  # F(x)
        out = out + x  # F(x) + x
        out = self.relu(out)
        return out
'''
class params() :
    def __init__(self):
#        self.lowpass = [.5, [11,12,13,14,15,16,17,18]]
        self.highpass = [.5, [78,79,80,81,82,83,84,85]]
#        self.randfilter = [3, [18,19,20,21,22,23]]
        
        self.lowpass = False
#        self.highpass = False
        self.randfilter = False
        

class FFM() :
    def __init__(self) :
        param = params()
        self.lowpass = param.lowpass 
        self.highpass = param.highpass 
        self.randfilter = param.randfilter
    def forward(self,input) :
        '''input : (B,F,T)'''
        Xn = input
        h = input.shape[1]
        if self.lowpass :
            uv, lp = self.lowpass
#            dec1 = np.random.choice(2, size = input.shape[0])
            for i in range(input.shape[0]) :
                loc1 = np.random.choice(lp, size = 1)[0]
                Xn[i,:loc1,:] = 0
        if self.highpass :
            uv, hp = self.highpass
#            dec1 = np.random.choice(2, size = input.shape[0])
            for i in range(input.shape[0]) :
                loc1 = np.random.choice(hp, size = 1)[0]
                Xn[i,loc1:,:] = 0
        if self.randfilter :                
            raniter, ranf = self.randfilter
            dec1 = np.random.choice(raniter, size = input.shape[0])
            for i in range(input.shape[0]) :
                if dec1[i] > 0 :
                    for j in range(dec1[i]) :
                        b1 = np.random.choice(ranf, size = 1)[0]
                        loc1 = np.random.choice(h - b1, size = 1)[0]
                        Xn[i, loc1:(loc1 + b1 - 1), :] = 0
        return Xn

                
                
                
class Channel_Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0),-1,x.size(3))
    
class CNN(nn.Module) :
    def __init__(self) :
        super(CNN,self).__init__()
        self.conv2d_half_freq0 = nn.Sequential(
                                            nn.Conv2d(in_channels = 1,out_channels = 128,kernel_size = (3,3),
                                                     stride = (2,1),padding = (1,1)),
                                            nn.ReLU(),
                                            nn.BatchNorm2d(128))
        self.residual_block0 = nn.Sequential(
                                            nn.Conv2d(128, 32, kernel_size=3, padding=1),
                                            nn.ReLU(),
                                            nn.Conv2d(32, 128, kernel_size=3, padding=1)) 
        self.residual_block1 = nn.Sequential(
                                            nn.Conv2d(128, 32, kernel_size=3, padding=1),
                                            nn.ReLU(),
                                            nn.Conv2d(32, 128, kernel_size=3, padding=1))
        self.conv2d_half_freq1 = nn.Sequential(
                                            nn.Conv2d(in_channels = 128,out_channels = 128,kernel_size = (3,3),
                                                     stride = (2,1),padding = (1,1)),
                                            nn.ReLU(),
                                            nn.BatchNorm2d(128))
        self.flatten = Channel_Flatten()
        self.relu = nn.ReLU()
        
    def forward(self,input) :
        x = input.unsqueeze(1)
        
        x = self.conv2d_half_freq0(x)
        copy_x0 = x
        
        x = self.residual_block0(x)
        x = x + copy_x0
        x = self.relu(x)
        copy_x1 = x
        
        x = self.residual_block1(x)
        x = x + copy_x1
        x = self.relu(x)
        
        x = self.conv2d_half_freq1(x)
        out = self.flatten(x)
        
        return out
        



def NormalizeData_2dim(x):
    mi,_ = torch.min(x,dim=2,keepdim=True)
    ma,_ = torch.max(x,dim=2,keepdim=True)
    return (x - mi)/(ma - mi)
def NormalizeData_0dim(x):
    mi,_ = torch.min(x,dim=0,keepdim=True)
    ma,_ = torch.max(x,dim=0,keepdim=True)
    return (x - mi)/(ma - mi)
def NormalizeData_1dim(x):
    mi,_ = torch.min(x,dim=1,keepdim=True)
    ma,_ = torch.max(x,dim=1,keepdim=True)
    return (x - mi)/(ma - mi)


class stack_freq_importance(nn.Module):
    def __init__(self, freq_len, reduction = 8, pool_types = ['mean','max']):
        super(stack_freq_importance,self).__init__()
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
        freq_att_sum = freq_att_sum.mean(dim=0,keepdim=True)   
        scale = torch.sigmoid( freq_att_sum ).unsqueeze(2)
#        print('att scale : ',scale.shape)
        return scale

class freq_weight(nn.Module):
    def __init__(self, freq_len,reduction=8):
        super(freq_weight,self).__init__()
        self.linear = nn.Sequential(
                    nn.Linear(freq_len, freq_len // reduction),
                    nn.ReLU(),                                          
                    nn.Linear(freq_len // reduction, freq_len)
                    )
    def forward(self,input):
        x = input.permute(0,2,1)
        w = self.linear(x)
        w = NormalizeData(w)
        w = w.permute(0,2,1)
        
        x = input*w
        
        return x
        





class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0),-1)
    
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

class Local_Time_Freq_attention(nn.Module):
    def __init__(self,time, local_time, freq_len, reduction = 8, pool_types = ['avg','max']):
        super(Local_Time_Freq_attention,self).__init__()
        '''input : (B,F,T)'''
        self.freq_len = freq_len
        self.time = time
        self.local_time = local_time
        self.n_local = time//local_time
        self.n_pad = time - (self.n_local*local_time)
        

        self.mlp_freq = nn.Sequential(
            Flatten(),
            nn.Linear(freq_len, freq_len // reduction),
            nn.ReLU(),                                          
            nn.Linear(freq_len // reduction, freq_len)
            )
        
        self.mlp_time = nn.Sequential(
            Flatten(),
            nn.Linear(self.n_local, self.n_local // 2),
            nn.ReLU(),                                          
            nn.Linear(self.n_local // 2, self.n_local)
            )
        self.pool_types = pool_types
        
    def forward(self, x):
        freq_att_sum = None
        time_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
#                print('before att : ',x.shape)
                avg_pool = F.avg_pool2d( x, (1, x.size(2)), stride=(1, x.size(2)))
#                print('after att : ',avg_pool.shape)
                freq_att_raw = self.mlp_freq( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (1, x.size(2)), stride=(1, x.size(2)))
                freq_att_raw = self.mlp_freq( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (1, x.size(2)), stride=(1, x.size(2)))
                freq_att_raw = self.mlp_freq( lp_pool )
            elif pool_type=='lse':
                lse_pool = logsumexp_2d(x)
                freq_att_raw = self.mlp_freq( lse_pool )
                
            if freq_att_sum == None :
                freq_att_sum = freq_att_raw
            else :
                freq_att_sum = freq_att_sum + freq_att_raw
                
        for pool_type in self.pool_types:
            if pool_type=='avg':
#                print('before att : ',x.shape)
                avg_pool = F.avg_pool2d( x, (self.freq_len, self.local_time), stride=(self.freq_len, self.local_time))
#                print('after att : ',avg_pool.shape)
                time_att_raw = self.mlp_time( avg_pool )
          
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (self.freq_len, self.local_time), stride=(self.freq_len, self.local_time))
                time_att_raw = self.mlp_time( max_pool )
                
            if time_att_sum == None :
                time_att_sum = time_att_raw
            else :
                time_att_sum = time_att_sum + time_att_raw
                
        scale_freq = torch.sigmoid( freq_att_sum ).unsqueeze(2).expand_as(x)
        scale_time = F.pad(torch.sigmoid( time_att_sum ).repeat_interleave(self.local_time,dim=1).reshape(-1,1,self.local_time*self.n_local).repeat_interleave(self.freq_len,dim=1),(0,self.n_pad),'constant',1)
        
        scale = F.normalize(scale_freq * scale_time,dim=2)
        
        return x * scale

class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), 
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            # nn.BatchNorm1d(bottleneck), # I remove this layer
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0), 
            nn.Sigmoid(),
            )

    def forward(self, input):
        x = self.se(input)
        return input * x

class Bottle2neck(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale = 8):
        super(Bottle2neck, self).__init__()
        width       = int(math.floor(planes / scale))
        self.conv1  = nn.Conv1d(inplanes, width*scale, kernel_size=1)
        self.bn1    = nn.BatchNorm1d(width*scale)
        self.nums   = scale -1
        convs       = []
        bns         = []
        num_pad = math.floor(kernel_size/2)*dilation
        for i in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            bns.append(nn.BatchNorm1d(width))
        self.convs  = nn.ModuleList(convs)
        self.bns    = nn.ModuleList(bns)
        self.conv3  = nn.Conv1d(width*scale, planes, kernel_size=1)
        self.bn3    = nn.BatchNorm1d(planes)
        self.relu   = nn.ReLU()
        self.width  = width
        self.se     = SEModule(planes)
  #      self.fa     = Freq_attention(planes)
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i==0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(sp)
            sp = self.bns[i](sp)
            if i==0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]),1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)
     #   out = self.fa(out)
        out = self.se(out)
        out += residual
        return out 

class FbankAug(nn.Module):

    def __init__(self, freq_mask_width = (0, 8), time_mask_width = (0, 10)):
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width
        super().__init__()

    def mask_along_axis(self, x, dim):
        original_size = x.shape
        batch, fea, time = x.shape
        if dim == 1:
            D = fea
            width_range = self.freq_mask_width
        else:
            D = time
            width_range = self.time_mask_width

        mask_len = torch.randint(width_range[0], width_range[1], (batch, 1), device=x.device).unsqueeze(2)
        mask_pos = torch.randint(0, max(1, D - mask_len.max()), (batch, 1), device=x.device).unsqueeze(2)
        arange = torch.arange(D, device=x.device).view(1, 1, -1)
        mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))
        mask = mask.any(dim=1)

        if dim == 1:
            mask = mask.unsqueeze(2)
        else:
            mask = mask.unsqueeze(1)
            
        x = x.masked_fill_(mask, 0.0)
        return x.view(*original_size)

    def forward(self, x):    
        x = self.mask_along_axis(x, dim=2)
        x = self.mask_along_axis(x, dim=1)
        return x


    
class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)
    
    
'''    
class freq_weight(nn.Module):
    def __init__(self,freq_len,reduction = 8) :
        super(freq_weight,self).__init__()
        self.freq_len = freq_len
        self.mlp = nn.Sequential(
#            Flatten(),
            nn.Linear(freq_len, freq_len // reduction),
            nn.ReLU(),                                          
            nn.Linear(freq_len // reduction, freq_len)
            )
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x) :

        score = torch.diagonal(torch.matmul(x,torch.transpose(x,1,2)),dim1=1,dim2=2)
        score_mlp = self.mlp(score)

        score_scale = self.softmax(score_mlp).reshape(-1,self.freq_len,1)
        x = score_scale * x 
        return x
'''    
    
    
class ECAPA_TDNN(nn.Module):

    def __init__(self, C):

        super(ECAPA_TDNN, self).__init__()

  #      self.torchmfcc = torch.nn.Sequential(
  #          PreEmphasis(),            
  #          torchaudio.transforms.MFCC(sample_rate = 16000, n_mfcc= 40, dct_type= 2, norm= 'ortho', log_mels= False),
  #      )
        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),            
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
                                                 f_min = 20, f_max = 7600, window_fn=torch.hamming_window, n_mels=80),
            )
        self.specaug = FbankAug() # Spec augmentation
   #     self.cnn = CNN()
   #     self.att = Freq_Self_Att()
   #     self.att = Time_Self_Att()
   #     self.att = Freq_Self_Att_weight()
   #     self.att = Time_Self_Att_weight()
   #     self.att = Freq_bundle_Self_Att_weight()
   #     self.att = Time_bundle_Self_Att_weight()
   #     self.sfi = stack_freq_importance(80)
#        self.weight0 = freq_weight(80)
#        self.weight1 = freq_weight(2560)
#        self.fw = freq_weight(80)
#        self.tf_att0 = Local_Time_Freq_attention(202,10,80)
#        self.tf_att1 = Local_Time_Freq_attention(202,10,C)
#        self.tf_att2 = Local_Time_Freq_attention(202,20,C)
#        self.tf_att3 = Local_Time_Freq_attention(202,20,C)
#        self.tf_att4 = Local_Time_Freq_attention(202,20,C)
 #       self.freq0 = Freq_attention(80)
 #       self.freq1 = Freq_attention(C)
        self.conv1  = nn.Conv1d(80, C, kernel_size=5, stride=1, padding=2)  # 2560
        self.relu   = nn.ReLU()
        self.bn1    = nn.BatchNorm1d(C)
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        # I fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
        self.layer4 = nn.Conv1d(3*C, 1536, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(), # I add this layer
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
            )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, 192)
        self.bn6 = nn.BatchNorm1d(192)
#        self.maxpool = nn.AdaptiveMaxPool1d(1)
#        self.avgpool = nn.AdaptiveAvgPool1d(1)

    

    def forward(self, x, aug,train=True):  #weight,
        
        if train==False:
            with torch.no_grad():
                x = self.torchfbank(x)+1e-6
                x = x.log()   
                x = x - torch.mean(x, dim=-1, keepdim=True)   ## ????
                if x.dim() == 3 :
                    cepstrum = torch.empty((int(x.shape[0]),int(x.shape[1]),int(x.shape[2])))
                    for i in range(int(x.shape[0])) :
                        for j in range(int(x.shape[1])) :
                            cepstrum[i,j,:] = torch.fft.ifft(x[i,j,:]).real
                    
                elif x.dim() == 2:
                    cepstrum = torch.empty((int(x.shape[0]),int(x.shape[1])))
                    for j in range(int(x.shape[0])) :
                        cepstrum[j,:] = torch.fft.ifft(x[j,:]).real
                                   
                x = cepstrum.cuda()         
                if aug == True:
                    x = self.specaug(x)
        else :
            with torch.no_grad():
                if aug == True :
                    x = self.specaug(x)
                                   
   #     x = self.freq0(x)
#        x = self.weight0(x)
   #     x = self.att(x)
   #     x = self.cnn(x)
        
#        x = self.weight1(x)
#        print(x.shape)
        x = self.conv1(x)
   #     x = self.freq1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x1 = self.layer1(x)  
        
        x2 = self.layer2(x+x1)            # x1만
#        x2 = self.layer2(x1)                # x1만

        x3 = self.layer3(x+x1+x2)         # x2만 
#        x3 = self.layer3(x2)              # x2만 


        x = self.layer4(torch.cat((x1,x2,x3),dim=1))
        x = self.relu(x)

        t = x.size()[-1]

        global_x = torch.cat((x,torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)
        
        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-4) )

        x = torch.cat((mu,sg),1)
        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)
   #     if train == True :
    #        return x,we
        return x