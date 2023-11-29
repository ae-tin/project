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

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim,**kwargs):
        super().__init__()

        # attention map
        self.adpt_max = nn.AdaptiveAvgPool1d(in_dim)
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = self._init_new_params(out_dim, 1)

        # project
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)

        # batch norm
        self.bn = nn.BatchNorm1d(out_dim)

        # dropout for inputs
        self.input_drop = nn.Dropout(p=0.2)

        # activate
        self.act = nn.SELU(inplace=True)

        # temperature
        self.temp = 1.
        if "temp" in kwargs:
            self.temp = kwargs["temp"]

    def forward(self, x):
        '''
        x   :(#bs, #node, #dim)
        '''
        # apply input dropout
        x = self.input_drop(x)
        x = self.adpt_max(x)

        # derive attention map
        att_map = self._derive_att_map(x)

        
        
        # projection
#        x = self._project(x, att_map)

        # apply batch norm
#        x = self._apply_BN(x)
#        x = self.act(x)
        return torch.matmul(att_map,x)

    def _pairwise_mul_nodes(self, x):
        '''
        Calculates pairwise multiplication of nodes.
        - for attention map
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, #dim)
        '''

        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)

        return x * x_mirror

    def _derive_att_map(self, x):
        '''
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, 1)
        '''
        att_map = self._pairwise_mul_nodes(x)
#        print('pairwise_mul_nodes: ',att_map.shape)
        # size: (#bs, #node, #node, #dim_out)
        att_map = torch.tanh(self.att_proj(att_map))
        # size: (#bs, #node, #node, 1)
        att_map = torch.matmul(att_map, self.att_weight)

        # apply temperature
        att_map = att_map / self.temp

        att_map = F.softmax(att_map, dim=-2).squeeze(-1)
        
        return att_map

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)

        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)

        return x

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out

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

class Conv_Freq_Weight(nn.Module):
    def __init__(self, freq_len, pool_len, reduction = 8, pool_types = ['mean','max']):
        super(Conv_Freq_Weight,self).__init__()
        '''input : (B,F,T)'''
        self.freq_len = freq_len
        self.adapt_max = nn.AdaptiveMaxPool1d(pool_len)
        self.adapt_avg = nn.AdaptiveAvgPool1d(pool_len)
        self.conv1 = nn.Conv1d(freq_len, freq_len, kernel_size = pool_len, stride=pool_len, padding=0, dilation=1, groups=freq_len, bias=True) # relu?
        self.conv2 = nn.Conv1d(freq_len, freq_len, kernel_size = pool_len, stride=pool_len, padding=0, dilation=1, groups=freq_len, bias=True)
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
                avg_pool = self.adapt_avg(x)
                conv_avg = self.conv1(avg_pool)
                conv_avg = conv_avg.squeeze(2)
                freq_att_raw = self.mlp( conv_avg )
            elif pool_type=='max':
                max_pool = self.adapt_max(x)
                conv_max = self.conv2(max_pool)
                conv_max = conv_max.squeeze(2)
                freq_att_raw = self.mlp( conv_max )
            
                
            if freq_att_sum == None :
                freq_att_sum = freq_att_raw
            else :
                freq_att_sum = freq_att_sum + freq_att_raw

        scale = torch.sigmoid( freq_att_sum ).unsqueeze(2).expand_as(x)
        return x * scale

class Sigma_Freq_Masking(nn.Module):
    def __init__(self,nsigma):
        super(Sigma_Freq_Masking,self).__init__()
        self.nsigma = nsigma
        
    def forward(self, input) :
        
        one = torch.ones(input.shape).cuda()
        std = torch.std(input,1,unbiased=True,keepdim=True)
        avg = torch.mean(input,1,True)
        cut = ( input > ( one * avg - self.nsigma*std ) ) * (1) +1e-5
        
        return input*cut

class Self_Att_Bundle_Freq_weight(nn.Module):
    def __init__(self, freq_len, reduction = 8, pool_types = ['mean','max']):
        super(Self_Att_Bundle_Freq_weight,self).__init__()
        self.key = nn.Linear(100,256)
        self.query = nn.Linear(100,256)
        self.scale = math.sqrt(256)            # scale을 좀 더 키워야할 거 같은데,,?    혹은 80dim softmax 값이 너무 작기 때문에 좀더 키워준다던지
        self.adapt = nn.AdaptiveAvgPool2d((20,100))    # max 혹은 avg+mean 시도 가능
        self.softmax = nn.Softmax(dim=2)
        self.freq_len = freq_len
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(freq_len, freq_len // reduction),
            nn.ReLU(),                                          
            nn.Linear(freq_len // reduction, freq_len)
            )
        self.pool_types = pool_types
  
        
    def forward(self,input):
        
        x = self.adapt(input)
        key = self.key(x)
        query = self.query(x)
        att = self.softmax(torch.matmul(key,torch.transpose(query,1,2)) / (self.scale))    #(80,80) *scale2
        input_att = torch.matmul(att.repeat_interleave(int(input.size(1)/att.size(1)),dim=1).repeat_interleave(int(input.size(1)/att.size(2)),dim=2),input)
        
        freq_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='mean':
                avg_pool = F.avg_pool2d( input_att, (1, input_att.size(2)), stride=(1, input_att.size(2)))
                freq_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( input_att, (1, input_att.size(2)), stride=(1, input_att.size(2)))
                freq_att_raw = self.mlp( max_pool )
                
            if freq_att_sum == None :
                freq_att_sum = freq_att_raw
            else :
                freq_att_sum = freq_att_sum + freq_att_raw

        scale = torch.sigmoid( freq_att_sum ).unsqueeze(2).expand_as(input)
        return input * scale

class Self_Att_Freq_weight(nn.Module):
    def __init__(self, freq_len, reduction = 8, pool_types = ['mean','max']):
        super(Self_Att_Freq_weight,self).__init__()
        self.key = nn.Linear(100,256)
        self.query = nn.Linear(100,256)
        self.scale = math.sqrt(256)            # 
        self.adapt = nn.AdaptiveAvgPool1d(100)   
        self.softmax = nn.Softmax(dim=2)
        self.freq_len = freq_len
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(freq_len, freq_len // reduction),
            nn.ReLU(),                                          
            nn.Linear(freq_len // reduction, freq_len)
            )
        self.pool_types = pool_types
  
        
    def forward(self,input):
        
        x = self.adapt(input)
        key = self.key(x)
        query = self.query(x)
        att = self.softmax(torch.matmul(key,torch.transpose(query,1,2)) / (self.scale))    #(80,80) *scale2
        input_att = torch.matmul(att,input)
        
        freq_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='mean':
                avg_pool = F.avg_pool2d( input_att, (1, input_att.size(2)), stride=(1, input_att.size(2)))
                freq_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( input_att, (1, input_att.size(2)), stride=(1, input_att.size(2)))
                freq_att_raw = self.mlp( max_pool )
                
            if freq_att_sum == None :
                freq_att_sum = freq_att_raw
            else :
                freq_att_sum = freq_att_sum + freq_att_raw

        scale = torch.sigmoid( freq_att_sum ).unsqueeze(2).expand_as(input)
        return input * scale
        





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
  #      self.cfw    = Conv_Freq_Weight(planes,50)
  #      self.fa     = Freq_attention(planes)
  #      self.safw   = Self_Att_Freq_weight(planes)
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
     #   out = self.safw(out)
     #   out = self.cfw(out)
        out = self.se(out)
        out += residual
        return out 
class Bottle2neck2(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale = 8):
        super(Bottle2neck2, self).__init__()
        width       = int(math.floor(planes / scale))
        self.conv1  = nn.Conv1d(inplanes, width*scale, kernel_size=1)
        self.bn1    = nn.BatchNorm1d(width*scale)
        self.nums   = scale 
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
  #      self.se     = SEModule(planes)
        self.fa     = Freq_attention(planes)
  #      self.safw   = Self_Att_Freq_weight(planes)
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
 #       out = torch.cat((out, spx[self.nums]),1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)
        out = self.fa(out)
     #   out = self.safw(out)
     #   out = self.se(out)
        out += residual
        return out
class Bottle2neck_origin(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale = 8):
        super(Bottle2neck_origin, self).__init__()
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
  #      self.se     = SEModule(planes)
        self.fa     = Freq_attention(planes)
  #      self.safw   = Self_Att_Freq_weight(planes)
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i==0:
                sp = spx[i+1]
            else:
                sp = sp + spx[i+1]
            sp = self.convs[i](sp)
            sp = self.relu(sp)
            sp = self.bns[i](sp)
            if i==0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = torch.cat((spx[0], out),1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)
        out = self.fa(out)
     #   out = self.safw(out)
     #   out = self.se(out)
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

        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),            
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
                                                 f_min = 20, f_max = 7600, window_fn=torch.hamming_window, n_mels=80))
        self.torchfbank2 = torch.nn.Sequential(
            PreEmphasis(),            
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=800, win_length=800, hop_length=320, \
                                                 f_min = 20, f_max = 7600, window_fn=torch.hamming_window, n_mels=80))
        self.torchfbank3 = torch.nn.Sequential(
            PreEmphasis(),            
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=1600, win_length=1600, hop_length=640, \
                                                 f_min = 20, f_max = 7600, window_fn=torch.hamming_window, n_mels=80))
        self.specaug = FbankAug() # Spec augmentation

 #       self.cnn = CNN()
 #       self.freq1 = Freq_attention(80)
 #       self.freq2 = Freq_attention(80)
 #       self.freq3 = Freq_attention(80)

        self.conv1  = nn.Conv1d(80, C, kernel_size=5, stride=1, padding=2)  # 2560 #80
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
        
        self.conv1_2  = nn.Conv1d(80, C, kernel_size=5, stride=1, padding=2)  # 2560 #80
        self.relu_2   = nn.ReLU()
        self.bn1_2    = nn.BatchNorm1d(C)
        self.layer1_2 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2_2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3_2 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        # I fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
        self.layer4_2 = nn.Conv1d(3*C, 1536, kernel_size=1)
        self.attention_2 = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(), # I add this layer
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
            )
        
        self.conv1_3  = nn.Conv1d(80, C, kernel_size=5, stride=1, padding=2)  # 2560 #80
        self.relu_3   = nn.ReLU()
        self.bn1_3    = nn.BatchNorm1d(C)
        self.layer1_3 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2_3 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3_3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        # I fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
        self.layer4_3 = nn.Conv1d(3*C, 1536, kernel_size=1)
        self.attention_3 = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(), # I add this layer
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
            )
        
        self.bn5 = nn.BatchNorm1d(3072*3)
        self.fc6 = nn.Linear(3072*3, 192)
        self.bn6 = nn.BatchNorm1d(192)
#        self.maxpool = nn.AdaptiveMaxPool1d(1)
#        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x, aug,train=True):  #weight,
        if train==False:
            with torch.no_grad():
                x_1 = self.torchfbank(x)+1e-6
                x_2 = self.torchfbank2(x)+1e-6
                x_3 = self.torchfbank3(x)+1e-6
                
                x_1 = x_1.log()
                x_2 = x_2.log()
                x_3 = x_3.log()
                
                x_1 = x_1 - torch.mean(x_1, dim=-1, keepdim=True)   ## ????
                x_2 = x_2 - torch.mean(x_2, dim=-1, keepdim=True) 
                x_3 = x_3 - torch.mean(x_3, dim=-1, keepdim=True) 
                
                if aug == True:
                    x_1 = self.specaug(x_1)
                    x_2 = self.specaug(x_2)
                    x_3 = self.specaug(x_3)
        else :
            if aug== True :
                with torch.no_grad():
                    x_1 = self.specaug(x[0])
                    x_2 = self.specaug(x[1])
                    x_3 = self.specaug(x[2])
            else :
                x_1 = x[0]
                x_2 = x[1]
                x_3 = x[2]
        
   #     x_1 = self.freq1(x_1)
   #     x_2 = self.freq2(x_2)
   #     x_3 = self.freq3(x_3)
    
   #     x = self.cnn(x)
        x = self.conv1(x_1)
        x_2 = self.conv1_2(x_2)
        x_3 = self.conv1_3(x_3)
   #     x = self.freq1(x)
        x = self.relu(x)
        x_2 = self.relu_2(x_2)
        x_3 = self.relu_3(x_3)
        
        x = self.bn1(x)
        x_2 = self.bn1_2(x_2)
        x_3 = self.bn1_3(x_3)

        x1 = self.layer1(x) 
        x1_2 = self.layer1_2(x_2)
        x1_3 = self.layer1_3(x_3)
        
#        x2 = self.layer2(x+x1)            # x1만
        x2 = self.layer2(x1)   
        x2_2 = self.layer2_2(x1_2) 
        x2_3 = self.layer2_3(x1_3) 

#        x3 = self.layer3(x+x1+x2)         # x2만 
        x3 = self.layer3(x2)
        x3_2 = self.layer3_2(x2_2) 
        x3_3 = self.layer3_3(x2_3) 


        x = self.layer4(torch.cat((x1,x2,x3),dim=1))
        x_2 = self.layer4_2(torch.cat((x1_2,x2_2,x3_2),dim=1))
        x_3 = self.layer4_3(torch.cat((x1_3,x2_3,x3_3),dim=1))
        
        x = self.relu(x)
        x_2 = self.relu_2(x_2)
        x_3 = self.relu_3(x_3)

        t = x.size()[-1]
        t_2 = x_2.size()[-1]
        t_3 = x_3.size()[-1]

        global_x = torch.cat((x,torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)
        
        global_x_2 = torch.cat((x_2,torch.mean(x_2,dim=2,keepdim=True).repeat(1,1,t_2), torch.sqrt(torch.var(x_2,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t_2)), dim=1)
        
        global_x_3 = torch.cat((x_3,torch.mean(x_3,dim=2,keepdim=True).repeat(1,1,t_3), torch.sqrt(torch.var(x_3,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t_3)), dim=1)
        
        w = self.attention(global_x)
        w_2 = self.attention_2(global_x_2)
        w_3 = self.attention_3(global_x_3)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-4) )
        
        mu_2 = torch.sum(x_2 * w_2, dim=2)
        sg_2 = torch.sqrt( ( torch.sum((x_2**2) * w_2, dim=2) - mu_2**2 ).clamp(min=1e-4) )
        
        mu_3 = torch.sum(x_3 * w_3, dim=2)
        sg_3 = torch.sqrt( ( torch.sum((x_3**2) * w_3, dim=2) - mu_3**2 ).clamp(min=1e-4) )

        x = torch.cat((mu,sg,mu_2,sg_2,mu_3,sg_3),1)
        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)
        
        return x
    
class ECAPA_TDNN2(nn.Module):

    def __init__(self, C):

        super(ECAPA_TDNN2, self).__init__()

        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),            
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
                                                 f_min = 20, f_max = 7600, window_fn=torch.hamming_window, n_mels=80))
        self.torchfbank2 = torch.nn.Sequential(
            PreEmphasis(),            
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=800, win_length=800, hop_length=320, \
                                                 f_min = 20, f_max = 7600, window_fn=torch.hamming_window, n_mels=80))
        self.torchfbank3 = torch.nn.Sequential(
            PreEmphasis(),            
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=1600, win_length=1600, hop_length=640, \
                                                 f_min = 20, f_max = 7600, window_fn=torch.hamming_window, n_mels=80))
        self.specaug = FbankAug() # Spec augmentation
 
 #       self.freq1 = Freq_attention(80)
 #       self.freq2 = Freq_attention(80)
 #       self.freq3 = Freq_attention(80)
 #       self.cnn = CNN()

        self.conv1  = nn.Conv1d(80, C, kernel_size=5, stride=1, padding=2)  # 2560 #80
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
        
        self.conv1_2  = nn.Conv1d(80, C, kernel_size=5, stride=1, padding=2)  # 2560 #80
        self.relu_2   = nn.ReLU()
        self.bn1_2    = nn.BatchNorm1d(C)
        self.layer1_2 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2_2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3_2 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        # I fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
     
        
        self.conv1_3  = nn.Conv1d(80, C, kernel_size=5, stride=1, padding=2)  # 2560 #80
        self.relu_3   = nn.ReLU()
        self.bn1_3    = nn.BatchNorm1d(C)
        self.layer1_3 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2_3 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3_3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        # I fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
     
        
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, 192)
        self.bn6 = nn.BatchNorm1d(192)
#        self.maxpool = nn.AdaptiveMaxPool1d(1)
#        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x, aug,train=True):  #weight,
        
        if train==False:
            with torch.no_grad():
                x_1 = self.torchfbank(x)+1e-6
                x_2 = self.torchfbank2(x)+1e-6
                x_3 = self.torchfbank3(x)+1e-6
                
                x_1 = x_1.log()
                x_2 = x_2.log()
                x_3 = x_3.log()
                
                x_1 = x_1 - torch.mean(x_1, dim=-1, keepdim=True)   ## ????
                x_2 = x_2 - torch.mean(x_2, dim=-1, keepdim=True) 
                x_3 = x_3 - torch.mean(x_3, dim=-1, keepdim=True) 
                
                if aug == True:
                    x_1 = self.specaug(x_1)
                    x_2 = self.specaug(x_2)
                    x_3 = self.specaug(x_3)
        else :
            if aug== True :
                with torch.no_grad():
                    x_1 = self.specaug(x[0])
                    x_2 = self.specaug(x[1])
                    x_3 = self.specaug(x[2])
            else :
                x_1 = x[0]
                x_2 = x[1]
                x_3 = x[2]
        
   #     x_1 = self.freq1(x_1)
   #     x_2 = self.freq2(x_2)
   #     x_3 = self.freq3(x_3)
    
   #     x = self.cnn(x)
        x = self.conv1(x_1)
        x_2 = self.conv1_2(x_2)
        x_3 = self.conv1_3(x_3)
   #     x = self.freq1(x)
        x = self.relu(x)
        x_2 = self.relu_2(x_2)
        x_3 = self.relu_3(x_3)
        
        x = self.bn1(x)
        x_2 = self.bn1_2(x_2)
        x_3 = self.bn1_3(x_3)

        x1 = self.layer1(x) 
        x1_2 = self.layer1_2(x_2)
        x1_3 = self.layer1_3(x_3)
        
#        x2 = self.layer2(x+x1)            # x1만
        x2 = self.layer2(x1)   
        x2_2 = self.layer2_2(x1_2) 
        x2_3 = self.layer2_3(x1_3) 

#        x3 = self.layer3(x+x1+x2)         # x2만 
        x3 = self.layer3(x2)
        x3_2 = self.layer3_2(x2_2) 
        x3_3 = self.layer3_3(x2_3) 

        x = torch.cat((x1,x2,x3),dim=1)
        x_2 = torch.cat((x1_2,x2_2,x3_2),dim=1)
        x_3 = torch.cat((x1_3,x2_3,x3_3),dim=1)
        
        x = torch.cat((x,x_2,x_3),dim=2)
        
        x = self.layer4(x)
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
        
        return x