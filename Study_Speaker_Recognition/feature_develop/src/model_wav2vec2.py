

import math, torch, torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
import math
from jiwer import wer
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


                
                
class Channel_Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0),-1,x.size(3))
    



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





        
class Bottle2neck2(nn.Module):

    def __init__(self, inplanes, planes, module, kernel_size=None, dilation=None, scale = 8):
        super(Bottle2neck2, self).__init__()
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
        if module == 'fa' :
            self.module     = Freq_attention(planes)
        else :
            self.module     = SEModule(planes)
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
        out = self.module(out)
        out += residual
        return out 
#####################################################################3    
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
    
    
    
    
class ECAPA_TDNN(nn.Module):

    def __init__(self, C):

        super(ECAPA_TDNN, self).__init__()

        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),            
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
                                                 f_min = 20, f_max = 7600, window_fn=torch.hamming_window, n_mels=80),
            )

        self.specaug = FbankAug() # Spec augmentation

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
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, 192)
        self.bn6 = nn.BatchNorm1d(192)

    def forward(self, x, aug):  

        with torch.no_grad():
            x = self.torchfbank(x)+1e-6
            x = x.log()   
            x = x - torch.mean(x, dim=-1, keepdim=True)  
            if aug == True:
                x = self.specaug(x)


    
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        
        x1 = self.layer1(x)  
        
        x2 = self.layer2(x1)                

        x3 = self.layer3(x2)             


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
        return x    

class Res2Net(nn.Module):

    def __init__(self, C, module, n_layer):

        super(Res2Net, self).__init__()

        self.n_layer_ml = n_layer-1
        self.n_layer = n_layer
#        self.conv1  = nn.Conv1d(2560, C, kernel_size=5, stride=1, padding=2)  # 2560 #80
        self.relu   = nn.ReLU()
#        self.bn1    = nn.BatchNorm1d(C)

        self.bottle_1 = Bottle2neck2(1280, C, module, kernel_size=3, dilation=2, scale=8)
        if n_layer != 1 :
            self.bottle_n = nn.ModuleList([Bottle2neck2(C, C, module, kernel_size=3, dilation=i+3, scale=8) for i in range(self.n_layer_ml)])
#        if n_layer > 0 :
#            self.layer1 = Bottle2neck2(1280, C, module, kernel_size=3, dilation=2, scale=8)
#            if n_layer > 1 :
#                self.layer2 = Bottle2neck2(C, C, module, kernel_size=3, dilation=3, scale=8)
#                if n_layer > 2 :
#                    self.layer3 = Bottle2neck2(C, C, module, kernel_size=3, dilation=4, scale=8)
#                    if n_layer > 3 :
#                        self.layer4 = Bottle2neck2(C, C, module, kernel_size=3, dilation=5, scale=8)
#                        if n_layer > 4 :
#                            self.layer5 = Bottle2neck2(C, C, module, kernel_size=3, dilation=6, scale=8)
#                            if n_layer > 5 :
#                                self.layer6 = Bottle2neck2(C, C, module, kernel_size=3, dilation=7, scale=8)
        self.conv = nn.Conv1d(n_layer*C, 1536, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(), # I add this layer
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
            )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6_1 = nn.Linear(3072, 192)
        self.fc6_2 = nn.Linear(3072, 192)
        self.bn6 = nn.BatchNorm1d(192)

    def forward(self, x):#, aug,train=True): 
        
#        if self.n_layer > 0 :
#            x1 = self.layer1(x)
#            x = self.conv(x1)
#            if self.n_layer > 1 :
#                x2 = self.layer2(x1)
#                x = self.conv(torch.cat((x1,x2),dim=1))
#                if self.n_layer > 2 :
#                    x3 = self.layer3(x2)  
#                    x = self.conv(torch.cat((x1,x2,x3),dim=1))
#                    if self.n_layer > 3 :
#                        x4 = self.layer4(x3) 
#                        x = self.conv(torch.cat((x1,x2,x3,x4),dim=1))
#                        if self.n_layer > 4 :
#                            x5 = self.layer5(x4) 
#                            x = self.conv(torch.cat((x1,x2,x3,x4,x5),dim=1))
#                            if self.n_layer > 5 :
#                                x6 = self.layer6(x5) 
#                                x = self.conv(torch.cat((x1,x2,x3,x4,x5),dim=1))

        res2net_out = list()
    
#        x1 = self.bottle_1(x)
        globals()['x1'] = self.bottle_1(x)
        res2net_out.append(x1)
        if self.n_layer != 1 :
#            print(globals())
            for i,layer in enumerate(self.bottle_n) :
                globals()['x{}'.format(i+2)] = layer(globals()['x{}'.format(i+1)])
                
                res2net_out.append(globals()['x{}'.format(i+2)])
        re2net_out = tuple(res2net_out)
                                                 
        x = self.conv(torch.cat(re2net_out,dim=1))
        x = self.relu(x)

        t = x.size()[-1]

        global_x = torch.cat((x,torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)
        
        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-4) )

        x = torch.cat((mu,sg),1)
        x = self.bn5(x)
        x_1 = self.fc6_1(x)
        x_2 = self.fc6_2(x)
        x = torch.maximum(x_1,x_2)
        x = self.bn6(x)
        return x
    
class TDNN(nn.Module):

    def __init__(self, C):

        super(TDNN, self).__init__()

        self.tdnn1  = nn.Conv1d(1280, C, kernel_size=3, dilation=2)  
        self.relu   = nn.ReLU()
        self.tdnn2  = nn.Conv1d(C, C, kernel_size=3, dilation=3)

        self.fc_1 = nn.Linear(C*2, 512)
        self.fc_2 = nn.Linear(C*2, 512)
        self.bn6 = nn.BatchNorm1d(192)

    def forward(self, x):
        
                                                 
        x = self.tdnn1(x)
        x = self.relu(x)
        x = self.tdnn2(x)

        mu = torch.sum(x, dim=2)
        sd = torch.sqrt( torch.var(x, dim=2 ) ).clamp(min=1e-4) 

        x = torch.cat((mu,sd),1)
        
        x_1 = self.fc_1(x)
        x_2 = self.fc_2(x)
        x = torch.maximum(x_1,x_2)
        
        return x    
                                                 
def layer_freeze(w2v2, total_layer,frz_layer) :
    w2v2.dropout = nn.Identity()
    w2v2.lm_head = nn.Identity()
    for para in w2v2.parameters():
        para.requires_grad = False

    for name ,child in w2v2.named_children():
        if name == 'wav2vec2':
            for nam,chil in child.named_children():
                if nam == 'encoder' :
                    for na,chi in chil.named_children():
                        if na == 'layers' :
                            for i in range(total_layer):
                                if frz_layer == -1 :
                                    for para in chi.parameters():
                                        para.requires_grad = True
                                elif frz_layer <= i :
                                    for i,para in enumerate(chi[i].parameters()):
                                        para.requires_grad = True
    return w2v2                                                 
                                                 
class W2V2_SE_Res2Net_frz_xls_r_1b(nn.Module):

    def __init__(self):

        super(W2V2_SE_Res2Net_frz_xls_r_1b, self).__init__()
                                                 
        total_transformer_layers = 12
        n_frz_layers = -1                              # -1 : 모든 Transformer Layer.requires_grad = True , n : n Transformer Layer까지 frz.
        res2net_layers = 2
        
        self.w2v2 = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xls-r-1b",num_hidden_layers=total_transformer_layers)
                                                 
        self.w2v2 = layer_freeze(self.w2v2, total_transformer_layers, n_frz_layers)
                                                 
        self.res2net = Res2Net(C = 1280, module = 'se', n_layer = res2net_layers)

    def forward(self, x):
                                                 
        w2v2_out = self.w2v2(x).logits
        w2v2_out = torch.transpose(w2v2_out,1,2)
                                                 
        out = self.res2net(w2v2_out)
        
        return out

class W2V2_TDNN_frz_xls_r_1b(nn.Module):

    def __init__(self):

        super(W2V2_TDNN_frz_xls_r_1b, self).__init__()
                                                 
        total_transformer_layers = 12
        n_frz_layers = -1                              # -1 : 모든 Transformer Layer.requires_grad = True , n : n Transformer Layer까지 frz.
        
        self.w2v2 = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xls-r-1b",num_hidden_layers=total_transformer_layers)
                                                 
        self.w2v2 = layer_freeze(self.w2v2, total_transformer_layers, n_frz_layers)
                                                 
        self.TDNN = TDNN(C = 2048)
        
    def forward(self, x):
                                                 
        w2v2_out = self.w2v2(x).logits
        w2v2_out = torch.transpose(w2v2_out,1,2)
                                                 
        out = self.TDNN(w2v2_out)
        
        return out                  
                                                 
                                                 
                                                 
                                                 
                                                 
                                                 
                                                 
                                                 
                                                 
                                                 
                                                 
                                                 
                                                 
                                                 
                                                 
                                                 
                                                 
                                                 
                                                 
                                                 
                                                 