import torch
import torch.nn as nn 
import torch.nn.functional as F
#from torch import Generator

from typing import Any
#from collections import OrderedDict

from ..prixfixe import CoreBlock
#from .add_blocks import SELayer
#from .utils import initialize_weights

class GLULayer(nn.Module):
    def __init__(self, dim):
        super(GLULayer, self).__init__()
        self.dim = dim
        self.sig = nn.Sigmoid()

    def forward(self, x):
        out,gate = torch.chunk(x, 2, dim = self.dim)
        return out * self.sig(gate)
    

class SwiGLULayer(nn.Module):
    def __init__(self, dim):
        super(SwiGLULayer, self).__init__()
        self.dim = dim
        self.swish = nn.SiLU() # same as swish

    def forward(self, x):
        out, gate = torch.chunk(x, 2, dim = self.dim)
        return out * self.swish(gate)


class FeedForwardSwiGLU(nn.Module):
    def __init__(self, embedding_dim, mult=4, rate = 0.0, use_bias = False):
        super(FeedForwardSwiGLU, self).__init__()
        swiglu_out = int(embedding_dim * mult/2)
        self.layernorm = nn.LayerNorm(embedding_dim,eps = 1e-6)
        self.linear1 = nn.Linear(embedding_dim,embedding_dim * mult, bias = use_bias)
        self.swiglulayer = SwiGLULayer(dim = 1)
        self.drop = nn.Dropout(rate)
        self.linear2 = nn.Linear(swiglu_out,embedding_dim, bias = use_bias)

    def forward(self, inputs):
        x = self.layernorm(inputs.transpose(1,2)) # 차원바뀌고 채널 dim=2
        x = self.linear1(x) 
        x = self.swiglulayer(x.transpose(1,2)) # 또 차원 바뀌고 채널 dim =1
        x = self.drop(x)
        x = self.linear2(x.transpose(1,2)) # 차원 바뀌고 채널 dim=2
        out = self.drop(x.transpose(1,2)) # 차원 또 바뀌고 채널 dim =1
        return out


class ConformerSASwiGLULayer(nn.Module):
    def __init__(self, embedding_dim,  ff_mult = 4, kernel_size = 15, rate = 0.2, num_heads = 4, use_bias = False):
        super(ConformerSASwiGLULayer, self).__init__()
        self.ff1 = FeedForwardSwiGLU(embedding_dim = embedding_dim, mult = ff_mult, rate = rate, use_bias = use_bias)
        self.layernorm1 = nn.LayerNorm(embedding_dim,eps = 1e-6)
        self.conv = nn.Sequential(   
          nn.Conv1d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=kernel_size, groups=embedding_dim, padding='same'),
          nn.Conv1d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=1, padding='same'),
          nn.ReLU(),
          nn.Dropout(rate),
        )
        self.layernorm2 = nn.LayerNorm(embedding_dim,eps = 1e-6)    
        self.attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads,batch_first=True)
        self.ff2 = FeedForwardSwiGLU(embedding_dim = embedding_dim, mult = ff_mult, rate = rate, use_bias = use_bias)

    def forward(self, x):
        x = x.float()
        x = x + 0.5 * self.ff1(x)
        x = self.layernorm1(x.transpose(1,2)) #채널 dim = 2
        x = x + self.conv(x.transpose(1, 2)).transpose(1, 2) # output 채널 dim = 2
        x = self.layernorm2(x)
        x = x + self.attn(x, x, x)[0]
        x = x.transpose(1,2) + 0.5 * self.ff2(x.transpose(1,2))
        return x
    
    

class UnlockDNA_CoreBlock(CoreBlock):
    def __init__(
        self,
        in_channels: int=512,
        out_channels: int=512,
        seqsize: int=232, # for compatibity. Isn't used by block itself
    ):
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         seqsize=seqsize)
        
        self.n_blocks = 4

        self.blocks = nn.ModuleList([ConformerSASwiGLULayer(embedding_dim = in_channels,
                                    kernel_size = 15, rate = 0.1, num_heads = 8) for _ in range(self.n_blocks)])

        
    def forward(self, x): 
#        print('core_input_shape : ',x.shape)
        for i in range(self.n_blocks) :
            x = self.blocks[i](x)
#        print('core_output_shape : ',x.shape)
        return x   # output :(b,512,232)
    
#    def weights_init(self, generator: Generator) -> None:
#        self.apply(lambda x: initialize_weights(x, generator))
    
