import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
#from torch import Generator

from typing import Any
from ..prixfixe import FirstLayersBlock
#from .utils import initialize_weights

class UnlockDNA_FirstLayersBlock(FirstLayersBlock):
    def __init__(
        self,
        in_channels: int=6,
        out_channels: int=512,
        seqsize: int=200  # for compatibity. Isn't used by block itself
    ):
        super().__init__(in_channels=in_channels, 
                         out_channels=out_channels, 
                         seqsize=seqsize)
        
        self.n_positions = seqsize # max_width=100 *2
        self.input_dim = in_channels  # 6
        self.kmer = 10
        self.strides = 1
        self.num_projectors = 32
        embedding_dim = out_channels  # 512
        
        
        self.pos_embedding = nn.Embedding(self.n_positions, embedding_dim)
        self.strand_embedding = nn.Embedding(2, embedding_dim) # plus/minus strands
        self.expression_embedding = nn.Linear(1,embedding_dim)
        self.kmer_dense = nn.Linear(self.input_dim*self.kmer,embedding_dim)
  

    def forward(self, x): # input = (batch, embed=6, seq=200)
#        print('input_shape : ',x.shape)
        
        batch_size = x.shape[0]
        x = x.transpose(1,2).unsqueeze(2)  # output = b,seq,em,1
        x_shape = x.shape
        fold_shape = x.unfold(1,self.kmer,self.strides).transpose(3,4).shape
        div = x_shape[1] - fold_shape[1]
        x = F.pad(x.unfold(1,self.kmer,self.strides).transpose(3,4),(0,0,0,0,0,0,0,div),'constant',0).reshape(x.shape[0],x.shape[1]//self.strides,x.shape[2],-1)
        x = x.squeeze(2).float()   # output : (b, seq= 200, embed*kmer = 60)
#        print('unfold shape : ',x.shape)
        x = self.kmer_dense(x)

        pos = torch.arange(start=0, end = self.n_positions, step=1).cuda()
        pos = pos.unsqueeze(0)
        pos = self.pos_embedding(pos.long())

        strand = torch.tensor(np.repeat([0,1], repeats = int(self.n_positions / 2))).cuda()
        strand = strand.unsqueeze(0)
        strand = self.strand_embedding(strand.long())

        x = x + pos + strand  # 채널 dim=2
#        print('sum x shape : ',x.shape)

        expression = torch.zeros((batch_size, self.num_projectors, 1)).cuda()
        expression = self.expression_embedding(expression.float())
#        print('exp shape : ',expression.shape)
#        print('x shape : ',x.shape)
        x = torch.cat([expression, x], dim = 1)
        x = x.transpose(1,2)
#        print('first_output_shape : ',x.shape)

        return x
    
#    def weights_init(self, generator: Generator) -> None:
#        self.apply(lambda x: initialize_weights(x, generator))