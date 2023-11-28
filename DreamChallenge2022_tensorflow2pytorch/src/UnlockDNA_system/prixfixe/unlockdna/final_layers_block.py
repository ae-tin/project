from typing import Any 
import torch
import torch.nn as nn 
import torch.nn.functional as F
#from torch import nn, Generator

from ..prixfixe import FinalLayersBlock
#from .utils import initialize_weights




class UnlockDNA_FinalLayersBlock(FinalLayersBlock):
    def __init__(
        self,
        in_channels: int=512,
        seqsize: int=232 # for compatibity. Isn't used by block itself
    ):
        super().__init__(in_channels=in_channels,
                         seqsize=seqsize)
        
        self.n_positions = 200
        self.num_projectors = 32
        input_dim = 6
        
        self.dropout = nn.Dropout(0.1)
        self.expression_dense = nn.Linear(in_channels,1)
        self.nucleotide_dense = nn.Linear(in_channels,input_dim)

    def forward(self, x): 
        
#        print('final_input_shape : ',x.shape)
        
        x = x.transpose(1,2)
        expression = x[:,:self.num_projectors,:]
        x = x[:, -self.n_positions:, :]

        expression = self.dropout(expression)
        expression = self.expression_dense(expression)
        expression = torch.mean(expression, 1)

        x = self.nucleotide_dense(x)
#        print('final_seq_reproduce_shape : ',x.transpose(1,2).shape)
#        print('final_output_shape : ',expression.shape)
        return expression, x.transpose(1,2)
'''
    def train_step(self, batch: dict[str, Any]):
        x = batch["x"].to(self.device)
       
        x = self.mapper(x)
        x = F.adaptive_avg_pool1d(x, 1)
        x = x.squeeze(2)
        logprobs = F.log_softmax(x, dim=1) 
        x = F.softmax(x, dim=1)
        score = (x * self.bins).sum(dim=1)
        
        if "y_probs" in batch: # classification
            y_probs = batch["y_probs"].to(self.device)
            loss = self.classification_criterion(logprobs, y_probs)
        else: # regression
            y = batch["y"].to(self.device)
            loss = self.regression_criterion(score, y.squeeze(-1))
            
        return score, loss
    
    def weights_init(self, generator: Generator) -> None:
        self.apply(lambda x: initialize_weights(x, generator))
'''
