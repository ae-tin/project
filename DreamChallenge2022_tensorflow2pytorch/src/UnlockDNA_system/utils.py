import warnings
import math
import os
import sys
import time
from optparse import OptionParser
import collections
from collections import OrderedDict
from itertools import repeat
import json
import random
import numpy as np
import pandas as pd
from scipy import stats
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
from torchsummary import summary
from ignite.metrics import MeanSquaredError, Metric
from scipy.stats import pearsonr, spearmanr
import wandb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# train_txt_file = '/home/muntakimrafi/rafi/Codes/Chard/data/train_sequences.txt'
# X, Y = get_seq_exp(train_txt_file)
# X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.1, shuffle=True, random_state=42)

def createDataLoader(X, Y, num_workers, batch_size, shuffle, drop_last, random_seed):

    
    X=torch.tensor(X, dtype=torch.float)
    Y=torch.tensor(Y, dtype = torch.float)

    g = torch.Generator()
    g.manual_seed(random_seed)

    dataLoader = DataLoader(dataset=TensorDataset(X, Y), num_workers=num_workers, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, 
                           worker_init_fn=np.random.seed(random_seed), generator=g)
    
    return dataLoader

def seq2feature(data):
    
    A_onehot = np.array([1,0,0,0] ,  dtype=bool)
    C_onehot = np.array([0,1,0,0] ,  dtype=bool)
    G_onehot = np.array([0,0,1,0] ,  dtype=bool)
    T_onehot = np.array([0,0,0,1] ,  dtype=bool)
    N_onehot = np.array([0,0,0,0] ,  dtype=bool)

    mapper = {'A':A_onehot,'C':C_onehot,'G':G_onehot,'T':T_onehot,'N':N_onehot}
    worddim = len(mapper['A'])

    ###Make sure the length is 110bp
    for i in (range(0,len(data))):
        if (len(data[i]) > 110):
            data[i] = data[i][-110:]
        elif (len(data[i]) < 110):
            while (len(data[i]) < 110):
                data[i] = 'N'+data[i]
    transformed = np.asarray(([[mapper[k] for k in (data[i])] for i in (range(len(data)))]))
    return transformed

def get_seq_exp(train_file):
    sequences = []
    expressions = []

    with open(train_file) as f:
        lines = f.readlines()[1:]
        for j in tqdm(range(len(lines))):
            seq = lines[j].split('\t')[0]
            exp = lines[j].split('\t')[1].split('\n')[0]
            sequences.append(seq)
            expressions.append(exp)

    for i in tqdm(range(0,len(sequences))):
        if (len(sequences[i]) > 110):
            sequences[i] = sequences[i][-110:]
        if (len(sequences[i]) < 110):
            while (len(sequences[i]) < 110):
                sequences[i] = 'N'+sequences[i]
    
    X = seq2feature(sequences).transpose(0,2,1)
    Y = np.asarray(expressions).astype('float')
        
    return X, Y

class PearsonMetric(Metric):
    def __init__(self, output_transform=lambda x: x, device="cpu"):
        self._ys = []
        self._ypreds = []
        super().__init__(output_transform=output_transform, device=device)

    def reset(self):
        self._ys = []
        self._ypreds = []
        super().reset()

    def update(self, output):
        y_pred, y = output[0].cpu().numpy(), output[1].cpu().numpy()
        self._ys.append(y)
        self._ypreds.append(y_pred)
        
    def compute(self):
        y = np.concatenate(self._ys)
        y_pred = np.concatenate(self._ypreds)
        cor, _ = pearsonr(y, y_pred)
        return cor 
class SpearmanMetric(Metric):
    def __init__(self, output_transform=lambda x: x, device="cpu"):
        self._ys = []
        self._ypreds = []
        super().__init__(output_transform=output_transform, device=device)

    def reset(self):
        self._ys = []
        self._ypreds = []
        super().reset()

    def update(self, output):
        y_pred, y = output[0].cpu().numpy(), output[1].cpu().numpy()
        self._ys.append(y)
        self._ypreds.append(y_pred)
        
    def compute(self):
        y = np.concatenate(self._ys)
        y_pred = np.concatenate(self._ypreds)
        cor, _ = spearmanr(y, y_pred)
        return cor 



def train(dataloader, model, loss_fn, optimizer, scheduler, valid_loader, epochs, modelID):
    
    min_train_mse = 1
    max_train_pearson = 0
    max_train_spearman = 0
    
    train_mse =  MeanSquaredError()
    train_pearson = PearsonMetric()
    train_spearman = SpearmanMetric()       
    
#     wandb.init(project="MnM-DREAM2022", entity="muntakimrafi", config={})
#     wandb.run.name = f'{modelID}'

#     wandb.config.update({   
#         "modelID":modelID,
#         "model": model,
#         "loss_fn": loss_fn,
#         "optimizer": optimizer,
#         "epochs": epochs,
#         "scheduler": scheduler,
#         "epochs": epochs,
#     })
    
    for epoch in range(epochs):
            
        model.train()
        
        train_loss = 0
        
        start = time.time()
        train_step = 0
        for (X, y) in dataloader: 
            X, y = X.to(device), y.to(device)
    
            pred = model(X).squeeze()
            loss = loss_fn(pred, y)

            optimizer.zero_grad() 
            loss.backward()  
            optimizer.step() 
            optimizer.zero_grad()

            train_loss += loss.item()
            
            out = (pred.detach().cpu(), y.cpu())
            
            train_mse.update(out)
            train_pearson.update(out)
            train_spearman.update(out)
            train_step+=1
            # for test run, we are reducing the number of steps
            if train_step == 90:
                break
        
        train_loss /= train_step*1.0
        
        if train_mse.compute() > min_train_mse:
            torch.save(model.state_dict(), f'SavedModels/{modelID}_train_mse.pth')
            min_train_mse = train_mse.compute()
                
        if train_pearson.compute() > max_train_pearson:
            torch.save(model.state_dict(), f'SavedModels/{modelID}_train_pearson.pth')
            max_train_pearson = train_pearson.compute()

        if train_spearman.compute() < max_train_spearman:
            torch.save(model.state_dict(), f'SavedModels/{modelID}_train_spearman.pth')
            max_train_spearman = train_spearman.compute()
        
        # wandb.log({"train/train_loss": train_loss})
        # wandb.log({"train/train_mse": train_mse.compute()})
        # wandb.log({"train/train_pearson": train_pearson.compute()})
        # wandb.log({"train/train_spearman": train_spearman.compute()})
            

        min_val_mse = 1
        max_val_pearson = 0
        max_val_spearman = 0
        
        val_mse =  MeanSquaredError()
        val_pearson = PearsonMetric()
        val_spearman = SpearmanMetric() 
    
        valid_loss = 0
    
        model.eval()
        val_step = 0
        with torch.no_grad():
            for (X, y) in valid_loader:
                X, y = X.to(device), y.to(device)
    
                pred = model(X).squeeze()
                loss = loss_fn(pred, y)
                
                valid_loss += loss.item()
                out = (pred.detach().cpu(), y.cpu())
            
                val_mse.update(out)
                val_pearson.update(out)
                val_spearman.update(out)
                val_step += 1
                # for test run, we are reducing the number of steps
                if val_step == 10:
                    break
        valid_loss /= val_step*1.0
        
        if val_mse.compute() > min_val_mse:
            torch.save(model.state_dict(), f'SavedModels/{modelID}_val_mse.pth')
            min_val_mse = train_mse.compute()
                
        if val_pearson.compute() > max_val_pearson:
            torch.save(model.state_dict(), f'SavedModels/{modelID}_val_pearson.pth')
            max_val_pearson = val_pearson.compute()

        if val_spearman.compute() < max_val_spearman:
            torch.save(model.state_dict(), f'SavedModels/{modelID}_val_spearman.pth')
            max_val_spearman = val_spearman.compute()
            
        # wandb.log({"validation/val_loss": valid_loss})
        # wandb.log({"validation/val_mse": val_mse.compute()})
        # wandb.log({"validation/val_pearson": val_pearson.compute()})
        # wandb.log({"validation/val_spearman": val_spearman.compute()})
        
        template = ("Epoch {}, {} sequences, time: {}s, train Loss: {}, train mse {}, train pearson: {}, train_spearman: {}, val Loss: {}, "
                      "val mse: {}, val pearson: {}, val spearman: {}")
        print(template.format(epoch + 1, len(X) * 90, time.time()-start, train_loss,
                             train_mse.compute(), train_pearson.compute(), train_spearman.compute(), valid_loss,
                             val_mse.compute(), val_pearson.compute(), val_spearman.compute()))

        train_mse.reset()
        train_pearson.reset()
        train_spearman.reset()
        val_mse.reset()
        val_pearson.reset()
        val_spearman.reset()

    return model

class PrixFixeNet(nn.Module):
    def __init__(self, layers):
        
        super(PrixFixeNet, self).__init__()
        self.l1 = layers[0]
        self.l2 = layers[1]
        self.l3 = layers[2]
    
    def forward(self, x):
        x = self.l1(x)
        
        if self.l2 is not None:
            x = self.l2(x)
        if self.l3 is not None:
            x = self.l3(x)
        
        return x