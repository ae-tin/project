#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Import libraries and functions. You can change or remove them.
#
################################################################################
import pickle as pk
import glob
import os
import time
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import sys
import random
from tqdm import tqdm
from tools.utils import  get_class_distribution,get_class_distribution2,get_logger,  murmur_score,compute_weighted_accuracy
sys.path.append("..")
from reader.data_reader_physionet import myDataLoader, myDataset
import gc
#from torch.utils.tensorboard import SummaryWriter
from helper_code import *
import numpy as np, scipy as sp, scipy.stats, os, sys, joblib
from sklearn.impute import SimpleImputer
#from sklearn.ensemble import RandomForestClassifier
from models import *
from get_feature import *
import pickle as pk
from sklearn.metrics import f1_score
from jiwer import wer
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
import warnings
warnings.filterwarnings('ignore')

#
################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments.
#
################################################################################

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    gc.collect()
    torch.cuda.empty_cache()
    GPU_NUM = 2 # 원하는 GPU 번호 입력
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
#    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('current device : ',device)
    
    # Find data files.
    if verbose >= 1:
        print('Finding data files...')

    # Find the patient data files.
    patient_files = find_patient_files(data_folder)

    num_patient_files = len(patient_files)

    if num_patient_files==0:
        raise Exception('No data was provided.')


    murmur_classes = ['Present', 'Absent'] # ,'Unknown'
    num_murmur_classes = len(murmur_classes)

    outcome_classes = ['Abnormal', 'Normal']
    num_outcome_classes = len(outcome_classes)


    # Extract the features and labels.
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')

    features, labels = get_features(data_folder, patient_files)
#    print(type(features['mel']))
#    print(features['mel'][0].shape)
#    print(len(features['mel']))
    
    
    log_dir = './log/'
    project = 'toy0'
    logger = get_logger(log_dir + '/' + project)

    #train
    label_mur = np.empty(len(labels['murmur']))
    for i,l in enumerate(labels['murmur']) :
        label_mur[i] = int(list(l).index(1))

    label_out = np.empty(len(labels['outcome']))
    for i,l in enumerate(labels['outcome']) :
        label_out[i] = int(list(l).index(1))

    label = {'murmur':label_mur, 'outcome':label_out}
        
    dataset_train = myDataset(features, label,mode ='murmur')
    dataloader_train = myDataLoader(dataset=dataset_train,
                                    batch_size=32,
                                    shuffle=True,
                                    num_workers=0)
    all_file_train = len(dataloader_train)
    if verbose >= 1:
        print("- {} murmur training samples".format(len(dataset_train)))
        print("- {} murmur training batches".format(len(dataloader_train)))

    #################################################################

    dataset_train2 = myDataset(features, label,mode ='outcome')
    dataloader_train2 = myDataLoader(dataset=dataset_train2,
                                    batch_size=32,
                                    shuffle=True,
                                    num_workers=0)
    dataloader_train3 = myDataLoader(dataset=dataset_train2,
                                    batch_size=32,
                                    shuffle=True,
                                    num_workers=0)
    all_file_train2 = len(dataloader_train2)
    if verbose >= 1:
        print("- {} outcome training samples".format(len(dataset_train2)))
        print("- {} outcome training batches".format(len(dataloader_train2)))
        
    nnet = W2V2_LCNN(mode ='murmur').cuda()
    nnet2 = LCNN(mode ='outcome').cuda()
    softmax = nn.Softmax(dim=1)
    sig = nn.Sigmoid()

    optimizer = optim.Adam(nnet.parameters(), lr=1e-5)
    optimizer2 = optim.Adam(nnet2.parameters(), lr=1e-3)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.97)
    scheduler2 = torch.optim.lr_scheduler.ExponentialLR(optimizer2, 0.97)

    criterion = nn.BCELoss()
    class_weights = torch.tensor([3,1],dtype=torch.float)
    criterion2 = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
                                                                                ##############################
    # Train the model.
    if verbose >= 1:
        print('Training model...')
    
    for iter_ in range(100):  # args.end_iter
        start_time = time.time()
        running_loss2 = 0.0

        train_epoch_out_f1 = 0.0

        nnet2.train()

        pre_train2 = []
        label_train2 = []

        for audio_feature, data_label_torch in tqdm(dataloader_train2): #tqdm(zip(dataloader_train2,dataloader_train3)): ),(x_, y_)

            audio_feature = audio_feature.permute(1,0,2)
            audio_feature = audio_feature.unsqueeze(1)

      #      x_ = x_.permute(1,0,2)
      #      x_ = x_.unsqueeze(1)

      #      audio_feature,data_label_torch = aug(audio_feature,data_label_torch,x_,y_)

            audio_feature[audio_feature != audio_feature] = 0
            audio_feature= torch.nan_to_num(audio_feature,nan=1e-6)+1e-6

            audio_feature = audio_feature.float().cuda()

            data_label_torch = data_label_torch.cuda()
            data_label_torch = data_label_torch.long()

            outputs = nnet2(audio_feature)

            loss2 = criterion2(outputs.float().squeeze(1), data_label_torch.squeeze(1))
            output_tags = softmax(outputs)
            _,output_tags = torch.max(output_tags,dim=1)
            output_tags = output_tags.detach().cpu().numpy()
            data_label_torch = data_label_torch.data.cpu().numpy()
     #       result = np.append(output_tags,data_label_torch,axis=1)
            train_out_f1 = f1_score(data_label_torch, output_tags)

            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()

            running_loss2 += loss2.item()
            train_epoch_out_f1 += train_out_f1.item()

            pre_train2.extend(list(output_tags))
            label_train2.extend(list(data_label_torch))
        logger.info("Iteration:{0}, loss = {1:.6f}, out_f1 = {2:.3f}".format(iter_, running_loss2/all_file_train2, train_epoch_out_f1/all_file_train2))

    
    if verbose >= 1:
        print('Training murmur model...')

    for iter_ in range(20):  # args.end_iter
        start_time = time.time()
        running_loss = 0.0

        train_epoch_mur_f1 = 0.0

        nnet.train()

        pre_train = []
        label_train = []

        for audio_feature, data_label_torch in tqdm(dataloader_train): 

            torch.autograd.set_detect_anomaly(True)
            audio_feature = audio_feature.permute(1,0)
            audio_feature[audio_feature != audio_feature] = 0
            audio_feature= torch.nan_to_num(audio_feature,nan=1e-6)+1e-6

            audio_feature = audio_feature.float().cuda()

            data_label_torch = data_label_torch.cuda()
            data_label_torch = data_label_torch.long()

            outputs = nnet(audio_feature)

            loss = criterion(outputs.float().squeeze(1), data_label_torch.float().squeeze(1))
            output_tags = torch.ceil(outputs-0.5)
            output_tags = output_tags.detach().cpu().numpy()
            data_label_torch = data_label_torch.data.cpu().numpy()
     #       result = np.append(output_tags,data_label_torch,axis=1)
            train_mur_f1 = f1_score(data_label_torch, output_tags)



            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_epoch_mur_f1 += train_mur_f1.item()

            pre_train.extend(list(output_tags))
            label_train.extend(list(data_label_torch))
        logger.info("Iteration:{0}, loss = {1:.6f}, mur_f1 = {2:.3f}".format(iter_, running_loss/all_file_train, train_epoch_mur_f1/all_file_train))

    # Save the model.
    save_challenge_model(model_folder, nnet,nnet2, murmur_classes, outcome_classes, m_name = 'sub_ti')
    
    if verbose >= 1:
        print('Done.')

# Load your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.

#def load_challenge_model(model_folder, verbose):
#    filename = os.path.join(model_folder, 'model.sav')
#    return joblib.load(filename)

def load_challenge_model(model_folder, verbose):
    info_fnm = os.path.join(model_folder, 'desc.pk')
    with open(info_fnm, 'rb') as f:
        info_m = pk.load(f)
    if info_m['model'] == 'sub_ti' :
        print('t')
        trained_model =  torch.load(info_m['mur_model_fnm'],map_location='cpu')
        trained_model2 =  torch.load(info_m['out_model_fnm'],map_location='cpu')
        
        nnet = W2V2_LCNN(mode = 'murmur').cpu()
        nnet.load_state_dict(trained_model)
        
        nnet2 = LCNN(mode = 'outcome').cpu()
        nnet2.load_state_dict(trained_model2)
        
        print('*************** load weight ***************')
    info_m['mur_classifier'] = nnet
    info_m['out_classifier'] = nnet2
    return info_m


# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
    
def run_challenge_model(model, data, recordings, verbose):
#    torch.cuda.empty_cache()
    GPU_NUM = 2 # 원하는 GPU 번호 입력
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
#    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    nnet = model['mur_classifier'].cuda()
    nnet2 = model['out_classifier'].cuda()


    nnet.eval()
    nnet2.eval()
    
    murmur_classes = ['Present','Unknown', 'Absent'] # 
    num_murmur_classes = len(murmur_classes)
    outcome_classes = ['Abnormal', 'Normal']
    num_outcome_classes = len(outcome_classes)

    # Load features.

    probab1 = np.empty((0,2))
    probab2 = np.empty((0,2))
    softmax = nn.Softmax(dim=1)
    for i in range(len(recordings)) :
        audio_feature,_ = feature_extract_raw(recordings[i]/ 32768,samp_sec=30)
        audio_feature = torch.from_numpy(np.array(audio_feature))
        audio_feature[audio_feature != audio_feature] = 0
        audio_feature = torch.nan_to_num(audio_feature,nan=1e-6) +1e-6
        audio_feature = audio_feature.float().cuda()
        
        mel_feature = feature_extract_raw_melspec(recordings[i]/ 32768)
        mel_feature = torch.from_numpy(np.array(mel_feature))
        mel_feature[mel_feature != mel_feature] = 0
        mel_feature = torch.nan_to_num(mel_feature,nan=1e-6) +1e-6
        mel_feature = mel_feature.float().cuda()
        mel_feature = mel_feature.unsqueeze(1)
        
        
        outputs = nnet(audio_feature)
        outputs2 = nnet2(mel_feature)
        
        outputs_murmur = outputs
        outputs_outcome = outputs2
        outputs_murmur = torch.cat((1-outputs_murmur,outputs_murmur),dim = 1)
        outputs_outcome = softmax(outputs_outcome)
        
        prob1 = outputs_murmur.mean(axis = 0).reshape(1,2) 
        prob1 = prob1.data.cpu().numpy()
        probab1 = np.concatenate((probab1,prob1),axis=0)

        prob2 = outputs_outcome.mean(axis = 0).reshape(1,2) 
        prob2 = prob2.data.cpu().numpy()
        probab2 = np.concatenate((probab2,prob2),axis=0)
        
    p1 = probab1.max(axis=0)
    p2 = probab2.mean(axis=0)
    
        
    
    if p1[0]>0.69:
        idx = 0
#    elif p1[0]>thres1 and p1[0]<thres2 :
#        idx = 1
    else :
        idx = 2
        
    if p2[0]>0.4295 :
        idx2 = 0
    else :
        idx2 = 1

    n_p1 = np.zeros(len(murmur_classes))
    n_p1[0] = p1[0]
    n_p1[1] = 0
    n_p1[2] = p1[1]
    # Choose label with higher probability.
    labels_murmur = np.zeros(len(murmur_classes), dtype=np.int_)
    labels_murmur[idx] = 1
    labels_outcome = np.zeros(len(outcome_classes), dtype=np.int_)
    labels_outcome[idx2] = 1
    
    classes = murmur_classes + outcome_classes
    labels = np.concatenate((labels_murmur, labels_outcome))
    probabilities = np.concatenate((n_p1, p2))

    return classes, labels, probabilities 


################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

    
def save_challenge_model(model_folder, model_mur,model_out, murmur_classes, outcome_classes, m_name) :
    os.makedirs(model_folder, exist_ok=True)
    info_fnm = os.path.join(model_folder, 'desc.pk')
    filename = os.path.join(model_folder, m_name + '_mur_model.hdf5')
    filename2 = os.path.join(model_folder, m_name + '_out_model.hdf5')
#    model.save(filename)
    torch.save(model_mur.state_dict(), filename)
    torch.save(model_out.state_dict(), filename2)
    d = {'model': m_name, 'murmur_classes': murmur_classes, 'outcome_classes': outcome_classes, 'mur_model_fnm': filename, 'out_model_fnm':filename2}    
    with open(info_fnm, 'wb') as f:
        pk.dump(d, f, pk.HIGHEST_PROTOCOL)

        
        
