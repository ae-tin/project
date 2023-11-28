
import argparse, glob, os, torch, warnings, time
import torch
import pandas as pd
import numpy as np

import os
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier




parser = argparse.ArgumentParser(description = "ECAPA_trainer")
## Training Settings
parser.add_argument("--lr_decay",   type=float, default=0.97,    help='Learning rate decay every [test_step] epochs')

## Training and evaluation path/lists, save path
parser.add_argument('--csv_path',  type=str,   default="exps/xgboost", help='Path to save the score.txt and models')
parser.add_argument('--save_path',  type=str,   default="exps/xgboost", help='Path to save the score.txt and models')

## Model and Loss settings

print ('Available devices ', torch.cuda.device_count())
GPU_NUM = 4 # 원하는 GPU 번호 입력
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)
print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())

## Initialization
warnings.simplefilter("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')
args = parser.parse_args()
args = init_args(args)

raw = pd.read_csv(args.csv_path, index_col=0)
raw['m'] = raw.apply(lambda x : x['loanapply_insert_time'][5:7],axis=1)


def cross_valid(data,i) :    # 전처리가 다 된 데이터, 모든 column은 들어갈 변수 + target  
    
    print('%d 번째 cv 구성 중....'%(i))
    
    data_34_0 = data[(data.m != '05') & (data.is_applied == 0)]
    data_34_1 = data[(data.m != '05') & (data.is_applied == 1)]
    data_05_0 = data[(data.m == '05') & (data.is_applied == 0)]
    data_05_1 = data[(data.m == '05') & (data.is_applied == 1)]
    
    data_34_0["set"] = "train"
    data_34_1["set"] = "train"
    data_05_0["set"] = np.random.choice(["train", "valid"], p =[.53, .57], size=(data_05_0.shape[0],))
    data_05_1["set"] = np.random.choice(["train", "valid"], p =[.46, .54], size=(data_05_1.shape[0],))
    
    all_data = pd.concat([data_34_0,data_34_1,data_05_0,data_05_1], ignore_index=True)
    
    train_indices = all_data[all_data.Set=="train"].index
    valid_indices = all_data[all_data.Set=="valid"].index
    
    target = ''                # Y 변수 이름
    unused_feat = ['set']           # 안 쓸 변수 이름
    features = [ col for col in all_data.columns if col not in unused_feat+[target]]
    
    
    
    
    return train,valid


    
    
    
    
    
xgb = XGBClassifier(n_estimators = 500, learning_rate = 0.1,max_depth = 100,tree_method="gpu_hist",gpu_id = 4)
















## Search for the exist models
modelfiles = glob.glob('%s/model_0*.model'%args.model_save_path)
modelfiles.sort()

## Only do evaluation, the initial_model is necessary
if args.eval == True:
	s = ECAPAModel(**vars(args))
	print("Model %s loaded from previous state!"%args.initial_model)
	s.load_parameters(args.initial_model)
	EER, minDCF = s.eval_network(eval_list = args.eval_list, eval_path = args.eval_path)
	print("EER %2.2f%%, minDCF %.4f%%"%(EER, minDCF))
	quit()

## If initial_model is exist, system will train from the initial_model
if args.initial_model != "":
	print("Model %s loaded from previous state!"%args.initial_model)
	s = ECAPAModel(**vars(args))
	s.load_parameters(args.initial_model)
	epoch = 1

## Otherwise, system will try to start from the saved model&epoch
elif len(modelfiles) >= 1:
	print("Model %s loaded from previous state!"%modelfiles[-1])
	epoch = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][6:]) + 1
	s = ECAPAModel(**vars(args))
	s.load_parameters(modelfiles[-1])
## Otherwise, system will train from scratch
else:
	epoch = 1
	s = ECAPAModel(**vars(args))

EERs = []
score_file = open(args.score_save_path, "a+")

while(1):
	## Training for one epoch
	start = time.time()
	loss, lr, acc = s.train_network(epoch = epoch, loader = trainLoader)
	
	## Evaluation every [test_step] epochs
	if epoch % args.test_step == 0:
		s.save_parameters(args.model_save_path + "/model_%04d.model"%epoch)
		EERs.append(s.eval_network(eval_list = args.eval_list, eval_path = args.eval_path)[0])
		end = time.time()
		print(time.strftime("%Y-%m-%d %H:%M:%S"), "%d epoch, ACC %2.2f%%, EER %2.2f%%, bestEER %2.2f%%, time per epoch : %d"%(epoch, acc, EERs[-1], min(EERs),end-start))
		score_file.write("%d epoch, LR %f, LOSS %f, ACC %2.2f%%, EER %2.2f%%, bestEER %2.2f%%, time per epoch : %d\n"%(epoch, lr, loss, acc, EERs[-1], min(EERs), end-start))
		score_file.flush()
        
	if epoch >= args.max_epoch:
		
		quit()

	epoch += 1
