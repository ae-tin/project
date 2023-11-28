"""
Main script that trains, validates, and evaluates
various models including AASIST.

AASIST
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import argparse
import json
import os
import sys
import warnings
from importlib import import_module
from pathlib import Path
from shutil import copy
from typing import Dict, List, Union
from tqdm import tqdm
#from models.Wav2vec2_base import Model as w2v2_model
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
#from torch.utils.tensorboard import SummaryWriter
from torchcontrib.optim import SWA

from data_utils import (Dataset_embd_Train,
                        Dataset_embd_Dev,
                        Dataset_embd_Eval, gen_list)
from evaluation import calculate_tDCF_EER
from utils import create_optimizer, seed_worker, set_seed, str_to_bool
import pandas as pd
import pickle as pk
warnings.filterwarnings("ignore", category=FutureWarning)


def main(args: argparse.Namespace) -> None:
    """
    Main function.
    Trains, validates, and evaluates the ASVspoof detection model.
    """
    # load experiment configurations
    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())
    model_config = config["model_config"]
    optim_config = config["optim_config"]
    optim_config["epochs"] = config["num_epochs"]
    if "eval_all_best" not in config:
        config["eval_all_best"] = "True"
    classifier = model_config["classifier"]
    print("classifier : ", classifier)
    w2v2_ecapa = model_config["w2v2_ecapa"][0]
    if w2v2_ecapa == 1 :
        w2v2_ecapa = 'ec_'
        print("---- w2v2_ecapa ----")
    else :
        w2v2_ecapa = ''
        
    w2v2_ecapa_text = model_config["w2v2_ecapa_text"][0]
    if w2v2_ecapa_text == 1 :
        w2v2_ecapa_text = 'ect_'
        print("---- w2v2_ecapa_text ----")
    else :
        w2v2_ecapa_text = ''
        
    w2v2_hybrid = model_config["w2v2_hybrid"][0]
    if w2v2_hybrid == 1 :
        w2v2_hybrid = 'hb_'
        print("---- w2v2_hybrid ----")
    else :
        w2v2_hybrid = ''
        
    w2v2_hybrid_text = model_config["w2v2_hybrid_text"][0]
    if w2v2_hybrid_text == 1 :
        w2v2_hybrid_text = 'hbt_'
        print("---- w2v2_hybrid_text ----")
    else :
        w2v2_hybrid_text = ''
    
    ensemble = w2v2_ecapa + w2v2_ecapa_text + w2v2_hybrid + w2v2_hybrid_text
    ensemble = ensemble[:-1]
        
    # make experiment reproducible
    set_seed(args.seed, config)

    # define database related paths
    output_dir = Path(args.output_dir)
    database_path = Path(config["database_path"])


    # define model related paths
    model_tag_o = "emotion_{}_ep{}_bs{}_lr{}_{}_{}".format(
        os.path.splitext(os.path.basename(args.config))[0],
        config["num_epochs"], config["batch_size"],config["optim_config"]["base_lr"]
        ,ensemble,classifier)
    if args.comment:
        model_tag_o = model_tag_o + "_{}".format(args.comment)
    model_tag = output_dir / model_tag_o
    model_save_path = model_tag / "weights"
#    eval_score_fnm = model_tag_o + '.csv'
#    eval_score_path = model_tag / eval_score_fnm
#    writer = SummaryWriter(model_tag)
    os.makedirs(model_save_path, exist_ok=True)
    copy(args.config, model_tag / "config.conf")

    # set device
#    device = "cuda" if torch.cuda.is_available() else "cpu"
    cuda_device = str(model_config["device"])
    device = torch.device('cuda:{}'.format(cuda_device) if torch.cuda.is_available() else 'cpu')
    print("Device: {}".format(device))
    if device == "cpu":
        raise ValueError("GPU not detected!")

    # define model architecture
    model = get_model(model_config, device)

    # define dataloaders
    trn_loader, dev_loader, eval_loader = get_loader(
        database_path, args.seed, config) # 

    # evaluates pretrained model and exit script
    if args.eval:
        model.load_state_dict(
            torch.load(config["model_path"], map_location=device))
        print("Model loaded : {}".format(config["model_path"]))
        print("Start evaluation...")

        produce_evaluation_file2(eval_loader, model, device, eval_score_path)
        
        print("DONE.")
        sys.exit(0)

    # get optimizer and scheduler
    optim_config["steps_per_epoch"] = len(trn_loader)
    optimizer, scheduler = create_optimizer(model.parameters(), optim_config)
    optimizer_swa = SWA(optimizer)

    best_dev_acc = 75.
    n_swa_update = 0  # number of snapshots of model to use in SWA
    f_log = open(model_tag / "metric_log.txt", "a")
    f_log.write("=" * 5 + "\n")

    # make directory for metric logging
    metric_path = model_tag / "metrics"
    os.makedirs(metric_path, exist_ok=True)

    # Training
    for epoch in range(config["num_epochs"]):
        print("Start training epoch{:03d}".format(epoch))
        train_loss, train_acc = train_epoch(trn_loader, model, optimizer, device,
                                   scheduler, config)
        dev_loss, dev_acc = valid_epoch(dev_loader, model, optimizer, device,
                                   scheduler, config)
 #       eval_score_fnm = model_tag_o + "_{}".format(epoch) + '.csv'
 #       eval_score_path = model_tag / eval_score_fnm
 #       produce_evaluation_file(eval_loader, model, device, eval_score_path)
        print("DONE.\nTrn_Loss:{:.5f}, Trn_ACC: {:.3f}, Dev_Loss:{:.5f}, Dev_ACC: {:.3f}".format(
            train_loss, train_acc, dev_loss,dev_acc))   #  
#        writer.add_scalar("Trn_Loss", train_loss, epoch)
#        writer.add_scalar("Trn_ACC", train_acc, epoch)
#        writer.add_scalar("Dev_Loss", dev_loss, epoch)
#        writer.add_scalar("Dev_ACC", dev_acc, epoch)

        if best_dev_acc < dev_acc:
            print("best model find at epoch", epoch)
            best_dev_acc = dev_acc
            torch.save(model.state_dict(),
                       model_save_path / "epoch_{}_{:03.3f}.pth".format(epoch, dev_acc))

            # do evaluation whenever best model is renewed
            if str_to_bool(config["eval_all_best"]):
                eval_score_fnm = model_tag_o + "_{}".format(epoch) + '.csv'
                eval_score_path = model_tag / eval_score_fnm
                produce_evaluation_file(eval_loader, model, device, eval_score_path)

            log_text = "epoch{:03d}, ".format(epoch)
            log_text += "  best eer, {:.4f}%".format(dev_acc)
            torch.save(model.state_dict(),
                       model_save_path / "best.pth")
            if len(log_text) > 0:
                print(log_text)
                f_log.write(log_text + "\n")

            print("Saving epoch {} for swa".format(epoch))
            optimizer_swa.update_swa()
            n_swa_update += 1
#        writer.add_scalar("best_dev_eer", best_dev_acc, epoch)

    print("Start final evaluation")
    epoch += 1
    if n_swa_update > 0:
        optimizer_swa.swap_swa_sgd()
        optimizer_swa.bn_update(trn_loader, model, device=device)
    
    f_log = open(model_tag / "metric_log.txt", "a")
    f_log.write("=" * 5 + "\n")
    f_log.write("Best Dev ACCURACY: {:.3f}".format(best_dev_acc))
    f_log.close()


    print("Exp FIN. Best Dev ACCURACY: {:.3f}".format(best_dev_acc))


def get_model(model_config: Dict, device: torch.device):
    """Define DNN model architecture"""
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "cat_embd_model")
    model = _model(model_config).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))

    return model


def get_loader(
        database_path: str,
        seed: int,
        config: dict) -> List[torch.utils.data.DataLoader]:
    """Make PyTorch DataLoaders for train / developement / evaluation"""
  
    trn_list_path = pd.read_csv('./data/train_95_aug.csv')
    dev_trial_path = pd.read_csv('./data/valid_05.csv')
    eval_trial_path = pd.read_csv('./data/test.csv')

    d_label_trn, file_train = gen_list(dir_meta=trn_list_path,
                                            is_train=True,
                                            is_eval=False)
    print("no. training files:", len(file_train))

    train_set = Dataset_embd_Train(file_list=file_train, labels=d_label_trn,config=config)
    
    gen = torch.Generator()
    gen.manual_seed(seed)
    trn_loader = DataLoader(train_set,
                            batch_size=config["batch_size"],
                            shuffle=True,
                            drop_last=False,
                            pin_memory=True,
                            worker_init_fn=seed_worker,
                            generator=gen)

    d_label_dev, file_dev = gen_list(dir_meta=dev_trial_path,
                                is_train=False,
                                is_eval=False)
    print("no. validation files:", len(file_dev))

    dev_set = Dataset_embd_Dev(file_list=file_dev, labels=d_label_dev,config=config)
    
    dev_loader = DataLoader(dev_set,
                            batch_size=config["batch_size"],
                            shuffle=False,
                            drop_last=False,
                            pin_memory=True)

    file_eval = gen_list(dir_meta=eval_trial_path,
                              is_train=False,
                              is_eval=True)
    
    print("no. evaluation files:", len(file_eval))
    eval_set = Dataset_embd_Eval(file_list=file_eval,config=config)
    
    eval_loader = DataLoader(eval_set,
                             batch_size=config["batch_size"],
                             shuffle=False,
                             drop_last=False,
                             pin_memory=True)

    return trn_loader,dev_loader, eval_loader  


def produce_evaluation_file(
    data_loader: DataLoader,
    model,
    device: torch.device,
    save_path: str) -> None:
    """Perform evaluation and save the score to a file"""
    model.eval()
    submission = pd.read_csv('./data/sample_submission.csv')
    check_id = submission.id
    
    fname_list = []
    score_list = []
    for batch_x, utt_id in tqdm(data_loader):
        batch_x = batch_x.to(device)
        with torch.no_grad():
            _, batch_out = model(batch_x)
            batch_score = batch_out.max(1, keepdim=True)[1].data.cpu().numpy().ravel()
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())

    assert len(check_id) == len(fname_list) == len(score_list)
    for fn, test_id in zip(fname_list, check_id):
        assert fn == test_id
    submission['label'] = score_list
    submission.to_csv(save_path,index=False)
    print("Scores saved to {}".format(save_path))

def produce_evaluation_file2(
    data_loader: DataLoader,
    model,
    device: torch.device,
    save_path: str) -> None:
    """Perform evaluation and save the score to a file"""
    model.eval()
    submission = pd.read_csv('./data/sample_submission.csv')
    check_id = submission.id
    embedding_dict = {}
    embedding_list = torch.empty(0)
    fname_list = []
    score_list = []
    for batch_x, utt_id in tqdm(data_loader):
        batch_x = batch_x.to(device)
        with torch.no_grad():
            embed, batch_out = model(batch_x)
            batch_score = batch_out.max(1, keepdim=True)[1].data.cpu().numpy().ravel()
            embed = embed.data.cpu()
            
        # add outputs
        embedding_list = torch.cat((embedding_list,embed),0)
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())
    
    assert len(check_id) == len(fname_list) == len(score_list) == len(embedding_list)
    for fn, test_id, embedding in zip(fname_list, check_id,embedding_list):
        assert fn == test_id
        embedding_dict[test_id] = embedding
    fnm = str(save_path).replace('.csv','_test.pk')
    with open(fnm, 'wb') as f:
        pk.dump(embedding_dict, f)
    print("Embedding saved to {}".format(fnm))

    submission['label'] = score_list
    submission.to_csv(save_path,index=False)
    print("Scores saved to {}".format(save_path))    

def produce_train_embedding_file(
    data_loader: DataLoader,
    model,
    device: torch.device,
    save_path: str) -> None:
    """Perform evaluation and save the score to a file"""
    model.eval()
    submission = pd.read_csv('./data/train_95.csv')
    check_id = submission.id
    embedding_dict = {}
    embedding_list = torch.empty(0)
    fname_list = []
    score_list = []
    for batch_x, utt_id in tqdm(data_loader):
        batch_x = batch_x.to(device)
        with torch.no_grad():
            embed, batch_out = model(batch_x)
            batch_score = batch_out.max(1, keepdim=True)[1].data.cpu().numpy().ravel()
            embed = embed.data.cpu()
            
        # add outputs
        embedding_list = torch.cat((embedding_list,embed),0)
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())
#    print(embedding_list.shape[0])
#    print(len(embedding_list))
#    print(len(check_id))
#    print(len(fname_list))
#    print(len(score_list))
    assert len(check_id) == len(fname_list) == len(score_list) == len(embedding_list)
    
    for fn, embedding in zip(fname_list,embedding_list):
        embedding_dict[fn] = embedding
    fnm = str(save_path).replace('.csv','_train.pk')
    with open(fnm, 'wb') as f:
        pk.dump(embedding_dict, f)
    print("Train Embedding saved to {}".format(fnm))


def train_epoch(
    trn_loader: DataLoader,
    model,
    optim: Union[torch.optim.SGD, torch.optim.Adam],
    device: torch.device,
    scheduler: torch.optim.lr_scheduler,
    config: argparse.Namespace):
    """Train the model for one epoch"""
    running_loss = 0
    num_total = 0.0
    correct = 0.0
    
    model.train()

    # set objective (Loss) functions
#    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss()#weight=weight)
    for batch_x, batch_y in tqdm(trn_loader):
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.type(torch.int64).to(device)
        _, batch_out = model(batch_x)#, Freq_aug=str_to_bool(config["freq_aug"]))
        batch_loss = criterion(batch_out, batch_y)
        running_loss += batch_loss.item() * batch_size
        optim.zero_grad()
        batch_loss.backward()
        optim.step()

        if config["optim_config"]["scheduler"] in ["cosine", "keras_decay"]:
            scheduler.step()
        elif scheduler is None:
            pass
        else:
            raise ValueError("scheduler error, got:{}".format(scheduler))
            
        pred = batch_out.max(1, keepdim=True)[1]
            
        correct += pred.eq(batch_y.view_as(pred)).sum().item()
        

    running_loss /= num_total
    accuracy = correct / num_total *100
    return running_loss, accuracy

def valid_epoch(
    trn_loader: DataLoader,
    model,
    optim: Union[torch.optim.SGD, torch.optim.Adam],
    device: torch.device,
    scheduler: torch.optim.lr_scheduler,
    config: argparse.Namespace):
    """Train the model for one epoch"""
    running_loss = 0
    num_total = 0.0
    correct = 0.0
    model.eval()

    # set objective (Loss) functions
#    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss()#weight=weight)
    for batch_x, batch_y in tqdm(trn_loader):
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.type(torch.int64).to(device)
        with torch.no_grad():
            _, batch_out = model(batch_x)#, Freq_aug=str_to_bool(config["freq_aug"]))
        batch_loss = criterion(batch_out, batch_y)
        running_loss += batch_loss.item() * batch_size
        
        pred = batch_out.max(1, keepdim=True)[1]
        correct += pred.eq(batch_y.view_as(pred)).sum().item()
        
    running_loss /= num_total
    
    accuracy = correct / num_total *100
    return running_loss, accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emotion Recognition system")
    parser.add_argument("--config",
                        dest="config",
                        type=str,
                        help="configuration file",
                        default="./config/embedding_ensemble.conf")
                   #     required=True)
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        help="output directory for results",
        default="./exp_result",
    )
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="random seed (default: 1234)")
    parser.add_argument(
                        "--eval",
                        action="store_true",
                        help="when this flag is given, evaluates given model and exit")
    parser.add_argument("--comment",
                        type=str,
                        default=None,
                        help="comment to describe the saved model")
    parser.add_argument("--eval_model_weights",
                        type=str,
                        default=None,
                        help="directory to the model weight file (can be also given in the config file)")
    main(parser.parse_args())
