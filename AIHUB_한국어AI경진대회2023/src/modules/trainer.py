import torch
import torch.nn as nn
import numpy as np
import math
from dataclasses import dataclass
import time
from tqdm import tqdm
#from nova import DATASET_PATH
DATASET_PATH = '/home/work/dataADD/ktelspeech/Training/D60/'

def trainer(mode, config, dataloader, optimizer, model, criterion, metric, train_begin_time, device):

    log_format = "[INFO] step: {:4d}/{:4d}, total_loss: {:.6f}, ctc_loss: {:.6f}, rmse_loss: {:.6f}, " \
                              "cer: {:.2f}, elapsed: {:.2f}s {:.2f}m {:.2f}h, lr: {:.6f}"
    score_file = open(config.score_save_path.replace(".txt","_log.txt"), "a+")
    total_num = 0
    epoch_loss_total = 0.
    print(f'[INFO] {mode} Start')
    epoch_begin_time = time.time()
    cnt = 0
    criterion2 = nn.MSELoss()
    t = 5
    log_format2 = log_format+"\n"
    for inputs, targets, input_lengths, target_lengths in dataloader:
        begin_time = time.time()

        optimizer.zero_grad()
        inputs = inputs.to(device)
        targets = targets.to(device)
        input_lengths = input_lengths.to(device)
        target_lengths = torch.as_tensor(target_lengths).to(device)
        model = model.to(device)
        outputs, output_lengths, rnn_outputs, masked_rnn_outputs = model(inputs, input_lengths)
        loss1 = criterion(
            outputs.transpose(0, 1),
            targets[:, 1:],
            tuple(output_lengths),
            tuple(target_lengths)
        )
        loss2 = torch.sqrt(criterion2(rnn_outputs, masked_rnn_outputs))
#        print(loss2)
        loss = loss1+loss2*t
        y_hats = outputs.max(-1)[1]

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step(model)

        total_num += int(input_lengths.sum())
        epoch_loss_total += loss.item()

        torch.cuda.empty_cache()
        
        if cnt % config.mid_checkpoint_every == 0:
            torch.save(model.state_dict(), config.model_save_path + 'model_mid{}'.format(cnt))
        
        
        if cnt % config.print_every == 0:

            current_time = time.time()
            elapsed = current_time - begin_time
            epoch_elapsed = (current_time - epoch_begin_time) / 60.0
            train_elapsed = (current_time - train_begin_time) / 3600.0
            cer = metric(targets[:, 1:], y_hats)
            print(log_format.format(
                cnt, len(dataloader), loss,loss1,loss2,
                cer, elapsed, epoch_elapsed, train_elapsed,
                optimizer.get_lr(),
            ))
            score_file.write(log_format2.format(
                cnt, len(dataloader), loss,loss1,loss2,
                cer, elapsed, epoch_elapsed, train_elapsed,
                optimizer.get_lr(),
            ))
            score_file.flush()
        cnt += 1
    return model, epoch_loss_total/len(dataloader), metric(targets[:, 1:], y_hats)











def evaluater(mode, config, dataloader, optimizer, model, criterion, metric, train_begin_time, device):
    log_format = "[INFO] Eval cer: {:.2f}"
    total_num = 0
    epoch_loss_total = 0.
    cnt = 0
    cer = 0
    print(f'[INFO] {mode} Start')
    epoch_begin_time = time.time()
    for inputs, targets, input_lengths, target_lengths in tqdm(dataloader):

        inputs = inputs.to(device)
        targets = targets.to(device)
        input_lengths = input_lengths.to(device)
        target_lengths = torch.as_tensor(target_lengths).to(device)
        
        model = model.to(device)
        with torch.no_grad():
            outputs, output_lengths,_,__ = model(inputs, input_lengths)

        y_hats = outputs.max(-1)[1]

        total_num += int(input_lengths.sum())

        torch.cuda.empty_cache()


        cer += metric(targets[:, 1:], y_hats)
        cnt += 1
    cer /= cnt

    print(log_format.format(
        cer
    ))
    return cer
