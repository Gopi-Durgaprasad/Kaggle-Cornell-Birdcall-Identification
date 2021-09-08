import warnings
warnings.filterwarnings('ignore')

import cv2
import audioread
import logging
import os
import random
import time
import warnings

import librosa
import librosa.display as display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

from contextlib import contextmanager
from IPython.display import Audio
from pathlib import Path
from typing import Optional, List

from catalyst.data.sampler import DistributedSampler, BalanceClassSampler
#from catalyst.dl import SupervisedRunner, State, CallbackOrder, Callback, CheckpointCallback
from fastprogress import progress_bar
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, average_precision_score

import classifiers
import losses
import dataset

from audio_albu import augmenter
from config import args
from utils import AverageMeter, MetricMeter
from schedulers import GradualWarmupSchedulerV2
from pytorch_utils import do_mixup, Mixup, move_data_to_device

mixup_augmenter = Mixup(mixup_alpha=1.)

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    
    #if alpha > 0:
    #    lam = np.random.beta(alpha, alpha)
    #else:
    #    lam = 1
    
    batch_size = x.size()[0]

    if use_cuda:
        index0 = torch.randperm(batch_size).cuda()
        index1 = torch.randperm(batch_size).cuda()
        index2 = torch.randperm(batch_size).cuda()
        index3 = torch.randperm(batch_size).cuda()
        index4 = torch.randperm(batch_size).cuda()

    else:
        index = torch.randperam(bath_size)
    
    #mixed_x = lam * x + (1 - lam) * x[index, :]

    ind = random.choice([0,1,2,3,4])

    if ind == 0:
        mixed_x = x
        mixed_y = y
    elif ind == 1:
        mixed_x = torch.cat([x, x[index1, :]], dim=1)
        mixed_y = y + y[index1, :]
    elif ind == 2:
        mixed_x = torch.cat([x, x[index1, :], x[index2]], dim=1)
        mixed_y = y + y[index1, :] + y[index2, :]
    elif ind == 3:
        mixed_x = torch.cat([x, x[index1, :], x[index2], x[index3, :]], dim=1)
        mixed_y = y + y[index1, :] + y[index2, :] + y[index3, :]
    elif ind == 4:
        mixed_x = torch.cat([x, x[index1, :], x[index2], x[index3, :], x[index4, :]], dim=1)
        mixed_y = y + y[index1, :] + y[index2, :] + y[index3, :] + y[index4, :]
    
    #mixed_y = torch.clamp(mixed_y, min=0, max=1)

    """
    if np.random.random() > 0.5:
        mixed_x = lam * x + (1 - lam) * x[index, :]
        mixed_y = lam * y + (1 - lam) * y[index, :]
    else:
        mixed_x =  x + x[index, :]
        mixed_y =  y + y[index, :]
        
    """
    return mixed_x, mixed_y

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_epoch(args, model, loader, criterion, optimizer, epoch):
    losses = AverageMeter()
    scores = MetricMeter(train=True)

    model.train()
    scaler = torch.cuda.amp.GradScaler()

    t = tqdm(loader, total=1000)
    for i, sample in enumerate(t):

        optimizer.zero_grad()

        input = sample['waveform'].to(args.device)
        target = sample['targets'].to(args.device)

        if args.new_mixup:
            input, target = mixup_data(input, target, args.alpha, True)
        
        ran = np.random.random()

        if args.mixup and ran > 0.5:
            mixup_lambda = mixup_augmenter.get_lambda(
                batch_size=len(input)
            )
            mixup_lambda = move_data_to_device(mixup_lambda, args.device)

            target = do_mixup(target, mixup_lambda)

        with torch.cuda.amp.autocast(enabled=False):
            if args.mixup and ran > 0.5:
                output = model(input, True, mixup_lambda)
            else:
                output = model(input, True)
            
            #print(target.sum(dim=1))
            #print(input.shape)

            loss = criterion(output, target)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        #loss.backward()
        #optimizer.step()
        #scheduler.step()

        bs = input.size(0)
        scores.update(target, output)
        losses.update(loss.item(), bs)

        t.set_description(f"Train E:{epoch} - Loss:{losses.avg:0.4f}")

        if i == 1000:
            break

    t.close()

    return scores.avg, losses.avg

def valid_epoch(args, model, loader, criterion, epoch):
    losses = AverageMeter()
    scores = MetricMeter(train=False)

    model.eval()

    with torch.no_grad():
        t = tqdm(loader)
        for i, sample in enumerate(t):

            input = sample['waveform'].to(args.device)
            target = sample['targets'].to(args.device)

            output = model(input)
            loss = criterion(output, target)

            bs = input.size(0)
            scores.update(target, output)
            losses.update(loss.item(), bs)

            t.set_description(f"Valid E:{epoch} - Loss:{losses.avg:0.4f}")

    t.close()

    return scores.avg, losses.avg

def main(fold):

    # Setting seed
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    args.fold = fold
    args.save_path = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(args.save_path, exist_ok=True)

    model = classifiers.__dict__[args.network](**args.model_config)
    if args.pretrained_path:
        weights = torch.load(args.pretrained_path, map_location=args.device)
        model.load_state_dict(weights["model"], strict=False)
    
    model = model.to(args.device)

    criterion = losses.__dict__[args.losses]()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)

    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.epochs)
    scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=1, after_scheduler=scheduler_cosine)
    
    train_df = pd.read_csv(args.train_csv)
    #valid_df = pd.read_csv(args.valid_csv)
    if args.DEBUG:
        train_df = train_df.sample(1000)
        #valid_df = valid_df.sample(1000)
    
    train_fold = train_df[train_df.kfold != fold]
    valid_fold = train_df[train_df.kfold.isin([fold])]

    #print(valid_fold.head())

    valid_fold = valid_fold[valid_fold.ebird_code == valid_fold.pred_code]

    #print(valid_fold.head())

    train_file_list = train_fold[['path', 'pred_code', 'pred_prob']].values.tolist()
    valid_file_list = valid_fold[['path', 'pred_code', 'pred_prob']].values.tolist()

    train_dataset = dataset.PANNsDataset(
        train_file_list,
        period=args.PERIOD,
        transforms=augmenter,#train_transforms(args.mel_param),
        train=True
    )
    valid_dataset = dataset.PANNsDataset(
        valid_file_list,
        period=args.PERIOD,
        transforms=False,#valid_transforms(args.mel_param)
        train=False
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        #sampler = BalanceClassSampler(labels=train_dataset.__get_labels__(), mode="upsampling"),
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers
    )

    print(".................... Training started .................")

    best_map = -999999
    best_f1  = -999999
    best_row_f1 = -99999

    if args.load_from:
        # "drive/My Drive/Cornell Birdcall Identification/weights/Cnn14_16k_5"
        weights = torch.load(os.path.join("drive/My Drive/Cornell Birdcall Identification/weights/Cnn14_16k_5", f"fold-{args.fold}.bin"))
        model.load_state_dict(weights["model"], strict=False)
        #optimizer.load_state_dict(weights["optimizer"])
        #scheduler_warmup.load_state_dict(weights["scheduler_warmup"])
        #args.start_epoch = weights["epoch"] + 1
        #best_map = 0.6212948058165978
        
        model = model.to(args.device)


    for epoch in range(args.start_epoch, args.epochs):

        #scheduler_warmup.step(epoch)

        train_avg, train_loss = train_epoch(
            args,
            model,
            train_loader,
            criterion,
            optimizer,
            epoch
        )

        valid_avg, valid_loss = valid_epoch(
            args,
            model,
            valid_loader,
            criterion,
            epoch
        )

        scheduler_warmup.step()    
        if epoch==2: scheduler_warmup.step() # bug workaround 

        content = f"""
                {time.ctime()} \n
                Fold:{args.fold}, Epoch:{epoch}, lr:{optimizer.param_groups[0]['lr']:.7}, \n
                Train Loss:{train_loss:0.4f} - ACC:{train_avg['acc']:0.4f} - F1:{train_avg['f1']:0.4f} - MAP:{train_avg['map']:0.4f}  \n
                Valid Loss:{valid_loss:0.4f} - ACC:{valid_avg['acc']:0.4f} - F1:{valid_avg['f1']:0.4f} - ROW_F1:{valid_avg['row_f1']:0.4f} - MAP:{valid_avg['map']:0.4f} \n\n
        """
        print(content)

        with open(f'{args.save_path}/log_{args.exp_name}.txt', 'a') as appender:
            appender.write(content + '\n')
        
        if valid_avg["map"] > best_map:
            print(f"######### >>>>>>> Model Improved from {best_map} -----> {valid_avg['map']}")
            checkpoint = { 
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler_warmup': scheduler_warmup.state_dict()
            }

            torch.save(checkpoint, os.path.join(args.save_path, f"checkpoint-fold-{args.fold}.bin"))
            torch.save(model.state_dict(), os.path.join(args.save_path, f"fold-{args.fold}.bin"))

            best_map = valid_avg["map"]
        
        torch.save(model.state_dict(), os.path.join(args.save_path, f"last-fold-{args.fold}.bin"))

if __name__ == "__main__":
    
    for fold in range(5):
        print("#"*20)
        print(f"####### FOLD : {fold}")
        if fold == 0:
            main(fold)
        