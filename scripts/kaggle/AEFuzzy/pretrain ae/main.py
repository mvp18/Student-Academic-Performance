import torch
import torch.optim as optim
import torch.nn as nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

from utils import *
from model import *

import torch.backends.cudnn as cudnn
cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import argparse
import copy
from torchsummary import summary

def main(args):
    seed = 42
    np.random.seed(seed)

    torch.manual_seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    rng = np.arange(0, 101, 1)

    dataloader = Create_Dataset(num_fuzz_var=4, rng=rng)

    X_trval, X_test, y_trval, y_test = train_test_split(dataloader.X, dataloader.Y, test_size=0.1, random_state=seed, stratify=dataloader.Y)
    X_train, X_val, y_train, y_val = train_test_split(X_trval, y_trval, test_size=0.11, random_state=seed, stratify=y_trval)

    model = Autoencoder(data_dim=X_train.shape[1], drop_rate=args.dropout_rate)
    model.apply(init_weights)
    model.to(device)

    criterion = nn.MSELoss()

    if args.optimizer:
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, nesterov=True, momentum=0.9, weight_decay=1e-5)

    min_loss = 999.0

    for epoch in range(args.num_epochs):

        if (epoch+1)%50==0:
            optimizer.param_groups[0]['lr']/=2
            print('\nNew learning rate :', optimizer.param_groups[0]['lr'])

        rn = torch.randperm(X_train.shape[0])
        shuffled_data = X_train[rn]

        model, train_loss = train_epoch(model, shuffled_data, criterion, optimizer, args.batch_size, device)
        val_loss = val_epoch(model, X_val, criterion, args.batch_size, device)

        print('\nEpoch:{}, Train Loss:{}, Val Loss:{}'.format(epoch+1, train_loss, val_loss,))

        if val_loss<min_loss:
            min_loss = val_loss
            bestEp = epoch+1
            best_encoder_wts = copy.deepcopy(model.encoder.state_dict())
            best_decoder_wts = copy.deepcopy(model.decoder.state_dict())

    print('\nBest performance at epoch {} with validation loss {}.'.format(bestEp, min_loss))
    print('Saving encoder and decoder weights')
    torch.save(best_encoder_wts, '../weights/encoder_wts.pth')
    torch.save(best_decoder_wts, '../weights/decoder_wts.pth')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Neuro-fuzzy methods for predicting student academic performance.")
    parser.add_argument('-opt', '--optimizer', help="if 0 optimizer changes to SGD", default=1, type=bool)
    parser.add_argument('-lr', '--learning_rate', help="learning rate for Adam", default=1e-3, type=float)
    parser.add_argument('-e', '--num_epochs', help="epochs to run", default=400, type=int)
    parser.add_argument('-bs', '--batch_size', help="batch size used throughout", default=8, type=int)
    parser.add_argument('-dr', '--dropout_rate', help="dropout rate for encoder and classifier", default=0.1, type=float)

    args = parser.parse_args()
    main(args)
