import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
import os

from utils import *
from model import *

import torch
import torch.optim as optim
import torch.nn as nn

import argparse
import copy
from torchsummary import summary

import torch.backends.cudnn as cudnn
cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def main(args):
    seed = 42
    np.random.seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Membership functions for 3 fuzzy variables - absence and 2 sets of grades
    rng_absence = np.arange(0, 76, 1)
    rng_grades = np.arange(0,21,1)

    dataloader = Create_Dataset(data_size=395, num_fuzz_var=3, rng_absence=rng_absence, rng_grades=rng_grades, subject=args.subject)

    # Train-val-test split
    X_trval, X_test, y_trval, y_test = train_test_split(dataloader.X, dataloader.Y, test_size=0.1, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_trval, y_trval, test_size=0.11, random_state=seed)

    model = AEFuzzy(data_dim=X_train.shape[1], drop_rate=args.dropout_rate).to(device)
    model.apply(init_weights)
    print('Model Summary:\n')
    print(summary(model, (X_train.shape[1],)))

    criterion_decoder = nn.MSELoss()
    if args.reg_loss:
        criterion_regressor = nn.SmoothL1Loss()
    else:
        criterion_regressor = nn.L1Loss()
    
    if args.optimizer:
        optimizer = optim.Adam([{'params': model.regressor.parameters(), 'weight_decay':args.dense_l2} 
                           ], lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    else:
        optimizer = optim.SGD([{'params': model.regressor.parameters(), 'weight_decay':args.dense_l2} 
                           ], lr=args.learning_rate, nesterov=True, momentum=0.9)

    training_loss = []
    validation_loss = []
    validation_r2 = []

    best_r2 = 0.0

    for epoch in range(args.num_epochs):

        if (epoch+1)%50==0:
            optimizer.param_groups[0]['lr']/=2
            print('\nNew learning rate :', optimizer.param_groups[0]['lr'])

        rn = torch.randperm(X_train.shape[0])
        shuffled_data, shuffled_label = X_train[rn], y_train[rn]

        model, train_loss, train_r2 = train_epoch(model, shuffled_data, shuffled_label, criterion_decoder, criterion_regressor, 
                                        optimizer, args.batch_size, args.reg_loss_wt, device)
        val_loss, val_r2 = val_epoch(model, X_val, y_val, criterion_decoder, criterion_regressor, args.batch_size, device)

        training_loss.append(train_loss)
        validation_loss.append(val_loss)
        validation_r2.append(val_r2)
        
        print('\nEpoch:{}, Train Loss:{}, Train R2:{}, Val Loss:{}, Val R2:{}'.format(epoch+1, train_loss, train_r2, val_loss, val_r2))

        if val_r2>best_r2:
            best_r2 = val_r2
            bestEp = epoch+1
            best_model_wts = copy.deepcopy(model.state_dict())

    print('\nBest performance at epoch {} with validation R2 {}.'.format(bestEp, best_r2))
    print('Testing with best model')
    model.load_state_dict(best_model_wts)
    test_r2 = test_model(model, X_test, y_test, args.batch_size, device)
    print('Test R2 = {}'.format(test_r2))

    savepath = './results/'

    if not os.path.exists(savepath):
        os.makedirs(savepath)

    suffix = 'test_r2-' + str(float("{0:.4f}".format(test_r2))) + \
        '_val-r2-' + str(float("{0:.4f}".format(best_r2))) + \
        '_lr-' + str(args.learning_rate) + \
        '_e-' + str(args.num_epochs) + \
        '_rgl-' + str(args.reg_loss) + \
        '_rgwt-' + str(args.reg_loss_wt) + \
        '_bs-' + str(args.batch_size) + \
        '_dr-' + str(args.dropout_rate) + \
        '_l2-' + str(args.dense_l2)

    df_save = pd.DataFrame({'train_loss':training_loss, 'val_loss':validation_loss, 'val_r2':validation_r2})
    df_save.to_csv(open(savepath + suffix + '.csv', 'w'))

    loss_dir = './results/losses/'

    if not os.path.exists(loss_dir):
        os.makedirs(loss_dir)
    
    fig1 = plt.figure()
    plt.plot(training_loss, label='Training Loss')
    plt.plot(validation_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('Losses')
    fig1.savefig(loss_dir+suffix+'_losses.png', dpi=300)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Neuro-fuzzy methods for predicting student academic performance.")
    parser.add_argument('-sub', '--subject', help="choose between maths(1) and portugese(0)", default=1, type=bool)
    parser.add_argument('-opt', '--optimizer', help="if 0 optimizer changes to SGD", default=1, type=bool)
    parser.add_argument('-lr', '--learning_rate', help="learning rate for Adam", default=1e-3, type=float)
    parser.add_argument('-reg_l', '--reg_loss', help="choose between L1(0) and smooth L1(1) as regression loss", default=1, type=int)
    parser.add_argument('-reg_wt', '--reg_loss_wt', help="choose weight between 0 and 1 for regressor loss", default=0.7, type=float)
    parser.add_argument('-e', '--num_epochs', help="epochs to run", default=300, type=int)
    parser.add_argument('-bs', '--batch_size', help="batch size used throughout", default=8, type=int)
    parser.add_argument('-dr', '--dropout_rate', help="dropout rate for encoder and classifier", default=0.1, type=float)
    parser.add_argument('-l2', '--dense_l2', help="l2 regularization for regressor layer", default=0.0, type=float)

    args = parser.parse_args()
    main(args)
