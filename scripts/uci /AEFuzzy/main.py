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
from sklearn.metrics import r2_score, mean_absolute_error

import torch.backends.cudnn as cudnn
cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def main(args):
    seed = args.rand_seed
    np.random.seed(seed)

    torch.manual_seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Membership functions for 3 fuzzy variables - absence and 2 sets of grades
    rng_absence = np.arange(0, 76, 1)
    rng_grades = np.arange(0,21,1)

    dataloader = Create_Dataset(num_fuzz_var=3, rng_absence=rng_absence, rng_grades=rng_grades, subject=args.subject)

    # Train-val-test split
    X_trval, X_test, y_trval, y_test = train_test_split(dataloader.X, dataloader.Y, test_size=0.1, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_trval, y_trval, test_size=0.11, random_state=seed)

    model = AEFuzzy(data_dim=X_train.shape[1], drop_rate=args.dropout_rate).to(device)

    if args.pretrained:
        print('Loading pretrained weights for encoder and decoder')
        encoder_wts = torch.load('./weights/encoder_wts.pth')
        decoder_wts = torch.load('./weights/decoder_wts.pth')
        model.encoder.load_state_dict(encoder_wts)
        model.decoder.load_state_dict(decoder_wts)
    else:
        print('All weights initialized from a xavier distribution')
        model.apply(init_weights)

    print('Model Summary:\n')
    print(summary(model, (X_train.shape[1],)))

    criterion_decoder = nn.MSELoss()
    loss_dict = {0:nn.MSELoss(), 1:nn.L1Loss(), 2:nn.SmoothL1Loss()}
    criterion_regressor=loss_dict[args.reg_loss]

    params_to_optimize=[]
    params_to_optimize.append({'params':model.encoder.parameters()})
    params_to_optimize.append({'params':model.decoder.parameters()})
    params_to_optimize.append({'params':model.regressor.parameters(), 'weight_decay':args.dense_l2})

    if args.optimizer:
        optimizer = optim.Adam(params_to_optimize, lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    else:
        optimizer = optim.SGD(params_to_optimize, lr=args.learning_rate, nesterov=True, momentum=0.9)

    metric = {'mae':mean_absolute_error, 'r2':r2_score}[args.metric]

    training_loss = []
    validation_loss = []
    validation_score = []

    if args.metric=='r2':
        best_score = 0.0
    else:
        best_score = 999

    for epoch in range(args.num_epochs):

        if (epoch+1)%50==0:
            optimizer.param_groups[0]['lr']/=2
            print('\nNew learning rate :', optimizer.param_groups[0]['lr'])

        rn = torch.randperm(X_train.shape[0])
        shuffled_data, shuffled_label = X_train[rn], y_train[rn]

        model, train_loss, train_score = train_epoch(model, shuffled_data, shuffled_label, criterion_decoder, criterion_regressor, 
                                                     optimizer, metric, args.batch_size, args.reg_loss_wt, device)
        val_loss, val_score = val_epoch(model, X_val, y_val, criterion_decoder, criterion_regressor, metric, args.batch_size, device)

        training_loss.append(train_loss)
        validation_loss.append(val_loss)
        validation_score.append(val_score)
        
        print('\nEpoch:{}, Train Loss:{}, Train {}:{}, Val Loss:{}, Val {}:{}'.format(epoch+1, train_loss, args.metric, train_score, val_loss, args.metric, val_score))

        if args.metric=='mae':
            if val_score<best_score:
                best_score = val_score
                bestEp = epoch+1
                best_model_wts = copy.deepcopy(model.state_dict())
        else:
            if val_score>best_score:
                best_score = val_score
                bestEp = epoch+1
                best_model_wts = copy.deepcopy(model.state_dict())

    print('\nBest performance at epoch {} with validation {} {}.'.format(bestEp, args.metric, best_score))
    print('Testing with best model')
    model.load_state_dict(best_model_wts)
    test_score = test_model(model, X_test, y_test, metric, args.batch_size, device)
    print('Test {} = {}'.format(args.metric, test_score))

    subject={1:'maths', 0:'portugese'}[args.subject]
    savepath = './results/'+subject+'/'+args.metric+'/'

    if not os.path.exists(savepath):
        os.makedirs(savepath)

    suffix = 'val_score-' + str(float("{0:.4f}".format(best_score))) + \
        '_test_score-' + str(float("{0:.4f}".format(test_score))) + \
        '_pt-' + str(bool(args.pretrained)) + \
        '_lr-' + str(args.learning_rate) + \
        '_rgl-' + str(args.reg_loss) + \
        '_rgwt-' + str(args.reg_loss_wt) + \
        '_opt-' + str(args.optimizer) + \
        '_bs-' + str(args.batch_size) + \
        '_dr-' + str(args.dropout_rate) + \
        '_l2-' + str(args.dense_l2)

    df_save = pd.DataFrame({'train_loss':training_loss, 'val_loss':validation_loss, 'val_score':validation_score})
    df_save.to_csv(open(savepath + suffix + '.csv', 'w'))

    loss_dir = savepath+'losses/'

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
    parser.add_argument('-seed', '--rand_seed', help="random state for numpy, torch and sklearn", default=42, type=int)
    parser.add_argument('-sub', '--subject', help="choose between maths(1) and portugese(0)", default=1, type=int)
    parser.add_argument('-pt', '--pretrained', help="if 1 load autoencoder wts", default=0, type=int)
    parser.add_argument('-opt', '--optimizer', help="if 0 optimizer changes to SGD", default=1, type=int)
    parser.add_argument('-lr', '--learning_rate', help="learning rate for Adam", default=1e-3, type=float)
    parser.add_argument('-reg_l', '--reg_loss', help="choose between L2(0), L1(1) and smooth L1(2) as regression loss", default=0, type=int)
    parser.add_argument('-reg_wt', '--reg_loss_wt', help="choose weight between 0 and 1 for regressor loss", default=0.7, type=float)
    parser.add_argument('-met', '--metric', help="choose between MAE('mae') and R2('r2')", default='mae', type=str)
    parser.add_argument('-e', '--num_epochs', help="epochs to run", default=500, type=int)
    parser.add_argument('-bs', '--batch_size', help="batch size used throughout", default=32, type=int)
    parser.add_argument('-dr', '--dropout_rate', help="dropout rate for encoder and classifier", default=0.0, type=float)
    parser.add_argument('-l2', '--dense_l2', help="l2 regularization for regressor layer", default=0.0, type=float)

    args = parser.parse_args()
    main(args)
