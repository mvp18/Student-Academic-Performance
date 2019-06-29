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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    rng = np.arange(0, 101, 1)

    dataloader = Create_Dataset(num_fuzz_var=4, rng=rng)

    X_trval, X_test, y_trval, y_test = train_test_split(dataloader.X, dataloader.Y, test_size=0.1, random_state=seed, stratify=dataloader.Y)

    X_train, X_val, y_train, y_val = train_test_split(X_trval, y_trval, test_size=0.11, random_state=seed, stratify=y_trval)

    model = AEFuzzy(data_dim=X_train.shape[1], num_classes=3, drop_rate=args.dropout_rate, pretrained=args.pretrained)
    
    if args.pretrained:
        print('Loading pretrained weights for encoder and decoder')
        encoder_wts = torch.load('./weights/encoder_wts.pth')
        decoder_wts = torch.load('./weights/decoder_wts.pth')
        model.encoder.load_state_dict(encoder_wts)
        model.decoder.load_state_dict(decoder_wts)
    else:
        print('All weights initialized from a xavier distribution')
        model.apply(init_weights)
    
    model.to(device)
    print('Model Summary:\n')
    print(summary(model, (X_train.shape[1],)))

    criterion_decoder = nn.MSELoss()
    if args.weighted_loss:
        class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
        criterion_classifier = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights).float().to(device))
    else:
        criterion_classifier = nn.CrossEntropyLoss()
    
    if args.optimizer:
        optimizer = optim.Adam([{'params': model.classifier.parameters(), 'weight_decay':args.dense_l2} 
                           ], lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    else:
        optimizer = optim.SGD([{'params': model.classifier.parameters(), 'weight_decay':args.dense_l2} 
                           ], lr=args.learning_rate, nesterov=True, momentum=0.9)

    training_loss = []
    validation_loss = []
    validation_acc = []

    best_accuracy = 0.0

    for epoch in range(args.num_epochs):

        if (epoch+1)%50==0:
            optimizer.param_groups[0]['lr']/=2
            print('\nNew learning rate :', optimizer.param_groups[0]['lr'])

        rn = torch.randperm(X_train.shape[0])
        shuffled_data, shuffled_label = X_train[rn], y_train[rn]

        model, train_loss, train_acc = train_epoch(model, shuffled_data, shuffled_label, criterion_decoder, criterion_classifier, 
                                        optimizer, args.batch_size, args.classifier_loss_wt, device)
        val_loss, val_accuracy = val_epoch(model, X_val, y_val, criterion_decoder, criterion_classifier, args.batch_size, device)

        training_loss.append(train_loss)
        validation_loss.append(val_loss)
        validation_acc.append(val_accuracy)
        
        print('\nEpoch:{}, Train Loss:{}, Train Acc:{}, Val Loss:{}, Val Acc:{}'.format(epoch+1, train_loss, train_acc, val_loss, val_accuracy))

        if val_accuracy>best_accuracy:
            best_accuracy = val_accuracy
            bestEp = epoch+1
            best_model_wts = copy.deepcopy(model.state_dict())

    print('\nBest performance at epoch {} with validation accuracy {}.'.format(bestEp, best_accuracy))
    print('Testing with best model')
    model.load_state_dict(best_model_wts)
    test_acc = test_model(model, X_test, y_test, args.batch_size, device)
    print('Test accuracy = ',test_acc)

    savepath = './results/'

    if not os.path.exists(savepath):
        os.makedirs(savepath)

    suffix = 'test_acc-' + str(float("{0:.4f}".format(test_acc.item()))) + \
        '_val-acc-' + str(float("{0:.4f}".format(best_accuracy.item()))) + \
        '_pt-' + str(bool(args.pretrained)) + \
        '_lr-' + str(args.learning_rate) + \
        '_e-' + str(args.num_epochs) + \
        '_wl-' + str(args.weighted_loss) + \
        '_clwt-' + str(args.classifier_loss_wt) + \
        '_bs-' + str(args.batch_size) + \
        '_dr-' + str(args.dropout_rate) + \
        '_l2-' + str(args.dense_l2)

    df_save = pd.DataFrame({'train_loss':training_loss, 'val_loss':validation_loss, 'val_acc':validation_acc})
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
    parser.add_argument('-pt', '--pretrained', help="if 0 all weights are initialized from a xavier distribution", default=1, type=bool)
    parser.add_argument('-opt', '--optimizer', help="if 0 optimizer changes to SGD", default=1, type=bool)
    parser.add_argument('-lr', '--learning_rate', help="learning rate for Adam", default=1e-3, type=float)
    parser.add_argument('-wl', '--weighted_loss', help="whether or not to use weighted cross-entropy loss", default=False, type=bool)
    parser.add_argument('-cl_wt', '--classifier_loss_wt', help="choose weight between 0 and 1 for classifier loss", default=0.7, type=float)
    parser.add_argument('-e', '--num_epochs', help="epochs to run", default=300, type=int)
    parser.add_argument('-bs', '--batch_size', help="batch size used throughout", default=8, type=int)
    parser.add_argument('-dr', '--dropout_rate', help="dropout rate for encoder and classifier", default=0.1, type=float)
    parser.add_argument('-l2', '--dense_l2', help="l2 regularization for last dense classifier layer", default=0.0, type=float)

    args = parser.parse_args()
    main(args)
