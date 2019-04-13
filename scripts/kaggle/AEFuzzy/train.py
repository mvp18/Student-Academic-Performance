import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import os

from sklearn.model_selection import train_test_split
from utils import *
from models import *

import torch.optim as optim

import torch.backends.cudnn as cudnn
cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import copy


seed = 42
np.random.seed(seed)

data_size = 480
num_fuzz_var = 4

w1=0.3
w2=0.7	

lr=1e-3

batchsize = 8

num_epochs = 500

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model = AEFuzzy().to(device)

criterion_decoder = nn.MSELoss()
criterion_classifier = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)

Read = Create_Dataset(data_size,num_fuzz_var)

Read.read_data()

# Membership function
rng = np.arange(0, 101, 1)

Read.fuzzy_mf(rng) 	

# Prepare data
Read.prepare_data(rng)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(Read.X, Read.Y, test_size=0.2, random_state=seed, stratify=Read.Y)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train)
y_test = torch.FloatTensor(y_test)
y_train= torch.squeeze(y_train,dim=1)
y_test= torch.squeeze(y_test,dim=1)

best_accuracy = 0.0

training_loss = []
testing_loss = []

for epoch in range(num_epochs):

    if (epoch+1)%200==0:
        optimizer.param_groups[0]['lr']/=5
        print('\nNew learning rate :', optimizer.param_groups[0]['lr'])

    rn = torch.randperm(X_train.shape[0])
    shuffled_data, shuffled_label = X_train[rn], y_train[rn]

    model, train_loss = train(model, shuffled_data, shuffled_label, criterion_decoder, criterion_classifier, optimizer, batchsize, w1, w2, device)
    
    model_accuracy, test_loss = testNet(model, X_test, y_test, criterion_decoder, criterion_classifier, 1, device)

    training_loss.append(train_loss)

    testing_loss.append(test_loss)
    
    print('Epoch : {}, Training loss : {}, Test Accuracy : {}'.format(epoch+1, train_loss, model_accuracy))

    if model_accuracy>best_accuracy:
        best_accuracy = model_accuracy
        bestEp = epoch+1
        best_model = copy.deepcopy(model.state_dict())

print('\nBest performance at epoch {} with accuracy {}.'.format(bestEp, best_accuracy))
torch.save(best_model, './weights/weights.pth')

fig1 = plt.figure()
plt.plot(training_loss, label='Training Loss')
plt.plot(testing_loss, label='Test Loss')
plt.legend(loc='upper right')
plt.xlabel('Epoch')
plt.ylabel('Losses')
fig1.savefig('Losses.png', dpi=300)