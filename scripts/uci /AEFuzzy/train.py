import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import os

from sklearn.model_selection import train_test_split
from utils import *
from models import *

import torch.optim as optim

import copy

import torch.backends.cudnn as cudnn
cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"]="0"

seed = 42
np.random.seed(seed)

data_size = 395
num_fuzz_var = 3

w1=0.1
w2=0.9	

lr=2e-3

batchsize = 8

num_epochs = 500

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = AEFuzzy().to(device)

criterion_decoder = nn.MSELoss()
criterion_regressor = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)

Read = Create_Dataset(data_size,num_fuzz_var)

Read.read_data()

# Membership function
rng_absence = np.arange(0, 76, 1)
rng_grades = np.arange(0,21,1)

Read.fuzzy_mf(rng_absence, rng_grades) 	

# Prepare data
Read.prepare_data(rng_absence, rng_grades)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(Read.X, Read.Y, test_size=0.2, random_state=seed)

print(X_train.shape)

# exit()

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train)
y_test = torch.FloatTensor(y_test)
y_train= torch.squeeze(y_train,dim=1)
y_test= torch.squeeze(y_test,dim=1)

best_r2 = 0.0

training_loss = []
testing_loss = []

for epoch in range(num_epochs):

    if (epoch+1)%200==0:
        optimizer.param_groups[0]['lr']/=5
        print('\nNew learning rate :', optimizer.param_groups[0]['lr'])

    rn = torch.randperm(X_train.shape[0])
    
    shuffled_data, shuffled_label = X_train[rn], y_train[rn]

    model, train_loss = train(model, shuffled_data, shuffled_label, criterion_decoder, criterion_regressor, optimizer, batchsize, w1, w2, device)
    
    model_r2, test_loss = testNet(model, X_test, y_test, criterion_decoder, criterion_regressor, batchsize, device)

    training_loss.append(train_loss)

    testing_loss.append(test_loss)

    
    print('Epoch : {}, Training loss : {}, Test R2 : {}'.format(epoch+1, train_loss, model_r2))

    if model_r2>best_r2:
        best_r2 = model_r2
        bestEp = epoch+1
        best_model = copy.deepcopy(model.state_dict())

print('\nBest performance at epoch {} with R2-score {}.'.format(bestEp, best_r2))
torch.save(best_model, './weights/weights.pth')

fig1 = plt.figure()
plt.plot(training_loss, label='Training Loss')
plt.plot(testing_loss, label='Test Loss')
plt.legend(loc='upper right')
plt.xlabel('Epoch')
plt.ylabel('Losses')
fig1.savefig('Losses.png', dpi=300)