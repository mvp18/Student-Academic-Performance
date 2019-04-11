import torch
import numpy as np
import time


from sklearn.cross_validation import train_test_split
import read
import util
from model import AE


data_size = 480
num_fuzz_var = 4

Read = read.Read(data_size,num_fuzz_var)
Read.read_data()

# Membership function
rng = np.arange(0, 101, 1)
mf = util.fuzzy_mf(rng) 	

# Prepare data
Read.prepare_data(mf,rng)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(Read.X, Read.Y, test_size=0.2, random_state=42, stratify=Read.Y)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train)
y_test = torch.FloatTensor(y_test)
y_train= torch.squeeze(y_train,dim=1)
y_test= torch.squeeze(y_test,dim=1)

# Net
net = AE()

util.train(net,X_train, y_train)

accuracy=util.testNet(net, X_test, y_test, 1)

print(accuracy)

