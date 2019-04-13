
from __future__ import print_function

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.optim as optim
import torch.nn.functional as F
import time
import argparse
from torch.autograd import Variable
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import copy


class anfis_model(nn.Module):
    def __init__(self,num_var,num_set):
        super(anfis_model,self).__init__()
        self.num_var = num_var
        self.num_set = num_set 
        init_premise_params = torch.rand((sum(num_set),3))
        init_conseq_params = torch.randn((np.prod(num_set)*(num_var+1),1))
        self.premise_params = Parameter(init_premise_params,requires_grad=True)
        self.conseq_params  = Parameter(init_conseq_params)

    def forward(self, input, target=torch.randn((1,1))):
        batchsize=input.shape[1]
        #layer 1
        mem_out = torch.zeros((sum(self.num_set),batchsize))
        k=0
        for i in range(self.num_var):
            for j in range(self.num_set[i]):
                x = ((input[i]-self.premise_params[k,2])/self.premise_params[k,0])**2
                mem_out[k] = 1/(1 + x**self.premise_params[k,1])
                k += 1
     
        #layer 2
        total = np.prod(self.num_set)
        fire_strength = torch.ones((total,batchsize))
        k=0
        for i in range(self.num_var):
            k1=0
            for j in range(self.num_set[i]):
                k2 = k1
                k1 += int(total/self.num_set[i])
                fire_strength[k2:k1] *= mem_out[k]
                k += 1        
        #layer 3
        fire_strength = fire_strength/torch.sum(fire_strength)

        Train = self.training
        #layer 4 

        input_with_bias = torch.cat((input,torch.ones((1,batchsize))),0)
        fire_strength_input = input_with_bias*fire_strength[0]
        for i in range(1,total):
            fire_strength_input = torch.cat((fire_strength_input,input_with_bias*fire_strength[j]),0)   
        A = torch.transpose(fire_strength_input,0,1)
        
        output = torch.mm(A,self.conseq_params)  

        return output


# dataset = pd.read_csv('./fuzzy_kaggle.csv')

# data_np_array = dataset.values


# X = data_np_array[:,1:5]
# Y = data_np_array[:,data_np_array.shape[1]-1:data_np_array.shape[1]]



# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

# X_train = torch.FloatTensor(X_train)
# X_test = torch.FloatTensor(X_test)
# y_train = torch.FloatTensor(y_train)
# y_test = torch.FloatTensor(y_test)
# y_train= torch.squeeze(y_train,dim=1)
# y_test= torch.squeeze(y_test,dim=1)



def Normalize_train(X):
    mean_list=[]
    std_list=[]
    for i in range(X.shape[1]):
        mean = torch.mean(X[:,i])
        std = torch.std(X[:,i])
        mean_list.append(mean.item())
        std_list.append(std.item())
        X[:,i] = (X[:,i]-mean)/std
    return X,mean_list,std_list

def Normalize_test(X,mean_list,std_list):
    for i in range(X.shape[1]):
        mean = mean_list[i]
        std = std_list[i]
        X[:,i] = (X[:,i]-mean)/std
    return X

# X_train,mean,std = Normalize_train(X_train)
# X_test = Normalize_test(X_test,mean,std)


# num_var = X.shape[1]
# num_set = [2 for i in range (num_var)]
# net=anfis_model(num_var,num_set)


# BatchSize=2
# iterations = 20
# lr=1e-2
# dataSize2=X_train.shape[0]
# testsize = X_test.shape[0]
# criterion1=nn.MSELoss()
# optimizer= optim.Adam(net.parameters(),lr=lr,betas=(0.9,0.99))

# trainloss = []
# testloss = []
# start = time.time()


# for epoch in range(iterations):
#     epochStart = time.time()
#     runningtrainloss = 0
#     runningtestloss = 0
#     runningaccuracy = 0
#     net.train(True)   #train start
#     rn=torch.randperm(dataSize2)
#     X=X_train[rn]
#     Y=y_train[rn]
    
#     for i in range(0,dataSize2,BatchSize):
#         if i+BatchSize>dataSize2:
#             inputs=X[i:]
#             labels=Y[i:]
#         else:
#             inputs=X[i:i+BatchSize]
#             labels=Y[i:i+BatchSize]
        
#         inputs, labels = Variable(torch.transpose(inputs,1,0)), Variable(labels.unsqueeze(1))
        
#         output1 = net(inputs,labels)
#         loss = criterion1(output1, labels)
        
#         optimizer.zero_grad()
        
#         loss.backward()
        
#         optimizer.step()
        
#         runningtrainloss += loss.data[0]
        
#     avgTrainloss = runningtrainloss/dataSize2
#     trainloss.append(avgTrainloss)
    
#     net.eval()
#     for i in range(0,testsize,BatchSize):
#         if i+BatchSize>dataSize2:
#             inputs=X_test[i:]
#             labels=y_test[i:]
#         else:
#             inputs=X_test[i:i+BatchSize]
#             labels=y_test[i:i+BatchSize]
        
#         inputs, labels = Variable(torch.transpose(inputs,1,0)), Variable(labels.unsqueeze(1))
        
#         output1 = net(inputs)
#         loss = criterion1(output1, labels)
#         corrects = ((output1.round()==labels).sum(0)).float() 
#         runningtestloss += loss.data[0]
#         runningaccuracy += corrects 
        
#     avgTestloss = runningtestloss/testsize
#     avgTestaccuracy = runningaccuracy/testsize
#     testloss.append(avgTestloss)
    
#     fig1 = plt.figure(1)        
#     plt.plot(range(epoch+1),trainloss,'g--',label='train') 
#     plt.plot(range(epoch+1),testloss,'r--',label='test')
#     if epoch==0:
#         plt.legend(loc='upper left')
#         plt.xlabel('Epochs')
#         plt.ylabel('Loss')   
      
#     epochEnd = time.time()-epochStart
#     print('At Iteration: {:.0f} /{:.0f}  ;  Training Loss: {:.6f}; Test loss: {:.6f};Test_acc: {:.6f};Time consumed: {:.0f}m {:.0f}s '.format(epoch + 1,iterations,avgTrainloss,avgTestloss,avgTestaccuracy.squeeze(0),epochEnd//60,epochEnd%60))
# end = time.time()-start
# print('Training completed in {:.0f}m {:.0f}s'.format(end//60,end%60))



def testNet(model, input_tensor, label_tensor, batchsize):

    model.eval()

    testsize = int(input_tensor.shape[0])

    corrects = 0.0
    
    with torch.no_grad():      
    
        for i in range(0, testsize, batchsize):

            if i+batchsize<=testsize:
                inputs = input_tensor[i:i+batchsize]
                labels = label_tensor[i:i+batchsize]

            else:
                inputs = input_tensor[i:]
                labels = label_tensor[i:]

            inputs, labels = Variable(torch.transpose(inputs,1,0)), Variable(labels.unsqueeze(1))                      

            # Feed-forward
            output1 = model(inputs)

            corrects += ((output1.round()==labels).sum(0)).float()
    return corrects/float(testsize)

# net_dict = copy.deepcopy(net.state_dict())
# torch.save(net_dict,'./anfis_dict.pth')

# accuracy=testNet(net, X_test, y_test, 2)
# print("The accuracy is",accuracy[0].item()*100,"%")


# net_dict = copy.deepcopy(net.state_dict())
# torch.save(net_dict,'./anfis_dict.pth')
# net.load_state_dict(torch.load('./anfis_dict.pth',map_location=lambda storage,loc:storage))

# x = np.linspace(0.0, 2.0, num=100)
# premise_params = net.premise_params.detach().numpy()
# k=0
# s=['-r','-g','-b','-c','-y']
# fuz_var=['raised hands','visited resources','announcements viewed','discussion']
# fuz_mem=['membership value 1','membership value 2']
# for i in range(num_var):
#     for j in range(num_set[i]):
#         y1 = ((x - premise_params[k,2])/premise_params[k,0])**2
#         y = 1/(1+y1**premise_params[k,1])
#         plt.figure(k+1)
#         plt.plot(x,y,s[k%5])
#         plt.xlabel(fuz_var[i])
#         plt.ylabel(fuz_mem[j])
#         plt.savefig(str(k)+".png")
#         k+=1

#print(net.state_dict)
