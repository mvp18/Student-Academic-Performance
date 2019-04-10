import skfuzzy as fuzz
import time

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

def fuzzy_mf(rng):
	mf = {}
	mf['vlo'] = fuzz.trimf(rng, [0, 0, 25])
	mf['lo'] = fuzz.trimf(rng, [0, 25, 50])
	mf['md'] = fuzz.trimf(rng, [25, 50, 75])
	mf['hi'] = fuzz.trimf(rng, [50, 75, 100])
	mf['vhi'] = fuzz.trimf(rng, [75, 100, 100])

	return mf

def train(net,X,y):
	# Network Parameters
	BatchSize=8
	epochs = 20
	lr=1e-3
	w1=0.2
	w2=0.8
	data_size = X.shape[0]

	# Net
	criterion1=nn.MSELoss()
	criterion2=nn.CrossEntropyLoss()
	optimizer= optim.Adam(net.parameters(),lr=lr,betas=(0.9,0.99))

	trainloss = []
	start = time.time()

	for epoch in range(epochs):
	    epochStart = time.time()
	    runningloss = 0
	    net.train(True)   #train start
	    rn=torch.randperm(data_size)
	    X=X[rn]
	    Y=y[rn]
	    
	    for i in range(0,data_size,BatchSize):
	        if i+BatchSize>data_size:
	            inputs=X[i:]
	            labels=Y[i:]
	        else:
	            inputs=X[i:i+BatchSize]
	            #print(inputs.shape)
	            labels=Y[i:i+BatchSize]
	            #print(labels.shape)
	        
	        inputs, labels = Variable(inputs), Variable(labels)
	        
	        output1,output2 = net(inputs)
	        #print(output2.shape)

	        loss = w1*criterion1(output1, inputs)+w2*criterion2(output2,labels.long())
	        
	        optimizer.zero_grad()
	        
	        loss.backward()
	        
	        optimizer.step()
	        
	        runningloss += loss.item()
	        
	    avgTrainloss = runningloss/data_size
	    print (avgTrainloss)
	    trainloss.append(avgTrainloss)
	    
	    # fig1 = plt.figure(1)        
	    # plt.plot(range(epoch+1),trainloss,'r--',label='train')        
	    # if epoch==0:
	    #     plt.legend(loc='upper left')
	    #     plt.xlabel('Epochs')
	    #     plt.ylabel('Loss')   
	      
	    epochEnd = time.time()-epochStart
	    print('At Iteration: {:.0f} /{:.0f}  ;  Training Loss: {:.6f}; Time consumed: {:.0f}m {:.0f}s '\
	          .format(epoch + 1,epochs,avgTrainloss,epochEnd//60,epochEnd%60))
	end = time.time()-start
	print('Training completed in {:.0f}m {:.0f}s'.format(end//60,end%60))


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

            inputs, labels = Variable(inputs.float()), Variable(labels.long())                      

            # Feed-forward
            output1,output2 = model(inputs)

            _, predicted = torch.max(output2.data, 1)

            corrects += ((predicted==labels).sum(0)).float()

    return corrects/float(testsize)