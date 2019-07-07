import skfuzzy as fuzz
import time
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable, Function
# from sklearn.metrics import r2_score

class Create_Dataset:
    def __init__(self, data_size, num_fuzz_var, mf={}):
        
        self.data_size = data_size
        self.num_fuzz_var = num_fuzz_var

    def fuzzy_mf(self, rng_absence, rng_grades):
    
        self.mf_absence = {}
        
        self.mf_absence['lo'] = fuzz.trapmf(rng_absence, [0, 0, 10, 25])
        self.mf_absence['md'] = fuzz.trimf(rng_absence, [10, 25, 50])
        self.mf_absence['hi'] = fuzz.trapmf(rng_absence, [40, 50, 75, 75])

        self.mf_grades = {}

        self.mf_grades['lo'] = fuzz.trapmf(rng_grades, [0, 0, 5, 10])
        self.mf_grades['md'] = fuzz.trimf(rng_grades, [5, 10, 15])
        self.mf_grades['hi'] = fuzz.trapmf(rng_grades, [10, 15, 20, 20])



    def read_data(self):
        
        dataset = pd.read_csv('../dataset/student-mat.csv', sep=';')

        imp_features = dataset.drop(['school', 'sex', 'reason'], axis=1)
        address_mapping = {"U":0.5, "R":1}
        famsize_mapping = {"LE3":0.5,"GT3":1}
        Pstatus_mapping = {"T":0.5,"A":1}
        Mjob_mapping = {"teacher":0.2,"health":0.4,"services":0.6,"at_home":0.8,"other":1.0}
        Fjob_mapping = {"teacher":0.2,"health":0.4,"services":0.6,"at_home":0.8,"other":1.0}
        schoolsup_mapping = {"yes":0.5,"no":1}
        famsup_mapping = {"yes":0.5,"no":1}
        paid_mapping = {"yes":0.5,"no":1}
        activities_mapping = {"yes":0.5,"no":1}
        nursery_mapping = {"yes":0.5,"no":1}
        higher_mapping = {"yes":0.5,"no":1}
        internet_mapping = {"yes":0.5,"no":1}
        romantic_mapping = {"yes":0.5,"no":1}
        guardian_mapping = {"mother":0.33,"father":0.66,"other":1}
        
        numeric_features = imp_features
        numeric_features['address'] = imp_features['address'].map(address_mapping)
        numeric_features['famsize'] = imp_features['famsize'].map(famsize_mapping)
        numeric_features['Pstatus'] = imp_features['Pstatus'].map(Pstatus_mapping)
        numeric_features['Mjob'] = imp_features['Mjob'].map(Mjob_mapping)
        numeric_features['Fjob'] = imp_features['Fjob'].map(Fjob_mapping)
        numeric_features['schoolsup'] = imp_features['schoolsup'].map(schoolsup_mapping)
        numeric_features['famsup'] = imp_features['famsup'].map(famsup_mapping)
        numeric_features['paid'] = imp_features['paid'].map(paid_mapping)
        numeric_features['activities'] = imp_features['activities'].map(activities_mapping)
        numeric_features['nursery'] = imp_features['nursery'].map(nursery_mapping)
        numeric_features['higher'] = imp_features['higher'].map(higher_mapping)
        numeric_features['internet'] = imp_features['internet'].map(internet_mapping)
        numeric_features['romantic'] = imp_features['romantic'].map(romantic_mapping)
        numeric_features['guardian'] = imp_features['guardian'].map(guardian_mapping)

        data_np_array = numeric_features.values

        self.X_crisp = data_np_array[:,:-4]
        self.X_fuzzy = data_np_array[:,-4:-1]
        self.Y = data_np_array[:,-1:]


    def prepare_data(self, rng_absence, rng_grades):
        
        x = np.zeros((self.data_size,self.num_fuzz_var*3))

        for i in range(self.data_size):
            
            x[i,0]=fuzz.interp_membership(rng_absence, self.mf_absence['lo'], self.X_fuzzy[i,0])
            x[i,1]=fuzz.interp_membership(rng_absence, self.mf_absence['md'], self.X_fuzzy[i,0])
            x[i,2]=fuzz.interp_membership(rng_absence, self.mf_absence['hi'], self.X_fuzzy[i,0])

        for i in range(self.data_size):
            
            for j in range(1,self.num_fuzz_var):

                x[i,j*3] = fuzz.interp_membership(rng_grades, self.mf_grades['lo'], self.X_fuzzy[i,j]) 
                x[i,j*3+1] = fuzz.interp_membership(rng_grades, self.mf_grades['md'], self.X_fuzzy[i,j])
                x[i,j*3+2] = fuzz.interp_membership(rng_grades, self.mf_grades['hi'], self.X_fuzzy[i,j])

        self.X=np.concatenate((self.X_crisp,x),axis=1)


def train(model, input_tensor, label_tensor, criterion_decoder, criterion_regressor, optimizer, batch_size, w1, w2, device):
    
    model.train()

    runningLoss = 0 

    trainsize = int(input_tensor.shape[0])

    batch_counter = 0        
    
    for i in range(0, trainsize, batch_size):

        if i+batch_size<=trainsize:
            inputs = input_tensor[i:i+batch_size]
            labels = label_tensor[i:i+batch_size]

        else:
            inputs = input_tensor[i:]
            labels = label_tensor[i:]

        inputs, labels = Variable(inputs.float().to(device)), Variable(labels.to(device))                       

        # Feed-forward
        output_decoder, output_regressor = model(inputs) # 

        output_regressor = torch.squeeze(output_regressor, dim=1)

        # Compute loss/error
        loss =  w1*criterion_decoder(output_decoder, inputs)+w2*criterion_regressor(output_regressor, labels)
        # Accumulate loss per batch
        runningLoss += loss.data.cpu()
        # Initialize gradients to zero
        optimizer.zero_grad()
        # Backpropagate loss and compute gradients
        loss.backward()
        # Update the network parameters
        optimizer.step()

        batch_counter += 1

    return(model, runningLoss/float(batch_counter))


def testNet(model, input_tensor, label_tensor, criterion_decoder, criterion_regressor, batchsize, device):

    model.eval()

    testsize = int(input_tensor.shape[0])

    R2_score = 0.0

    runningLoss = 0.0

    with torch.no_grad():      

        for i in range(0, testsize, batchsize):

            if i+batchsize<=testsize:
                inputs = input_tensor[i:i+batchsize]
                labels = label_tensor[i:i+batchsize]

            else:
                inputs = input_tensor[i:]
                labels = label_tensor[i:]

            inputs, labels = Variable(inputs.float().to(device)), Variable(labels.to(device))                      

            # Feed-forward
            output_decoder, output_regressor = model(inputs) # 

            output_regressor = torch.squeeze(output_regressor, dim=1)

            loss = criterion_decoder(output_decoder, inputs)+criterion_regressor(output_regressor, labels)
            # Accumulate loss per batch
            runningLoss += loss.data.cpu()

            # print('LABEL:{}, OP:{}'.format(labels, output_regressor))

            # print(output_regressor, labels)

            R2_score = R2_score + r2_score(output_regressor.cpu().numpy(), labels.cpu().numpy())

            # print(r2_score(output_regressor.cpu().numpy(), labels.cpu().numpy()))

    return (R2_score/float(testsize/batchsize), runningLoss/float(testsize/batchsize))


def r2_score(y_pred, y_true):
    
    mean = np.mean(y_true)

    numerator = np.mean((y_true - y_pred)**2)
    denominator = np.mean((y_true-mean)**2)

    return 1.0 - numerator/(denominator+1e-8)
    