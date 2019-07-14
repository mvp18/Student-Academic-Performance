import skfuzzy as fuzz
import time
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable, Function
from sklearn.metrics import r2_score

class Create_Dataset():
    def __init__(self, data_size, num_fuzz_var, mf_absence={}, mf_grades={}, rng_absence, rng_grades, subject=0):
        
        self.data_size = data_size
        self.num_fuzz_var = num_fuzz_var
        self.mf_absence = mf_absence
        self.mf_grades = mf_grades
        self.rng_absence = rng_absence
        self.rng_grades = rng_grades
        self.subject = subject

        self.mf_absence['lo'] = fuzz.trapmf(rng_absence, [0, 0, 10, 25])
        self.mf_absence['md'] = fuzz.trimf(rng_absence, [10, 25, 50])
        self.mf_absence['hi'] = fuzz.trapmf(rng_absence, [40, 50, 75, 75])

        self.mf_grades['lo'] = fuzz.trapmf(rng_grades, [0, 0, 5, 10])
        self.mf_grades['md'] = fuzz.trimf(rng_grades, [5, 10, 15])
        self.mf_grades['hi'] = fuzz.trapmf(rng_grades, [10, 15, 20, 20])

        self.read_data()

    def read_data(self):
        
        if self.subject:
            dataset = pd.read_csv('../dataset/student-mat.csv', sep=';')
        else:
            dataset = pd.read_csv('../dataset/student-por.csv', sep=';')

        imp_features = dataset.drop(['school', 'sex', 'reason'], axis=1)
        address_mapping = {"U":0, "R":1}
        famsize_mapping = {"LE3":0,"GT3":1}
        Pstatus_mapping = {"T":0,"A":1}
        Mjob_mapping = {"teacher":0,"health":1,"services":2,"at_home":3,"other":4}
        Fjob_mapping = {"teacher":0,"health":1,"services":2,"at_home":3,"other":4}
        schoolsup_mapping = {"yes":0,"no":1}
        famsup_mapping = {"yes":0,"no":1}
        paid_mapping = {"yes":0,"no":1}
        activities_mapping = {"yes":0,"no":1}
        nursery_mapping = {"yes":0,"no":1}
        higher_mapping = {"yes":0,"no":1}
        internet_mapping = {"yes":0,"no":1}
        romantic_mapping = {"yes":0,"no":1}
        guardian_mapping = {"mother":0,"father":1,"other":2}
        
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
        self.Y = data_np_array[:, data_np_array.shape[1]-1]

        def __prepare_data():
            
            x = np.zeros((self.data_size,self.num_fuzz_var*3))

            for i in range(self.data_size):
                
                x[i,0]=fuzz.interp_membership(self.rng_absence, self.mf_absence['lo'], self.X_fuzzy[i,0])
                x[i,1]=fuzz.interp_membership(self.rng_absence, self.mf_absence['md'], self.X_fuzzy[i,0])
                x[i,2]=fuzz.interp_membership(self.rng_absence, self.mf_absence['hi'], self.X_fuzzy[i,0])

            for i in range(self.data_size):
                
                for j in range(1,self.num_fuzz_var):

                    x[i,j*3] = fuzz.interp_membership(self.rng_grades, self.mf_grades['lo'], self.X_fuzzy[i,j]) 
                    x[i,j*3+1] = fuzz.interp_membership(self.rng_grades, self.mf_grades['md'], self.X_fuzzy[i,j])
                    x[i,j*3+2] = fuzz.interp_membership(self.rng_grades, self.mf_grades['hi'], self.X_fuzzy[i,j])
            
            return np.concatenate((self.X_crisp,x),axis=1)

        self.X = __prepare_data()

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def train_epoch(model, input_array, label_array, criterion_decoder, criterion_regressor, optimizer, batch_size, w1, device):
    
    model.train()

    runningLoss = 0.0 

    trainsize = int(input_array.shape[0])

    y_pred=[]
    y_true=[]        
    
    for i in range(0, trainsize, batch_size):

        if i+batch_size<=trainsize:
            inputs = input_array[i:i+batch_size]
            labels = label_array[i:i+batch_size]
        else:
            inputs = input_array[i:]
            labels = label_array[i:]

        if y_true:
            y_true = np.vstack([y_true, labels])
        else:
            y_true = labels
        
        inputs, labels = Variable(torch.from_numpy(inputs).float().to(device)), Variable(torch.from_numpy(labels).float().to(device))                       
        # Initialize gradients to zero
        optimizer.zero_grad()
        # Feed-forward
        output_decoder, output_regressor = model(inputs)
        
        output_regressor = torch.squeeze(output_regressor, dim=1)
        
        if y_pred:
            y_pred = np.vstack([y_pred, output_regressor.detach().cpu().numpy()])
        else:
            y_pred = output_regressor.detach().cpu().numpy()
        # Compute loss/error
        loss =  (1-w1)*criterion_decoder(output_decoder, inputs)+w1*criterion_regressor(output_regressor, labels)
        # Accumulate loss per batch
        runningLoss += loss.item()
        # Backpropagate loss and compute gradients
        loss.backward()
        # Update the network parameters
        optimizer.step()

    train_r2 = r2_score(y_true, y_pred)

    return(model, runningLoss/float(trainsize/batch_size), train_r2)


def val_epoch(model, input_array, label_array, criterion_decoder, criterion_regressor, batch_size, device):

    model.eval()

    valsize = int(input_array.shape[0])

    runningLoss = 0.0

    y_pred=[]
    y_true=[] 

    with torch.no_grad():      

        for i in range(0, valsize, batch_size):

            if i+batch_size<=valsize:
                inputs = input_array[i:i+batch_size]
                labels = label_array[i:i+batch_size]
            else:
                inputs = input_array[i:]
                labels = label_array[i:]

            if y_true:
                y_true = np.vstack([y_true, labels])
            else:
                y_true = labels

            inputs, labels = torch.from_numpy(inputs).float().to(device), torch.from_numpy(labels).float().to(device)                      
            # Feed-forward
            output_decoder, output_regressor = model(inputs)

            output_regressor = torch.squeeze(output_regressor, dim=1)

            loss = criterion_decoder(output_decoder, inputs)+criterion_regressor(output_regressor, labels)
            # Accumulate loss per batch
            runningLoss += loss.item()

            if y_pred:
                y_pred = np.vstack([y_pred, output_regressor.cpu().numpy()])
            else:
                y_pred = output_regressor.cpu().numpy()

    val_r2 = r2_score(y_true, y_pred)

    return (runningLoss/float(valsize/batchsize), val_r2)

def test_model(model, input_array, label_array, batch_size, device):

    model.eval()

    testsize = int(input_array.shape[0])

    y_pred=[]
    y_true=[]

    with torch.no_grad():      

        for i in range(0, testsize, batch_size):

            if i+batch_size<=testsize:
                inputs = input_array[i:i+batch_size]
                labels = label_array[i:i+batch_size]
            else:
                inputs = input_array[i:]
                labels = label_array[i:]

            if y_true:
                y_true = np.vstack([y_true, labels])
            else:
                y_true = labels

            inputs = torch.from_numpy(inputs).float().to(device) 
            # Feed-forward
            output_decoder, output_regressor = model(inputs)

            output_regressor = torch.squeeze(output_regressor, dim=1)

            if y_pred:
                y_pred = np.vstack([y_pred, output_regressor.cpu().numpy()])
            else:
                y_pred = output_regressor.cpu().numpy()

    test_r2 = r2_score(y_true, y_pred)

    return test_r2
    