import skfuzzy as fuzz
import time
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function

class Create_Dataset():
    
    def __init__(self, num_fuzz_var, rng, mf={}):
        
        self.num_fuzz_var = num_fuzz_var
        self.rng = rng
        self.mf = mf
        self.mf['vlo'] = fuzz.trimf(self.rng, [0, 0, 25])
        self.mf['lo'] = fuzz.trimf(self.rng, [0, 25, 50])
        self.mf['md'] = fuzz.trimf(self.rng, [25, 50, 75])
        self.mf['hi'] = fuzz.trimf(self.rng, [50, 75, 100])
        self.mf['vhi'] = fuzz.trimf(self.rng, [75, 100, 100])

        self.read_data()

    def read_data(self):
        
        dataset = pd.read_csv('../dataset/xAPI-Edu-Data.csv')

        self.data_size = dataset.shape[0]

        imp_features = dataset.drop(['gender', 'NationalITy', 'Semester', 'PlaceofBirth', 'GradeID', 'Topic', 'SectionID', 'Relation'], axis=1)
        
        stage_mapping = {"lowerlevel":0, "MiddleSchool":1, "HighSchool":2}
        survey_mapping = {"No":0, "Yes":1}
        satisfaction_mapping = {"Bad":0, "Good":1}
        absence_mapping = {"Under-7":0, "Above-7":1}
        class_mapping = {"L":0, "M":1, "H":2}
        
        numeric_features = imp_features
        numeric_features['StageID'] = imp_features['StageID'].map(stage_mapping)
        numeric_features['ParentAnsweringSurvey'] = imp_features['ParentAnsweringSurvey'].map(survey_mapping)
        numeric_features['ParentschoolSatisfaction'] = imp_features['ParentschoolSatisfaction'].map(satisfaction_mapping)
        numeric_features['StudentAbsenceDays'] = imp_features['StudentAbsenceDays'].map(absence_mapping)
        numeric_features['Class'] = imp_features['Class'].map(class_mapping)

        data_np_array = numeric_features.values
        
        print("Data size = ",data_np_array.shape)
        
        self.X1 = data_np_array[:,1:5]
        self.X2 = np.concatenate((data_np_array[:,0:1], data_np_array[:,5:8]), axis=1)
        self.Y = data_np_array[:,data_np_array.shape[1]-1]

        def __prepare_data():
            
            x = np.zeros((self.data_size,self.num_fuzz_var*5))

            for i in range(self.data_size):
                for j in range(self.num_fuzz_var):
                    x[i,j*5]=fuzz.interp_membership(self.rng, self.mf['vlo'], self.X1[i,j])
                    x[i,j*5+1]=fuzz.interp_membership(self.rng, self.mf['lo'], self.X1[i,j])
                    x[i,j*5+2]=fuzz.interp_membership(self.rng, self.mf['md'], self.X1[i,j])
                    x[i,j*5+3]=fuzz.interp_membership(self.rng, self.mf['hi'], self.X1[i,j])
                    x[i,j*5+4]=fuzz.interp_membership(self.rng, self.mf['vhi'], self.X1[i,j])

            return np.concatenate((x,self.X2),axis=1)

        self.X = __prepare_data()

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def train_epoch(model, input_array, label_array, criterion_decoder, criterion_classifier, optimizer, batch_size, w1, device):
    
    model.train()

    runningLoss = 0 

    trainsize = int(input_array.shape[0]) 

    corrects = 0       
    
    for i in range(0, trainsize, batch_size):

        if i+batch_size<=trainsize:
            inputs = input_array[i:i+batch_size]
            labels = label_array[i:i+batch_size]

        else:
            inputs = input_array[i:]
            labels = label_array[i:]

        inputs, labels = Variable(torch.from_numpy(inputs).float().to(device)), Variable(torch.from_numpy(labels).long().to(device))
        # Initialize gradients to zero
        optimizer.zero_grad()                       
        # Feed-forward
        output_decoder, output_classifier = model(inputs)
        # Compute loss/error
        loss = (1-w1)*criterion_decoder(output_decoder, inputs)+w1*criterion_classifier(output_classifier, labels)
        # Accumulate loss per batch
        runningLoss += loss.item()
        # Backpropagate loss and compute gradients
        loss.backward()
        # Update the network parameters
        optimizer.step()

        class_scores = F.softmax(output_classifier, dim=1)
        _, predicted = torch.max(class_scores.data, 1)
        corrects += ((predicted==labels).sum(0)).float()
                
    return(model, runningLoss/float(trainsize/batch_size), corrects/trainsize)


def val_epoch(model, input_array, label_array, criterion_decoder, criterion_classifier, batch_size, device):

    model.eval()

    valsize = int(input_array.shape[0])

    corrects = 0.0

    runningLoss = 0.0

    with torch.no_grad():      

        for i in range(0, valsize, batch_size):

            if i+batch_size<=valsize:
                inputs = input_array[i:i+batch_size]
                labels = label_array[i:i+batch_size]

            else:
                inputs = input_array[i:]
                labels = label_array[i:]

            inputs, labels = torch.from_numpy(inputs).float().to(device), torch.from_numpy(labels).long().to(device)
            # Feed-forward
            output_decoder, output_classifier = model(inputs)
            
            loss = criterion_decoder(output_decoder, inputs)+criterion_classifier(output_classifier, labels)
            # Accumulate loss per batch
            runningLoss += loss.item()

            class_scores = F.softmax(output_classifier, dim=1)
            _, predicted = torch.max(class_scores.data, 1)
            corrects += ((predicted==labels).sum(0)).float()

    return (runningLoss/float(valsize/batch_size), corrects/valsize)

def test_model(model, input_array, label_array, batch_size, device):

    model.eval()

    testsize = int(input_array.shape[0])

    corrects = 0.0

    with torch.no_grad():      

        for i in range(0, testsize, batch_size):

            if i+batch_size<=testsize:
                inputs = input_array[i:i+batch_size]
                labels = label_array[i:i+batch_size]

            else:
                inputs = input_array[i:]
                labels = label_array[i:]

            inputs, labels = torch.from_numpy(inputs).float().to(device), torch.from_numpy(labels).long().to(device)                      
            # Feed-forward
            output_decoder, output_classifier = model(inputs)
            
            class_scores = F.softmax(output_classifier, dim=1)
            _, predicted = torch.max(class_scores.data, 1)
            corrects += ((predicted==labels).sum(0)).float()

    return corrects/testsize

