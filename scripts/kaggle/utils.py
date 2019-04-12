import skfuzzy as fuzz
import time
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable, Function

class Create_Dataset:
    def __init__(self, data_size, num_fuzz_var):
        
        self.data_size = data_size
        self.num_fuzz_var = num_fuzz_var

    def fuzzy_mf(self, rng):
    
        self.mf = {}
        self.mf['vlo'] = fuzz.trimf(rng, [0, 0, 25])
        self.mf['lo'] = fuzz.trimf(rng, [0, 25, 50])
        self.mf['md'] = fuzz.trimf(rng, [25, 50, 75])
        self.mf['hi'] = fuzz.trimf(rng, [50, 75, 100])
        self.mf['vhi'] = fuzz.trimf(rng, [75, 100, 100])


    def read_data(self):
        
        dataset = pd.read_csv('./xAPI-Edu-Data.csv')

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
        self.X2 = np.concatenate((data_np_array[:,0:1],data_np_array[:,5:8]),axis=1)
        self.Y = data_np_array[:,data_np_array.shape[1]-1:data_np_array.shape[1]]


    def prepare_data(self, rng):
        
        x = np.zeros((self.data_size,self.num_fuzz_var*5))

        for i in range(self.data_size):
            for j in range(self.num_fuzz_var):
                x[i,j*5]=fuzz.interp_membership(rng, self.mf['vlo'], self.X1[i,j])
                x[i,j*5+1]=fuzz.interp_membership(rng, self.mf['lo'], self.X1[i,j])
                x[i,j*5+2]=fuzz.interp_membership(rng, self.mf['md'], self.X1[i,j])
                x[i,j*5+3]=fuzz.interp_membership(rng, self.mf['hi'], self.X1[i,j])
                x[i,j*5+4]=fuzz.interp_membership(rng, self.mf['vhi'], self.X1[i,j])

        self.X=np.concatenate((x,self.X2),axis=1)


def train(model, input_tensor, label_tensor, criterion_decoder, criterion_classifier, optimizer, batch_size, w1, w2, device):
    
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

        inputs, labels = Variable(inputs.float().to(device)), Variable(labels.long().to(device))                       

        # Feed-forward
        output_decoder, output_classifier = model(inputs)
        # Compute loss/error
        loss = w1*criterion_decoder(output_decoder, inputs)+w2*criterion_classifier(output_classifier, labels)
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


def testNet(model, input_tensor, label_tensor, batchsize, device):

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

            inputs, labels = Variable(inputs.float().to(device)), Variable(labels.long().to(device))                      

            # Feed-forward
            output_decoder, output_classifier = model(inputs)

            _, predicted = torch.max(output_classifier.data, 1)

            corrects += ((predicted==labels).sum(0)).float()

    return corrects/float(testsize)

