import torch
import torch.nn as nn

from torchsummary import summary

class AEFuzzy(nn.Module):    
    def __init__(self):
        super(AEFuzzy, self).__init__()
          
        self.encoder = nn.Sequential(
            
            nn.Linear(24, 16),
            nn.BatchNorm1d(16),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.PReLU(),
            
        )
          
        self.decoder = nn.Sequential(
            
            nn.Linear(8, 16),
            nn.BatchNorm1d(16),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 24),
        
        )

        self.classifier = nn.Sequential(

        	nn.Linear(8,128),
        	nn.BatchNorm1d(128),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Linear(128,64),
            nn.BatchNorm1d(64),
            nn.PReLU(),
            nn.Linear(64,3),
            nn.Softmax()

        	)
        
        for m in self.classifier.children():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        classifier_out = self.classifier(encoded)
               
        return decoded, classifier_out  
