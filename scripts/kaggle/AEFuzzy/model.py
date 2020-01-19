import torch
import torch.nn as nn

class AEFuzzy(nn.Module):    
    def __init__(self, data_dim=24, num_classes=3, drop_rate=0.1):
        super(AEFuzzy, self).__init__()

        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.data_dim = data_dim  
        
        self.encoder = nn.Sequential(
            
            nn.Linear(self.data_dim, 16),
            nn.BatchNorm1d(16),
            nn.PReLU(),
            nn.Dropout(self.drop_rate),
            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.PReLU(),
            
        )
          
        self.decoder = nn.Sequential(
            
            nn.Linear(8, 16),
            nn.BatchNorm1d(16),
            nn.PReLU(),
            nn.Dropout(self.drop_rate),
            nn.Linear(16, 24),
        
        )

        self.classifier = nn.Sequential(

        	nn.Linear(8,128),
        	nn.BatchNorm1d(128),
            nn.PReLU(),
            nn.Dropout(self.drop_rate),
            nn.Linear(128,64),
            nn.BatchNorm1d(64),
            nn.PReLU(),
            nn.Linear(64, self.num_classes),

        	)
        
        # if self.pretrained:
        #     for m in self.classifier.children():
        #         if isinstance(m, nn.Linear):
        #             nn.init.xavier_uniform_(m.weight)
        #             nn.init.zeros_(m.bias)
        #         elif isinstance(m, nn.BatchNorm1d):
        #             nn.init.constant_(m.weight, 1)
        #             nn.init.constant_(m.bias, 0)
                

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        classifier_out = self.classifier(encoded)
               
        return decoded, classifier_out  
