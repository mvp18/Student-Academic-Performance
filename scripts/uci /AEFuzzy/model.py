import torch
import torch.nn as nn

class AEFuzzy(nn.Module):    
    def __init__(self, data_dim=35, drop_rate=0.0):
        super(AEFuzzy, self).__init__()
        
        self.data_dim=data_dim
        self.drop_rate=drop_rate

        self.encoder = nn.Sequential(
            
            nn.Linear(self.data_dim, 32),
            nn.BatchNorm1d(32),
            nn.PReLU(),
            nn.Dropout(self.drop_rate),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.PReLU()
            )
          
        self.decoder = nn.Sequential(
            
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.PReLU(),
            nn.Dropout(self.drop_rate),
            nn.Linear(32, self.data_dim)
            )

        self.regressor = nn.Sequential(

            nn.Linear(16, 128),
            nn.BatchNorm1d(128),
            nn.PReLU(),
            nn.Dropout(self.drop_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.PReLU(),
            nn.Linear(64,1)
            )
                
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        regressor_out = self.regressor(encoded)
               
        return decoded, regressor_out
