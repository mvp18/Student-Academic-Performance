import torch
import torch.nn as nn

class Autoencoder(nn.Module):    
    def __init__(self, data_dim=24, drop_rate=0.1):
        super(Autoencoder, self).__init__()

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

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
               
        return decoded  
