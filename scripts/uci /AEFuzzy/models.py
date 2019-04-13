import torch
import torch.nn as nn

# from torchsummary import summary

class AEFuzzy(nn.Module):    
    def __init__(self):
        super(AEFuzzy, self).__init__()
          
        self.encoder = nn.Sequential(
            
            nn.Linear(35, 32),
            nn.BatchNorm1d(32),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.PReLU(),
            
        )
          
        self.decoder = nn.Sequential(
            
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.PReLU(),
            # nn.Dropout(0.1),
            nn.Linear(32, 35),
        
        )

        self.regressor = nn.Sequential(

            nn.Dropout(0.15),
            nn.Linear(16, 128),
            nn.BatchNorm1d(128),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.PReLU(),
            # nn.Dropout(0.1),
            # nn.Linear(64, 32),
            # nn.BatchNorm1d(32),
            # nn.PReLU(),
            nn.Linear(64,1)

        	)
        
        for m in self.regressor.children():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        for m in self.encoder.children():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        for m in self.decoder.children():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        regressor_out = self.regressor(encoded)
               
        return decoded, regressor_out #  
