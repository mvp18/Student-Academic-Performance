import torch
import numpy as np
import time
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder=nn.Linear(24,8)
        self.decoder=nn.Linear(8,24)
        self.l1=nn.Linear(8,64)
        self.l2=nn.Linear(64,3)
        self.soft=nn.Softmax()
    def forward(self, x):
        x1=self.encoder(x)
        x=self.decoder(x1)
        y=self.l1(x1)
        y=self.l2(y)
        y=self.soft(y)
        return x,y