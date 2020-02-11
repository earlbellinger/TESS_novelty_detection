import torch 

import torch.nn as nn

class VanillaAE(nn.Module):
    def __init__(self, data_dim, dropout=0.1):
        
        super().__init__()
        
        self.fc1 = nn.Linear(data_dim, 64) 
        self.fc2 = nn.Linear(64, 32) 
        self.fc3 = nn.Linear(32, 64) 
        self.fc4 = nn.Linear(64, data_dim)
        
        self.relu = nn.LeakyReLU() 

        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_):
        
        x1 = self.relu(self.fc1(input_))
        #x1 = self.dropout(x1) 

        x2 = self.relu(self.fc2(x1))
        #x2 = self.dropout(x2) 

        x3 = self.relu(self.fc3(x2)) 
        #x3 = self.dropout(x3) 

        x4 = self.fc4(x3) 

        return x4 


