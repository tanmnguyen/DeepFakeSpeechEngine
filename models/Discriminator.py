import sys 
sys.path.append('../')

import torch.nn as nn

from einops import rearrange

class Discriminator(nn.Module):
    def __init__(self, feat_dim=80, hidden_dim=128):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(feat_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, mel):
        x = rearrange(mel, 'b c t -> b t c') 
        
        # compute embeddings
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = x.mean(dim=1)

        # compute the average along the feature dimension
        x = self.fc3(x)
        x = self.sigmoid(x)

        return x
    
    def loss(self, mel, labels):
        # torch.ones_like(output) 
        output = self.forward(mel).view(-1)
        return nn.BCELoss()(output, labels), output
