import sys 
sys.path.append('../')

import torch.nn as nn

from einops import rearrange

class Discriminator(nn.Module):
    def __init__(self, feat_dim=80):
        super(Discriminator, self).__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=feat_dim, nhead=2, batch_first=True), 
            num_layers=2
        )

        self.fc = nn.Linear(feat_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, mel):
        x = rearrange(mel, 'b c t -> b t c') 
        
        # compute embeddings
        x = self.encoder(x)

        # compute the average along the feature dimension
        x = x.mean(dim=1)
        x = self.fc(x)
        x = self.sigmoid(x)

        return x
    
    def loss(self, mel, labels):
        # torch.ones_like(output) 
        output = self.forward(mel).view(-1)
        return nn.BCELoss()(output, labels), output
