import sys 
sys.path.append('../')

import torch.nn as nn

from einops import rearrange
from utils.metrics import neg_cross_entropy_loss
from .ECAPA_TDNN.ecapa_tdnn import ECAPA_TDNN_GLOB_c512 

class Discriminator(nn.Module):
    def __init__(self, feat_dim=80, embed_dim=256, pooling_func='ASTP'):
        super(Discriminator, self).__init__()
        self.encoder = ECAPA_TDNN_GLOB_c512(
            feat_dim=feat_dim,
            embed_dim=embed_dim,
            pooling_func=pooling_func
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(embed_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, mel):
        x = rearrange(mel, 'b t c -> b c t')
        # compute embeddings
        x = self.encoder(x)
        # apply dropout
        x = self.dropout(x)
        # classification
        x = self.fc(x)
        x = self.sigmoid(x)

        return x
    
    def loss(self, mel, labels):
        # torch.ones_like(output) 
        output = self.forward(mel)
        return nn.BCELoss()(output, labels)
