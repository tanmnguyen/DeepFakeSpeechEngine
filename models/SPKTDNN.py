import sys 
sys.path.append('../')

import torch.nn as nn

from einops import rearrange
from utils.metrics import neg_cross_entropy_loss
from .ECAPA_TDNN.ecapa_tdnn import ECAPA_TDNN_GLOB_c512 

class SPKTDNN(nn.Module):
    def __init__(self, num_classes, feat_dim=80, embed_dim=256, pooling_func='ASTP'):
        super(SPKTDNN, self).__init__()
        # self.encoder = ECAPA_TDNN_GLOB_c512(
        #     feat_dim=feat_dim,
        #     embed_dim=embed_dim,
        #     pooling_func=pooling_func
        # )
        #self.rnn1 = nn.LSTM(feat_dim, embed_dim, 2, batch_first=True, bidirectional=True)
        #self.rnn2 = nn.LSTM(embed_dim * 2, embed_dim, 2, batch_first=True, bidirectional=True)
        # transformer encoder 
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=feat_dim,
                nhead=4,
                dim_feedforward=256,
                dropout=0.1
            ),
            num_layers=4g
        )

        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, mel):
        x = rearrange(mel, 'b t c -> b c t')
        # compute embeddings
        # x = self.encoder(x)
        # apply LSTM
        x = self.encoder(x)
        
        # average pooling
        x = x.mean(dim=1)

        # apply dropout
        x = self.dropout(x)
        # classification
        x = self.fc(x)

        return x
    
    def loss(self, mel, target):
        output = self.forward(mel)
        return nn.CrossEntropyLoss()(output, target), output

    def neg_cross_entropy_loss(self, mel, target):
        output = self.forward(mel)
        return neg_cross_entropy_loss(output, target), output