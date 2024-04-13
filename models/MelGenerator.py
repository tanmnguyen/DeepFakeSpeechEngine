import torch.nn as nn 
from einops import rearrange

class MelGenerator(nn.Module):
    def __init__(self, input_channels: int):
        super(MelGenerator, self).__init__()
        
        # self.encoder = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(d_model=input_channels, nhead=8, batch_first=True), # encoder layer 
        #     num_layers=2,
        # )
        self.encoder = nn.Linear(input_channels, input_channels)

        # init weights to 1 and bias to 0 
        nn.init.ones_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        
        
    def forward(self, x):
        # x shape: (batch_size, input_channels, seq_len)

        x = rearrange(x, 'b c t -> b t c') # rearrange to (batch_size, seq_len, input_channels)
        x = self.encoder(x)
        x = rearrange(x, 'b t c -> b c t') # rearrange to (batch_size, input_channels, seq_len)

        return x