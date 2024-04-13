import torch.nn as nn 
from einops import rearrange

class MelGenerator(nn.Module):
    def __init__(self, input_channels: int):
        super(MelGenerator, self).__init__()

        self.mhsa1 = nn.MultiheadAttention(embed_dim=input_channels, num_heads=1)
        self.mhsa2 = nn.MultiheadAttention(embed_dim=input_channels, num_heads=1)
        self.fc = nn.Linear(input_channels, input_channels)

        # init weight such that this network is an identity function
        # init weight of mhsa1 and mhsa2 to 0
        self.mhsa1.in_proj_weight.data.zero_()
        self.mhsa1.in_proj_bias.data.zero_()
        self.mhsa2.in_proj_weight.data.zero_()
        self.mhsa2.in_proj_bias.data.zero_()

        # make weight of fc to identity matrix
        self.fc.weight.data.zero_()
        self.fc.bias.data.zero_()
        for i in range(input_channels):
            self.fc.weight.data[i, i] = 1.0


    def forward(self, x):
        # x shape: (batch_size, input_channels, seq_len)

        x = rearrange(x, 'b c t -> b t c') # rearrange to (batch_size, seq_len, input_channels)

        x_mhsa, _ = self.mhsa1(x, x, x)
        x = x + x_mhsa

        x_mhsa, _ = self.mhsa2(x, x, x)
        x = x + x_mhsa

        x = self.fc(x)

        x = rearrange(x, 'b t c -> b c t') # rearrange to (batch_size, input_channels, seq_len)

        return x
    
import torch 
model = MelGenerator(input_channels=80)
print(model)
melspec = torch.rand(2, 80, 3000)
output = model(melspec)
print(output.shape)

# compare the melspec and output values they should all be equals 
print(melspec[0, 0, 0], output[0, 0, 0])
