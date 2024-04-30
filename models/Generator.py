import torch.nn as nn 
from einops import rearrange

class MelShiftNetwork(nn.Module):
    def __init__(self, in_channels=80):
        super(MelShiftNetwork, self).__init__()

        self.fc = nn.Linear(in_channels, in_channels)
        self.relu = nn.ReLU() 

        self.melspec_encoder_1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=in_channels, nhead=2, batch_first=True), 
            num_layers=2
        )
        self.fc1 = nn.Linear(in_channels, in_channels)

        self.melspec_encoder_2 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=in_channels, nhead=2, batch_first=True), 
            num_layers=2
        )
        self.fc2 = nn.Linear(in_channels, in_channels)

        self.melspec_encoder_3 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=in_channels, nhead=2, batch_first=True), 
            num_layers=2
        )
        self.fc3 = nn.Linear(in_channels, in_channels)
        self.fc_out = nn.Linear(in_channels, in_channels)

    def forward(self, x):
        """ 
        x shape: (batch_size, input_channels, seq_len) 
        """
        y = self.fc(x)
        x = self.relu(y) + x 

        y = self.melspec_encoder_1(x) # (batch_size, seq_len, in_channels)
        y = self.fc1(x)
        x = self.relu(y + x)

        y = self.melspec_encoder_2(x) # (batch_size, seq_len, in_channels)
        y = self.fc2(x)
        x = self.relu(y + x) 

        y = self.melspec_encoder_3(x) # (batch_size, seq_len, in_channels)
        y = self.fc3(x)
        x = self.relu(y + x)

        x = self.fc_out(x)  

        return x

class Generator(nn.Module):
    def __init__(self, in_channels=80):
        super(Generator, self).__init__()

        self.mel_shift_net_up = MelShiftNetwork(in_channels=in_channels)
        self.mel_shift_net_dn = MelShiftNetwork(in_channels=in_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        """ 
        x shape: (batch_size, input_channels, seq_len) 
        """
        # rearrange to (batch_size, seq_len, input_channels)
        x = rearrange(x, 'b c t -> b t c') 
        x0 = x

        shift_up = self.mel_shift_net_up(x)
        shift_dn = self.mel_shift_net_dn(x)

        x = self.relu(shift_up) + x0 - self.relu(shift_dn)

        x = rearrange(x, 'b t c -> b c t') # rearrange to (batch_size, input_channels, seq_len)
        return x