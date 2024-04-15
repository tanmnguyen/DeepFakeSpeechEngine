import sys 
sys.path.append("../")

import torch 
import torch.nn as nn 
from einops import rearrange
from utils.batch import process_mel_spectrogram


class Generator(nn.Module):
    def __init__(self, in_channels=80):
        super(Generator, self).__init__()

        self.fc = nn.Linear(in_channels, in_channels)
        self.relu = nn.ReLU() 

        self.melspec_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=in_channels, nhead=8), 
            num_layers=2
        )

        self.fc_out = nn.Linear(in_channels, in_channels)


    def forward(self, x):
        """ 
        x shape: (batch_size, input_channels, seq_len) 
        """

        mel_x0 = rearrange(x, 'b c t -> b t c') # rearrange to (batch_size, seq_len, input_channels)
        mel_x1 = self.fc(mel_x0)
        mel_x1 = self.relu(mel_x1) + mel_x0 

        mel_latent = self.melspec_encoder(mel_x1) # (batch_size, seq_len, in_channels)
        mel_latent = self.relu(mel_latent)  

        out = rearrange(mel_latent, 'b t c -> b c t') # rearrange to (batch_size, input_channels, seq_len)
        out = out + x 

        return out


class MelGenerator(nn.Module):
    def __init__(self, input_channels: int, asr_model: nn.Module, spk_model: nn.Module):
        super(MelGenerator, self).__init__()
        self.generator = Generator()
        self.asr_model = asr_model 
        self.spk_model = spk_model

        self.asr_model.eval()
        self.spk_model.eval()


    def forward(self, x):
        # generate the output
        output = self.generator(x)

        return output 

    def loss(self, x, tokens, labels, speaker_labels):
        gen_melspec = self.generator(x)
        processed_gen_melspec = process_mel_spectrogram(gen_melspec)

        loss_spk, spk_output = self.spk_model.neg_cross_entropy_loss(processed_gen_melspec, speaker_labels)
        loss_asr, asr_output = self.asr_model.loss(processed_gen_melspec, tokens, labels, encoder_no_grad=False)

        with torch.no_grad():
            mel_mse = nn.functional.mse_loss(
                gen_melspec.contiguous().view(x.shape[0], -1), 
                x.contiguous().view(x.shape[0], -1)
            )
        
        return loss_asr + loss_spk, spk_output, asr_output, mel_mse
