import sys 
sys.path.append("../")

import torch 
import torch.nn as nn 
from einops import rearrange
from utils.batch import process_mel_spectrogram


class Generator(nn.Module):
    def __init__(self, asr_encoder: nn.Module, in_channels=80, em_channels=384):
        super(Generator, self).__init__()

        self.fc = nn.Linear(in_channels, em_channels)

        self.melspec_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=em_channels, nhead=8), 
            num_layers=2
        )

        self.asr_encoder = asr_encoder 

        self.cross_attention = nn.MultiheadAttention(embed_dim=em_channels, num_heads=8)

        self.fc_out = nn.Linear(em_channels, in_channels)


    def forward(self, x):
        """ 
        x shape: (batch_size, input_channels, seq_len) 
        """

        mel_x = rearrange(x, 'b c t -> b t c') # rearrange to (batch_size, seq_len, input_channels)
        mel_x = self.fc(mel_x)
        asr_x = x 

        melspec_latent = self.melspec_encoder(mel_x)    # (batch_size, seq_len0, in_channels)
        asr_latent = self.asr_encoder(asr_x)            # (batch_size, seq_len1, in_channels)

        # Cross-attention
        # Query: melspec_latent, Key and Value: asr_latent
        melspec_latent = melspec_latent.permute(1, 0, 2)  # (seq_len0, batch_size, in_channels)
        asr_latent = asr_latent.permute(1, 0, 2)  # (seq_len1, batch_size, in_channels)
        attended_output, _ = self.cross_attention(melspec_latent, asr_latent, asr_latent)

        # Rearrange output and transpose back to original shape
        attended_output = attended_output.permute(1, 0, 2)  # (batch_size, seq_len0, in_channels)
        attended_output = self.fc_out(attended_output)

        attended_output = rearrange(attended_output, 'b t c -> b c t') # rearrange to (batch_size, input_channels, seq_len)
        return attended_output


class MelGenerator(nn.Module):
    def __init__(self, input_channels: int, asr_model: nn.Module, spk_model: nn.Module):
        super(MelGenerator, self).__init__()
        self.generator = Generator(asr_model.whisper_model.encoder)
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
        
        # tru_melspec = process_mel_spectrogram(x)
        gen_melspec = process_mel_spectrogram(gen_melspec)

        loss_spk, spk_output = self.spk_model.neg_cross_entropy_loss(gen_melspec, speaker_labels)
        # loss_asr, asr_output = self.asr_model.loss_encoder(tru_melspec, gen_melspec, tokens)
        loss_asr, asr_output = self.asr_model.loss(gen_melspec, tokens, labels, encoder_no_grad=False)

        return loss_asr, spk_output, asr_output
        # return loss_asr + loss_spk, spk_output, asr_output

    
# import torch 
# model = MelGenerator(input_channels=80)
# print(model)
# melspec = torch.rand(2, 80, 3000)
# output = model(melspec)
# print(output.shape)

# # compare the melspec and output values they should all be equals 
# print(melspec[0, 0, 0], output[0, 0, 0])
