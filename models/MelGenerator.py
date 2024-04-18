import sys 
sys.path.append("../")

import torch 
import configs 
import torch.nn as nn 
from einops import rearrange
from utils.batch import process_mel_spectrogram


class Generator(nn.Module):
    def __init__(self, in_channels=80):
        super(Generator, self).__init__()

        self.fc = nn.Linear(in_channels, in_channels)
        self.relu = nn.ReLU() 

        self.melspec_encoder_1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=in_channels, nhead=2), 
            num_layers=2
        )

        self.melspec_encoder_2 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=in_channels, nhead=2), 
            num_layers=2
        )

        self.melspec_encoder_3 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=in_channels, nhead=2), 
            num_layers=2
        )

        self.fc_out = nn.Linear(in_channels, in_channels)


    def forward(self, x):
        """ 
        x shape: (batch_size, input_channels, seq_len) 
        """
        x0 = x 

        # rearrange to (batch_size, seq_len, input_channels)
        x = rearrange(x, 'b c t -> b t c') 

        y = self.fc(x)
        x = self.relu(y) + x 

        y = self.melspec_encoder_1(x) # (batch_size, seq_len, in_channels)
        x = self.relu(y) + x

        y = self.melspec_encoder_2(x) # (batch_size, seq_len, in_channels)
        x = self.relu(y) + x

        y = self.melspec_encoder_3(x) # (batch_size, seq_len, in_channels)
        x = self.relu(y) + x

        x = rearrange(x, 'b t c -> b c t') # rearrange to (batch_size, input_channels, seq_len)
        x = self.relu(x) + x0 

        return x


class MelGenerator(nn.Module):
    def __init__(self, asr_model: nn.Module, spk_model: nn.Module):
        super(MelGenerator, self).__init__()
        self.generator = Generator()
        self.asr_model = asr_model 
        self.spk_model = spk_model

    def set_gen_optimizer(self, gen_optimizer, gen_scheduler):
        self.gen_optimizer = gen_optimizer
        self.gen_scheduler = gen_scheduler

    def set_spk_optimizer(self, spk_optimizer, spk_scheduler):
        self.spk_optimizer = spk_optimizer
        self.spk_scheduler = spk_scheduler

    def forward(self, x):
        # generate the output
        output = self.generator(x)

        return output 

    def train_generator(self, x, tokens, labels, speaker_labels):
        self.generator.train() 
        self.asr_model.eval()
        self.spk_model.eval()

        self.gen_optimizer.zero_grad()
        self.zero_grad() 
    
        gen_melspec = self.generator(x)
        magnitude_loss = nn.functional.mse_loss(
            torch.norm(gen_melspec, p=2, dim=(1,2)), 
            torch.norm(x, p=2, dim=(1,2))
        )

        processed_gen_melspec = process_mel_spectrogram(gen_melspec)
        loss_spk, spk_output = self.spk_model.loss(processed_gen_melspec, speaker_labels)
        # loss_spk, spk_output = self.spk_model.neg_cross_entropy_loss(processed_gen_melspec, speaker_labels)
        loss_asr, asr_output = self.asr_model.loss(processed_gen_melspec, tokens, labels, encoder_no_grad=False)

        with torch.no_grad():
            mel_mse = nn.functional.mse_loss(
                gen_melspec.contiguous().view(x.shape[0], -1), 
                x.contiguous().view(x.shape[0], -1)
            )
        
        loss = magnitude_loss + loss_asr / loss_spk
        
        loss.backward()
        self.gen_optimizer.step()

        # update scheduler 
        if self.gen_optimizer.param_groups[0]['lr'] >= configs.mel_generator_cfg['min_lr']:
            self.gen_scheduler.step()

        return loss, spk_output, asr_output, mel_mse, loss_spk, loss_asr
    
    def train_speaker_recognizer(self, x, speaker_labels):
        self.spk_model.train() 
        self.spk_model.zero_grad()
        self.spk_optimizer.zero_grad()

        gen_melspec = self.generator(x)
        processed_gen_melspec = process_mel_spectrogram(gen_melspec)

        loss_spk, spk_output = self.spk_model.loss(processed_gen_melspec, speaker_labels)

        loss_spk.backward()
        self.spk_optimizer.step()

        # update scheduler 
        if self.spk_optimizer.param_groups[0]['lr'] >= configs.speaker_recognition_cfg['min_lr']:
            self.spk_scheduler.step()

        return loss_spk, spk_output

    # def valid_generator(self, x, tokens, labels, speaker_labels):
    #     with torch.no_grad():
    #         gen_melspec = self.generator(x)
    #         processed_gen_melspec = process_mel_spectrogram(gen_melspec)

    #         loss_spk, spk_output = self.spk_model.neg_cross_entropy_loss(processed_gen_melspec, speaker_labels)
    #         loss_asr, asr_output = self.asr_model.loss(processed_gen_melspec, tokens, labels, encoder_no_grad=False)

    #         loss = loss_asr + loss_spk
    #         mel_mse = nn.functional.mse_loss(
    #             gen_melspec.contiguous().view(x.shape[0], -1), 
    #             x.contiguous().view(x.shape[0], -1)
    #         )

    #     return loss, spk_output, asr_output, mel_mse, loss_spk, loss_asr