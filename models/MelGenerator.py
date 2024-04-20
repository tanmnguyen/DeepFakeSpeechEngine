import sys 
sys.path.append("../")

import torch 
import configs 
import torch.nn as nn 
import torch.optim as optim
from torchmetrics import Accuracy

from einops import rearrange
from .Discriminator import Discriminator
from utils.batch import process_mel_spectrogram

class Generator(nn.Module):
    def __init__(self, in_channels=80):
        super(Generator, self).__init__()
        
        # Encoder layers
        self.enc1 = nn.Sequential(
            nn.Conv1d(in_channels, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.enc2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.enc3 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Decoder layers
        self.dec1 = nn.Sequential(
            nn.ConvTranspose1d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose1d(128, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
        )


    def forward(self, x):
        # Encoder
        enc1_out = self.enc1(x)
        enc2_out = self.enc2(enc1_out)
        enc3_out = self.enc3(enc2_out)
        
        # Decoder with skip connections
        dec1_out = self.dec1(enc3_out)
        dec1_out = dec1_out + enc2_out  # Adding skip connection (using addition, could use concatenation instead)
        dec2_out = self.dec2(dec1_out)
        dec2_out = dec2_out + enc1_out  # Adding skip connection
        dec3_out = self.dec3(dec2_out)
        dec3_out = dec3_out + x  # Adding skip connection

        out = nn.ReLU()(dec3_out)

        return out


class MelGenerator(nn.Module):
    def __init__(self, asr_model: nn.Module, spk_model: nn.Module):
        super(MelGenerator, self).__init__()
        self.generator = Generator()
        self.asr_model = asr_model 
        self.spk_model = spk_model
        self.discriminator = Discriminator()
        self.dis_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.disc_accuracy_fn = Accuracy(task="binary")

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

    def train_generator(self, x, tokens, labels, speaker_labels, beta):
        self.generator.train() 
        self.asr_model.eval()
        self.spk_model.eval()

        self.gen_optimizer.zero_grad()
        self.zero_grad() 
    
        gen_melspec = self.generator(x)

        # train adversarial discriminator
        self.discriminator.zero_grad()
        self.dis_optimizer.zero_grad()
        loss_d_real, d_out_real = self.discriminator.loss(x, torch.ones(x.shape[0],).to(configs.device))
        loss_d_fake, d_out_fake = self.discriminator.loss(gen_melspec.detach(), torch.zeros(x.shape[0],).to(configs.device))
        d_acc = (self.disc_accuracy_fn(d_out_fake, torch.zeros(x.shape[0],).to(configs.device)) + \
                self.disc_accuracy_fn(d_out_real, torch.ones(x.shape[0],).to(configs.device))) / 2
        loss_d = loss_d_real + loss_d_fake
        loss_d.backward()
        self.dis_optimizer.step()
        
        # train content 
        processed_gen_melspec = process_mel_spectrogram(gen_melspec)
        loss_spk, spk_output = self.spk_model.loss(processed_gen_melspec, speaker_labels)
        loss_asr, asr_output = self.asr_model.loss(processed_gen_melspec, tokens, labels, encoder_no_grad=False)
        adv_gen_loss, _ = self.discriminator.loss(gen_melspec, torch.ones(x.shape[0],).to(configs.device))
        loss = (loss_asr / loss_spk) * beta + adv_gen_loss

        # update generator
        loss.backward()
        self.gen_optimizer.step()
        
        if self.gen_optimizer.param_groups[0]['lr'] >= configs.mel_generator_cfg['min_lr']:
            self.gen_scheduler.step()

        with torch.no_grad():
            mel_mse = nn.functional.mse_loss(
                gen_melspec.contiguous().view(x.shape[0], -1), 
                x.contiguous().view(x.shape[0], -1)
            )

        return loss, spk_output, asr_output, mel_mse, loss_spk, loss_asr, d_acc
    
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