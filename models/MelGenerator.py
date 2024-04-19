import sys 
sys.path.append("../")

import torch 
import configs 
import torchaudio 
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F

from einops import rearrange
from torchmetrics import Accuracy
from .Discriminator import Discriminator
from utils.batch import process_mel_spectrogram

class Generator(nn.Module):
    def __init__(self, asr_model, in_channels=80, hidden_channels=384, out_channels=80):
        super(Generator, self).__init__()

        # content encoder 
        self.asr_encoder = asr_model.whisper_model.encoder
        self.asr_encoder.require_grad = False

        self.conv_mel = nn.Conv1d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1)

        # mel spectrogram decoder 
        self.melspec_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=384, nhead=2, batch_first=True), 
            num_layers=2
        )

        self.conv_out = nn.Conv1d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=1)

    def get_trainable_parameters(self, default_lr):
        return [
            {'params': self.conv_mel.parameters(), 'lr': default_lr},
            {'params': self.melspec_decoder.parameters(), 'lr': default_lr},
            {'params': self.conv_out.parameters(), 'lr': default_lr}
        ]

    def forward(self, x):
        """
        x shape: (batch_size, input_channels, seq_len)
        """

        # content encoder
        with torch.no_grad():
            x_encoded = process_mel_spectrogram(x)
            x_encoded = self.asr_encoder(x_encoded)

        # conv mel
        x = self.conv_mel(x) 
        x = rearrange(x, 'b c t -> b t c')

        # decode using transformer 
        x_decoded = self.melspec_decoder(tgt=x, memory=x_encoded)
        x_decoded = rearrange(x_decoded, 'b t c -> b c t')

        # conv out 
        output = self.conv_out(x_decoded)

        return output, x_encoded
    
    def loss_content_encoder(self, mel):
        output, tru_encoded = self(mel)
        
        processed_output = process_mel_spectrogram(output)
        gen_encoded = self.asr_encoder(processed_output)

        return nn.functional.mse_loss(
            gen_encoded.contiguous().view(mel.shape[0], -1),
            tru_encoded.contiguous().view(mel.shape[0], -1),
        ), output, processed_output
        

class MelGenerator(nn.Module):
    def __init__(self, asr_model: nn.Module, spk_model: nn.Module):
        super(MelGenerator, self).__init__()
        self.asr_model = asr_model 
        self.spk_model = spk_model
        self.generator = Generator(asr_model = self.asr_model)

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
        output = self.generator(x)
        return output 

    def train_generator(self, x, tokens, labels, speaker_labels):
        self.generator.train()
        self.asr_model.eval()
        self.spk_model.eval()

        self.gen_optimizer.zero_grad()
        self.zero_grad()

        loss_content, out_mel, processed_gen_melspec = self.generator.loss_content_encoder(x)
        loss_content.backward()
        self.gen_optimizer.step()

        loss_spk, spk_output = self.spk_model.loss(processed_gen_melspec, speaker_labels)
        loss_asr, asr_output = self.asr_model.loss(processed_gen_melspec, tokens, labels, encoder_no_grad=False)

        d_acc = torch.tensor(0)
        loss = loss_content 
        mel_mse = torch.tensor(0)
        return loss, spk_output, asr_output, mel_mse, loss_spk, loss_asr, d_acc
    
        # self.generator.train() 
        # self.asr_model.eval()
        # self.spk_model.eval()

        # self.gen_optimizer.zero_grad()
        # self.zero_grad() 
    
        # gen_melspec = self.generator(x)

        # # train adversarial discriminator
        # self.discriminator.zero_grad()
        # self.dis_optimizer.zero_grad()
        # loss_d_real, d_out_real = self.discriminator.loss(x, torch.ones(x.shape[0],).to(configs.device))
        # loss_d_fake, d_out_fake = self.discriminator.loss(gen_melspec.detach(), torch.zeros(x.shape[0],).to(configs.device))
        # d_acc = (self.disc_accuracy_fn(d_out_fake, torch.zeros(x.shape[0],).to(configs.device)) + \
        #         self.disc_accuracy_fn(d_out_real, torch.ones(x.shape[0],).to(configs.device))) / 2
        # loss_d = loss_d_real + loss_d_fake
        # loss_d.backward()
        # self.dis_optimizer.step()
        
        # # train content 
        # processed_gen_melspec = process_mel_spectrogram(gen_melspec)
        # processed_tru_melspec = process_mel_spectrogram(x)
        # loss_spk, spk_output = self.spk_model.loss(processed_gen_melspec, speaker_labels)
        # # loss_asr, asr_output = self.asr_model.loss(processed_gen_melspec, tokens, labels, encoder_no_grad=False)
        # loss_asr, asr_output = self.asr_model.loss_encoder(processed_tru_melspec, processed_gen_melspec, tokens)
        # adv_gen_loss, _ = self.discriminator.loss(gen_melspec, torch.ones(x.shape[0],).to(configs.device))
        # loss = loss_asr / loss_spk + adv_gen_loss

        # # update generator
        # loss.backward()
        # self.gen_optimizer.step()
        
        # if self.gen_optimizer.param_groups[0]['lr'] >= configs.mel_generator_cfg['min_lr']:
        #     self.gen_scheduler.step()

        # with torch.no_grad():
        #     mel_mse = nn.functional.mse_loss(
        #         gen_melspec.contiguous().view(x.shape[0], -1), 
        #         x.contiguous().view(x.shape[0], -1)
        #     )

        # return loss, spk_output, asr_output, mel_mse, loss_spk, loss_asr, d_acc
    
    def train_speaker_recognizer(self, x, speaker_labels):
        self.spk_model.train() 
        self.spk_model.zero_grad()
        self.spk_optimizer.zero_grad()

        gen_melspec, _ = self.generator(x)
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