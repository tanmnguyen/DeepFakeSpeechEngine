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

# class Generator(nn.Module):
#     def __init__(self, in_channels=80):
#         super(Generator, self).__init__()

#         self.fc1 = nn.Linear(in_channels, in_channels)

#         self.melspec_encoder_1 = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(d_model=in_channels, nhead=2, batch_first=True), 
#             num_layers=2
#         )
#         self.fc2 = nn.Linear(in_channels, in_channels)

#         self.melspec_encoder_2 = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(d_model=in_channels, nhead=2, batch_first=True), 
#             num_layers=2
#         )
#         self.fc3 = nn.Linear(in_channels, in_channels)

#         self.melspec_encoder_3 = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(d_model=in_channels, nhead=2, batch_first=True), 
#             num_layers=2
#         )

#         self.fc_out = nn.Linear(in_channels, in_channels)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         """ 
#         x shape: (batch_size, input_channels, seq_len) 
#         """
#         x0 = x 

#         # rearrange to (batch_size, seq_len, input_channels)
#         x = rearrange(x, 'b c t -> b t c') 

#         y = self.fc1(x)
#         x = self.relu(y) + x 

#         y = self.melspec_encoder_1(x) # (batch_size, seq_len, in_channels)
#         x = self.fc2(x)
#         x = self.relu(y) + x

#         y = self.melspec_encoder_2(x) # (batch_size, seq_len, in_channels)
#         x = self.fc3(x)
#         x = self.relu(y) + x

#         y = self.melspec_encoder_3(x) # (batch_size, seq_len, in_channels)
#         x = self.fc_out(x)
#         x = self.relu(y) + x

#         x = rearrange(x, 'b t c -> b c t') # rearrange to (batch_size, input_channels, seq_len)
#         x = self.relu(x) + x0 

#         return x


class Generator(nn.Module):
    def __init__(self, asr_model):
        super(Generator, self).__init__()
        bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH
        self.processor = bundle.get_text_processor()
        self.tacotron2 = bundle.get_tacotron2().to(configs.device)
        self.asr_model = asr_model
        self.asr_model.require_grad = False

    def forward(self, mel):   

        # compute text from mel spectrogram
        with torch.no_grad():
            processed_mel = process_mel_spectrogram(mel)
            results = self.asr_model.decode(processed_mel)

            batch_text, batch_len = [], []
            for res in results:
                processed, lengths = self.processor(res.text)
                batch_text.append(processed.squeeze(0))
                batch_len.append(lengths)
                print(batch_text[-1].shape, batch_len[-1].shape)

            # max len 
            max_len = max([len(text) for text in batch_text])
            for i in range(len(batch_text)):
                batch_text[i] = torch.cat(
                    [batch_text[i].to(configs.device), torch.zeros(max_len - len(batch_text[i])).long().to(configs.device)]
                )

            # concat to batch dim 
            batch_text = torch.stack(batch_text, dim=0)
            batch_len = torch.cat(batch_len, dim=0).to(torch.int)

            # sort by descending order
            sorted_indices = torch.argsort(batch_len, descending=True)
            batch_text = batch_text[sorted_indices]
            batch_len = batch_len[sorted_indices]

        spec, _, _ = self.tacotron2.infer(batch_text.to(configs.device), batch_len.to(configs.device))
        
        # spec has shape (b, c, t)
        # pad or trim t dim to 3000
        spec_padded = F.pad(spec, (0, max(0, 3000 - spec.shape[-1])))
        spec_padded = spec_padded[:, :, :3000]

        return spec_padded
        

class MelGenerator(nn.Module):
    def __init__(self, asr_model: nn.Module, spk_model: nn.Module):
        super(MelGenerator, self).__init__()
        self.asr_model = asr_model 
        self.spk_model = spk_model
        self.generator = Generator(self.asr_model)
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

    def train_generator(self, x, tokens, labels, speaker_labels):
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
        loss = loss_asr / loss_spk + adv_gen_loss

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