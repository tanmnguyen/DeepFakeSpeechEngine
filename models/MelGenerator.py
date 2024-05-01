# UNI: tmn2134
import sys 
sys.path.append("../")

import torch 
import configs 
import torch.nn as nn 
import torch.optim as optim
from torchmetrics import Accuracy

from .Generator import Generator
from .Discriminator import Discriminator
from utils.batch import process_mel_spectrogram

class MelGenerator(nn.Module):
    def __init__(self, asr_model: nn.Module, spk_model: nn.Module, step_size: int, dis_hidden_dim: int = 16):
        super(MelGenerator, self).__init__()
        self.generator = Generator()
        self.asr_model = asr_model 
        self.spk_model = spk_model
        self.discriminator = Discriminator(hidden_dim=dis_hidden_dim)

        self.set_training_config(step_size) 

    def set_training_config(self, step_size):
        # discriminator --------------------------------
        self.dis_optimizer = optim.Adam(
            self.discriminator.parameters(), 
            lr=configs.mel_generator_cfg['learning_rate'], 
            betas=(0.5, 0.999)
        )
        self.dis_scheduler = torch.optim.lr_scheduler.StepLR(
            self.dis_optimizer, 
            step_size=step_size, 
            gamma=configs.mel_generator_cfg['scheduler_gamma']
        )
        self.disc_accuracy_fn = Accuracy(task="binary")

        # generator --------------------------------
        self.gen_optimizer = optim.Adam(self.generator.parameters(), 
            lr=configs.mel_generator_cfg['learning_rate'], 
            eps=1e-8
        )

        self.gen_scheduler = torch.optim.lr_scheduler.StepLR(
            self.gen_optimizer, 
            step_size=step_size, 
            gamma=configs.mel_generator_cfg['scheduler_gamma']
        )

        # speaker recognition --------------------------------
        self.spk_optimizer = optim.Adam(self.spk_model.parameters(), 
            lr=configs.mel_generator_cfg['learning_rate'], 
            eps=1e-8
        )

        self.spk_scheduler = torch.optim.lr_scheduler.StepLR(
            self.spk_optimizer, 
            step_size=step_size, 
            gamma=configs.mel_generator_cfg['scheduler_gamma']
        )

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
        # loss_spk, spk_output = self.spk_model.loss(processed_gen_melspec, speaker_labels)
        loss_spk, spk_output = self.spk_model.neg_cross_entropy_loss(processed_gen_melspec, speaker_labels)
        loss_asr, asr_output = self.asr_model.loss(processed_gen_melspec, tokens, labels, encoder_no_grad=False)
        adv_gen_loss, _ = self.discriminator.loss(gen_melspec, torch.ones(x.shape[0],).to(configs.device))
        loss = loss_asr * beta[0] + loss_spk * beta[1] + adv_gen_loss * beta[2]

        # update generator
        loss.backward()
        self.gen_optimizer.step()
        
        if self.gen_optimizer.param_groups[0]['lr'] >= configs.mel_generator_cfg['min_lr']:
            self.gen_scheduler.step()
            self.dis_scheduler.step()

        with torch.no_grad():
            mel_mse = nn.functional.mse_loss(
                gen_melspec.contiguous().view(x.shape[0], -1), 
                x.contiguous().view(x.shape[0], -1)
            )

        return loss, spk_output, asr_output, mel_mse, loss_spk, loss_asr, d_acc
    
    def valid_generator(self, x, tokens, labels, speaker_labels, beta):
        self.generator.eval() 
        self.asr_model.eval()
        self.spk_model.eval()

        with torch.no_grad():
            gen_melspec = self.generator(x)

            # train adversarial discriminator
            _, d_out_real = self.discriminator.loss(x, torch.ones(x.shape[0],).to(configs.device))
            _, d_out_fake = self.discriminator.loss(gen_melspec.detach(), torch.zeros(x.shape[0],).to(configs.device))
            d_acc = (self.disc_accuracy_fn(d_out_fake, torch.zeros(x.shape[0],).to(configs.device)) + \
                    self.disc_accuracy_fn(d_out_real, torch.ones(x.shape[0],).to(configs.device))) / 2
            
            # train content 
            processed_gen_melspec = process_mel_spectrogram(gen_melspec)
            loss_spk, spk_output = self.spk_model.neg_cross_entropy_loss(processed_gen_melspec, speaker_labels)
            loss_asr, asr_output = self.asr_model.loss(processed_gen_melspec, tokens, labels, encoder_no_grad=False)
            adv_gen_loss, _ = self.discriminator.loss(gen_melspec, torch.ones(x.shape[0],).to(configs.device))
            loss = loss_asr * beta[0] + loss_spk * beta[1] + adv_gen_loss * beta[2]

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
        if self.spk_optimizer.param_groups[0]['lr'] >= configs.mel_generator_cfg['min_lr']:
            self.spk_scheduler.step()

        return loss_spk, spk_output
