# UNI: tmn2134
import sys 
sys.path.append('../')

import torch 
import configs 
import torch.nn as nn

from .whisper.model import Whisper
from utils.networks import load_state_dict

# Whisper model wrapper for ASR
class ASRWhisper(nn.Module):
    def __init__(self, dims, pad_token, whisper_model_weight: str): 
        super(ASRWhisper, self).__init__()
        # define the core whisper model
        self.whisper_model= Whisper(dims)
        # load the pre-trained whisper model
        self.whisper_model = load_state_dict(self.whisper_model, whisper_model_weight)
        # define loss function 
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token)

    def decode(self, mel):
        return self.whisper_model.decode(mel, configs.speech_recognition_cfg['decodeOption'])

    def forward(self, mel, tokens, encoder_no_grad):
        # get features from the encoder
        if encoder_no_grad:
            with torch.no_grad():
                features = self.whisper_model.encoder(mel)
        else:
            features = self.whisper_model.encoder(mel)

        out = self.whisper_model.decoder(tokens, features)
        return out 
        # return self.whisper_model(mel, tokens)
    
    def loss(self, mel, tokens, labels, encoder_no_grad=True):
        out = self(mel, tokens, encoder_no_grad)
        return self.loss_fn(out.view(-1, out.size(-1)), labels.contiguous().view(-1)), out
    
    def loss_encoder(self, tru_mel, gen_mel, tokens):
        loss, ori_features, gen_features= self.whisper_model.get_encoder_loss(tru_mel, gen_mel)

        # get features from the encoder
        # ori_features = self.whisper_model.encoder(tru_mel)
        # ori_features, gen_features = self.whisper_model.encoder(gen_mel)

        # # compute output 
        with torch.no_grad():
            out = self.whisper_model.decoder(tokens, gen_features)
            
        return loss, out 
        # return nn.functional.mse_loss(
        #     ori_features.contiguous().view(tru_mel.shape[0], -1),
        #     gen_features.contiguous().view(tru_mel.shape[0], -1)
        # ), out
