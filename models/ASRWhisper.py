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

    def forward(self, mel, tokens):
        # get features from the encoder
        with torch.no_grad():
            features = self.whisper_model.encoder(mel)

        out = self.whisper_model.decoder(tokens, features)
        return out 
        # return self.whisper_model(mel, tokens)
    
    def loss(self, mel, tokens, labels):
        out = self(mel, tokens)
        return self.loss_fn(out.view(-1, out.size(-1)), labels.contiguous().view(-1)), out