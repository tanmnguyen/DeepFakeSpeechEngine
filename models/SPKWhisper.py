# UNI: tmn2134
import torch
import torch.nn as nn 

from .whisper.model import Whisper
from utils.networks import load_state_dict

class SPKWhisper(nn.Module):
    def __init__(self, dims, num_classes: int, whisper_model_weight: str):
        super(SPKWhisper, self).__init__()
        # load the pre-trained whisper model 
        whisper_model = load_state_dict(Whisper(dims), whisper_model_weight)
        # define encoder 
        self.encoder = whisper_model.encoder

        self.flatten = nn.Flatten()

        # define the classifier 
        self.classifier = nn.Linear(1500 * 384, num_classes)

    def forward(self, mel):
        # feature extraction
        with torch.no_grad():
            x = self.encoder(mel)

        # print(x.shape)

        # flatten the output
        x = self.flatten(x)

        # classification
        x = self.classifier(x)
        
        return x
