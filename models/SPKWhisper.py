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

        # define the classifier 
        self.classifier = nn.Linear(dims.n_audio_state, num_classes)

    def forward(self, mel):
        # feature extraction
        x = self.encoder(mel)

        # compute the average across time 
        x = x.mean(dim=1)

        # classification
        x = self.classifier(x)
        
        return x
