import sys 
sys.path.append('../')

import torch 
import whisper 
import torch.nn as nn

from whisper.model import Whisper

def load_state_dict(model, weight_path):
    """
    Load a model's state dictionary while handling size mismatches.

    Parameters:
        model (torch.nn.Module): The model to load the state dictionary into.
        weight_path (str): The path to the saved state dictionary.
    """
    # Load the state dictionary while ignoring size mismatches
    loaded_state_dict = torch.load(weight_path, map_location=torch.device('cpu'))

    # Get the current model state dictionary
    current_state_dict = model.state_dict()

    # Update the current state dictionary with parameters from the loaded state dictionary,
    # while ignoring any size mismatches
    for name, param in loaded_state_dict.items():
        if name in current_state_dict:
            if param.size() != current_state_dict[name].size():
                print(f"Ignoring size mismatch for parameter '{name}'")
                continue
            current_state_dict[name].copy_(param)

    # Load the updated state dictionary into the model
    model.load_state_dict(current_state_dict)

    return model 

# Whisper model wrapper for ASR
class ASRWhisper(nn.Module):
    def __init__(self, dims, pad_token, whisper_model_weight: str): 
        super(ASRWhisper, self).__init__()
        # define the core whisper model
        self.whisper = Whisper(dims)
        # load the pre-trained whisper model
        self.whisper = load_state_dict(self.whisper, whisper_model_weight)
        # define loss function 
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token)

    def forward(self, mel, tokens):
        return self.whisper(mel, tokens)
    
    def loss(self, mel, tokens, labels):
        out = self(mel, tokens)
        return self.loss_fn(out.view(-1, out.size(-1)), labels.contiguous().view(-1)), out