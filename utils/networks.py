import sys 
sys.path.append("../")

import torch 
import configs 

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

def load_asr(weight_path):
    """
    Load the ASR model.

    Parameters:
        weight_path (str): The path to the saved ASR model state dictionary.

    Returns:
        torch.nn.Module: The loaded ASR model.
    """
    from models.ASRWhisper import ASRWhisper
    from models.whisper.model import ModelDimensions

    dims = ModelDimensions(
        n_mels=80, 
        n_audio_ctx=1500, 
        n_audio_state=384, 
        n_audio_head=6, 
        n_audio_layer=4, 
        n_vocab=51864, 
        n_text_ctx=448, 
        n_text_state=384, 
        n_text_head=6, 
        n_text_layer=4
    )

    model = ASRWhisper(
        dims, 
        pad_token=configs.speech_recognition_cfg['tokenizer'].eot,
        whisper_model_weight = "weights/asr/tiny_whisper_model.pth"
    ).to(configs.device)
    model = load_state_dict(model, weight_path)
    return model

def load_spk(weight_path, num_classes):
    """
    Load the speaker recognition model.

    Parameters:
        weight_path (str): The path to the saved speaker recognition model state dictionary.

    Returns:
        torch.nn.Module: The loaded speaker recognition model.
    """
    from models.SPKTDNN import SPKTDNN

    model = SPKTDNN(num_classes=num_classes).to(configs.device)
    model = load_state_dict(model, weight_path)
    return model