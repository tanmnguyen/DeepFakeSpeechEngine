import torch 

# speech recognition training configurations 
speech_recognition_cfg = {
    'batch_size': 32,
    'epochs': 30,
    'learning_rate': 1e-3,
}

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
# device = 'cpu'