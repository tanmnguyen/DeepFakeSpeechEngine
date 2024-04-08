import torch 

# speech recognition training configurations 
speech_recognition_cfg = {
    'batch_size': 8,
    'epochs': 30,
    'learning_rate': 1e-3,
    'no_decay': ["bias", "LayerNorm.weight"],
    'min_lr': 1e-5,
}
result_dir = 'results'
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
# device = 'cpu'