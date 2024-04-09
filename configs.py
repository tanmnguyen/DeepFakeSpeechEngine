import torch 
import whisper 

from whisper.normalizers import EnglishTextNormalizer

# speech recognition training configurations 
decode_option = whisper.DecodingOptions(language="en", without_timestamps=True)
speech_recognition_cfg = {
    'batch_size': 16,
    'epochs': 30,
    'learning_rate': 1e-3,
    'no_decay': ["bias", "LayerNorm.weight"],
    'min_lr': 1e-5,
    'scheduler_gamma': 0.9,
    'normalizer': EnglishTextNormalizer(),
    'decodeOption': decode_option,
    "tokenizer": whisper.tokenizer.get_tokenizer(
        False, 
        language="en", 
        num_languages=99,
        task=decode_option.task
    ), 
}

result_dir = 'results'
device = 'cuda' if torch.cuda.is_available() else 'cpu'