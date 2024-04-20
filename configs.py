import json 
import torch 
import whisper 
import numpy as np

from whisper.normalizers import EnglishTextNormalizer

# speech recognition training configurations 
decode_option = whisper.DecodingOptions(language="en", without_timestamps=True)
speech_recognition_cfg = {
    'batch_size': 16,
    'epochs': 30,
    'learning_rate': 1e-5,
    'no_decay': ["bias", "LayerNorm.weight"],
    'min_lr': 1e-7,
    'scheduler_gamma': 0.9,
    'normalizer': EnglishTextNormalizer(),
    'decodeOption': decode_option,
    "tokenizer": whisper.tokenizer.get_tokenizer(
        False, 
        language="en", 
        num_languages=99,
        task=decode_option.task
    ), 
    "result_dir": "results/asr"
}

def get_json(json_path, num_keys, shuffle=False):
    with open(json_path, 'r') as f:
        data = json.load(f)
    # get the first num_keys keys
    keys = list(data.keys())
    # shuffle keys 
    if shuffle:
        keys = np.random.permutation(keys)
    # build new json data 
    keys = keys[:num_keys]
    new_data = {key: data[key] for key in keys}

    return new_data

speaker_recognition_cfg = {
    'batch_size': 16,
    'epochs': 30,
    'learning_rate': 1e-3,
    'min_lr': 1e-7,
    'scheduler_gamma': 0.9,
    "result_dir": "results/spk",
    "speaker_ids": "train/utt2spk",
    "train_option": {
        "option": "spk2spk",  # first train option speaker to speaker
        "spk2idx": get_json("json/tedlium_train_spks.json", 50, shuffle=False)
    }, 
}


mel_generator_cfg = {
    'batch_size': 4,
    'epochs': 50,
    'learning_rate': 2e-5,
    'min_lr': 7e-6,
    'scheduler_gamma': 0.7,
    "result_dir": "results/gen",
    # "asr_weight": "weights/asr/asr_model.pt",
    "asr_weight": "weights/asr/tiny_whisper_model.pth",
    "spk_weight": "weights/spk/spk_model.pt"
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'