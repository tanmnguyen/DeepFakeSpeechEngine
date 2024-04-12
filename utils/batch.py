import sys 
sys.path.append('../')

import torch 
import configs 
import librosa
import numpy as np

from utils.io import read_melspectrogram_from_batch

def process_mel_spectrogram(mel_spec):
    # Clamp to avoid taking the log of values close to zero
    mel_spec_batch = torch.clamp(mel_spec, min=1e-10)
    
    # Take the logarithm (base 10) of the spectrogram
    log_spec_batch = mel_spec_batch.log10()
    
    # Adjust the minimum value to avoid overly large values after log transformation
    log_spec_min = log_spec_batch.view(-1, 80, 3000).max(dim=2, keepdim=True)[0]
    log_spec_batch = torch.maximum(log_spec_batch, log_spec_min - 8.0)
    
    # Normalize the values to a range of [0, 1]
    log_spec_batch = (log_spec_batch + 4.0) / 4.0

    return log_spec_batch

def to_db(mel_spec):
    db = librosa.power_to_db(mel_spec)
    return torch.tensor(db)

def get_melspectrogram(batch, process_fn, max_length=None):
    # read melspectrogram features with shape (features, time)
    melspectrogram_features = read_melspectrogram_from_batch(batch, max_length)

    # # process melspectrogram features 
    # melspectrogram_features = [process_fn(torch.from_numpy(features)) for features in melspectrogram_features]

    # determine the maximum length of melspectrogram features in the batch
    if max_length is None:
        max_length = max(features.shape[1] for features in melspectrogram_features)

    # pad melspectrogram features with zeros
    padded_melspectrogram_features = []
    for features in melspectrogram_features:
        # Calculate the amount of padding needed
        padding_length = max_length - features.shape[1]
        
        # Pad with zeros using torch
        padded_features = torch.nn.functional.pad(torch.from_numpy(features), 
                                                  pad=(0, padding_length),
                                                  mode='constant',
                                                  value=0)
        
        padded_melspectrogram_features.append(padded_features)
    
    # Stack padded melspectrogram features into a single tensor
    padded_melspectrogram_features = torch.stack(padded_melspectrogram_features)

    # process melspectrogram features
    padded_melspectrogram_features = process_fn(padded_melspectrogram_features)

    # the output now is (batch, feature, time)
    return padded_melspectrogram_features

def capitalize(text):
    # capitalize the first letter for each word in the text
    return ' '.join([word.capitalize() for word in text.split()])


def get_text(batch):
    tokenizer = configs.speech_recognition_cfg['tokenizer']

    max_len = 0
    tokens, labels = [], []
    for item in batch:
        # define text 
        text =  [*tokenizer.sot_sequence_including_notimestamps] + \
                tokenizer.encode(item['text']) + \
                [tokenizer.eot]
        
        tokens.append(text[:-1])
        labels.append(text[1: ])

        # update max length
        max_len = max(max_len, len(tokens[-1]))

    # pad labels 
    labels = [np.pad(lab, (0, max_len - len(lab)), 'constant', constant_values=tokenizer.eot) for lab in labels]

    # pad tokens
    tokens = [np.pad(tok, (0, max_len - len(tok)), 'constant', constant_values=tokenizer.eot) for tok in tokens]

    # to numpy array
    tokens = np.array(tokens)
    labels = np.array(labels)

    # to tensor
    tokens = torch.tensor(tokens)
    labels = torch.tensor(labels)

    return tokens, labels 

def speech_recognition_collate_fn(batch):
    melspectrogram_features = get_melspectrogram(batch, process_fn=process_mel_spectrogram, max_length=3000)
    tokens, labels = get_text(batch)

    return melspectrogram_features, tokens, labels 

def speaker_recognition_collate_fn(batch):
    melspectrogram_features = get_melspectrogram(batch, process_fn=process_mel_spectrogram, max_length=3000)
    speaker_labels = torch.tensor([item['speaker_id'] for item in batch])

    return melspectrogram_features, speaker_labels

def spectrogram_generation_collate_fn(batch):
    melspectrogram_features = get_melspectrogram(batch, process_fn=lambda x : x, max_length=3000)
    _, tokens, labels = speech_recognition_collate_fn(batch)  
    _, speaker_labels = speaker_recognition_collate_fn(batch)

    return melspectrogram_features, tokens, labels, speaker_labels