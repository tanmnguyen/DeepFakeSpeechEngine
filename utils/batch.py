import sys 
sys.path.append('../')

import torch 
import configs 
import numpy as np

from utils.io import read_melspectrogram_from_batch

def process_mel_spectrogram(mel_spec):
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0

    return log_spec 

def get_melspectrogram(batch, process = True, max_length=None):
    # read melspectrogram features with shape (features, time)
    melspectrogram_features = read_melspectrogram_from_batch(batch, max_length)

    # process melspectrogram features 
    if process:
        melspectrogram_features = [process_mel_spectrogram(torch.from_numpy(features)) for features in melspectrogram_features]
    else:
        melspectrogram_features = [torch.from_numpy(features) for features in melspectrogram_features]
        
    # determine the maximum length of melspectrogram features in the batch
    if max_length is None:
        max_length = max(features.shape[1] for features in melspectrogram_features)

    # pad melspectrogram features with zeros
    padded_melspectrogram_features = []
    for features in melspectrogram_features:
        # Calculate the amount of padding needed
        padding_length = max_length - features.shape[1]
        
        # Pad with zeros using torch
        padded_features = torch.nn.functional.pad(features, 
                                                  pad=(0, padding_length),
                                                  mode='constant',
                                                  value=0)
        
        padded_melspectrogram_features.append(padded_features)
    
    # Stack padded melspectrogram features into a single tensor
    padded_melspectrogram_features = torch.stack(padded_melspectrogram_features)

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
    # read mfcc features 
    melspectrogram_features = get_melspectrogram(batch, max_length=3000)
    # read and format text for training 
    tokens, labels = get_text(batch)

    return melspectrogram_features, tokens, labels 

def speaker_recognition_collate_fn(batch):
    # read mfcc features 
    melspectrogram_features = get_melspectrogram(batch, process=False, max_length=3000)
    # read and format text for training 
    speaker_labels = torch.tensor([item['speaker_id'] for item in batch])

    return melspectrogram_features, speaker_labels