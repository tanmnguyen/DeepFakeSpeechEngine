import torch 
from utils.io import read_mat_hdf5, read_melspectrogram_from_batch
from WordTokenizer import word_tokenizer

def get_melspectrogram(batch, max_length=None):
    # read melspectrogram features with shape (features, time)
    melspectrogram_features = read_melspectrogram_from_batch(batch)

    # determine the maximum length of melspectrogram features in the batch
    if max_length is None:
        max_length = max(features.shape[1] for features in melspectrogram_features)

    # pad melspectrogram features with zeros
    padded_melspectrogram_features = []
    for features in melspectrogram_features:
        # Calculate the amount of padding needed
        padding_length = max_length - features.shape[1]
        
        # Pad with zeros using torch
        padded_features = torch.nn.functional.pad(torch.tensor(features), 
                                                  pad=(0, padding_length),
                                                  mode='constant',
                                                  value=0)
        
        padded_melspectrogram_features.append(padded_features)
    
    # Stack padded melspectrogram features into a single tensor
    padded_melspectrogram_features = torch.stack(padded_melspectrogram_features)

    # trim to max length 
    padded_melspectrogram_features = padded_melspectrogram_features[:, :, :max_length]

    # the output now is (batch, feature, time)
    return padded_melspectrogram_features

def get_text(batch, word_tokenizer, max_length=None):
    # Read text transcriptions and tokenize
    text = [word_tokenizer.tokenize(word_tokenizer.SOS + " " + item['text'] + " " + word_tokenizer.EOS) for item in batch]

    # Determine maximum length of text sequences if not provided
    if max_length is None:
        max_length = max(len(tokens) for tokens in text)
    
    # Convert tokens to PyTorch tensors and pad
    padded_text = []
    for tokens in text:
        # Convert tokens to tensor
        token_tensor = torch.tensor(tokens)
        
        # Pad tokens with zeros to match max_length
        pad_length = max_length - len(tokens)
        padded_token_tensor = torch.nn.functional.pad(
            token_tensor, (0, pad_length), 
            mode='constant', 
            value=word_tokenizer.word2idx[word_tokenizer.PAD]
        )
        
        # Append to padded_text
        padded_text.append(padded_token_tensor)

    # Stack padded text into a single tensor
    padded_text = torch.stack(padded_text)
    
    return padded_text

def speech_recognition_collate_fn(batch):
    # read mfcc features 
    melspectrogram_features = get_melspectrogram(batch, max_length=3000)
    # read and format text for training 
    text = get_text(batch, word_tokenizer)
    # set tokens 
    tokens = text[:, :-1]
    # set labels 
    labels = text[:, 1:]

    return melspectrogram_features, tokens, labels 

    