import sys 
sys.path.append('../')

import torch 
from WordTokenizer import word_tokenizer

def greedy_decode(output):
    output = output.argmax(dim=-1) # (sequence, batch)
    output = output.permute(1, 0)  # (batch, sequence)
    for i in range(output.shape[0]):
        dec = torch.unique_consecutive(output[i])
        dec = dec[dec != word_tokenizer.word2idx[word_tokenizer.PAD]]
        output[i, :len(dec)] = dec
        output[i, len(dec):] = word_tokenizer.word2idx[word_tokenizer.PAD]
    return output 

def compute_error_rate(output, labels):
    # compute WER and SER for autogressive model 
    preds = torch.argmax(output, dim=-1)

    wer, ser, tot_words = 0.0, 0.0, 0.0
    for i in range(preds.shape[0]):
        cnt_incorrects = 0
        for j in range(labels.shape[1]):
            # finish processing if EOS token is found
            if labels[i,j] == word_tokenizer.word2idx[word_tokenizer.EOS]:
                break
            
            tot_words += 1
            if preds[i, j] != labels[i, j]:
                cnt_incorrects += 1

        # update WER and SER
        wer += cnt_incorrects
        ser += 1 if cnt_incorrects > 0 else 0

    # normalize WER and SER
    wer /= tot_words
    ser /= preds.shape[0]

    return wer, ser