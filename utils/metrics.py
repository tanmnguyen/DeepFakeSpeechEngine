# UNI: tmn2134
import sys 
sys.path.append('../')

import torch
import configs 
import torch.nn.functional as F

def compute_error_rate(output, labels):
    # get tokenizer
    tokenizer = configs.speech_recognition_cfg['tokenizer']

    # get normalizer 
    normalizer = configs.speech_recognition_cfg['normalizer']

    # greedy decoding
    output = output.argmax(dim=-1)

    wer, ser, tot_words = 0.0, 0.0, 0.0
    for i in range(output.shape[0]): # batch dim 
        cnt_incorrects = 0
        for j in range(labels.shape[1]): # seq len dim 
            # finish processing if EOS token is found
            if labels[i,j] == tokenizer.eot:
                break
            
            tot_words += 1
            pred_word = normalizer(tokenizer.decode([output[i, j].item()]))
            true_word = normalizer(tokenizer.decode([labels[i, j].item()]))
            if pred_word != true_word:
                cnt_incorrects += 1

        # update WER and SER
        wer += cnt_incorrects
        ser += 1 if cnt_incorrects > 0 else 0

    # normalize WER and SER
    wer /= tot_words
    ser /= output.shape[0]

    return wer, ser



def neg_cross_entropy_loss(output, labels):
    output = F.softmax(output, dim=-1)
    epsilon = 1e-7  # Small value to avoid log(0)
    output = torch.clamp(output, epsilon, 1 - epsilon)  # Clip probabilities to avoid log(0) or log(1)
    neg_log_likelihood = -torch.log(1 - output[torch.arange(output.size(0)), labels])
    return neg_log_likelihood.mean()


# output = torch.tensor([[0.1, 0.1, 0.8]])
# labels = torch.tensor([2])

# print(output)
# print(labels)

# loss = neg_cross_entropy_loss(output, labels)
# print(loss)