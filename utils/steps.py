import sys 
sys.path.append('../')

import math
import torch 
import configs 

from tqdm import tqdm
from utils.io import log
from utils.metrics import compute_error_rate

tokenizer = configs.speech_recognition_cfg['tokenizer']

def assert_loss(loss_val):
    if math.isnan(loss_val) or math.isinf(loss_val):
        raise ValueError(f'Loss value is {loss_val}')

def train_net(model, train_dataloader, scheduler, optimizer, log_file):

    model.train()

    epoch_loss, epoch_wer, epoch_ser = 0.0, 0.0, 0.0
    for i, (features, tokens, labels) in enumerate(tqdm(train_dataloader)):
        features, tokens, labels = features.to(configs.device), tokens.to(configs.device), labels.to(configs.device)

        optimizer.zero_grad()
        loss, output = model.loss(features, tokens, labels)

        loss_val = loss.item()
        assert_loss(loss_val)

        loss.backward()
        optimizer.step()

        if optimizer.param_groups[0]['lr'] >= configs.speech_recognition_cfg['min_lr']:
            scheduler.step()

        with torch.no_grad():
            wer, ser = compute_error_rate(output, labels)
            epoch_loss += loss_val
            epoch_wer += wer
            epoch_ser += ser

        torch.cuda.empty_cache()

        if i % 100 == 0:
            log(f"Loss: {epoch_loss / (i+1)} " + \
                f"| WER: {epoch_wer / (i+1)} " + \
                f"| SER: {epoch_ser / (i+1)} " + \
                f"| LR: {optimizer.param_groups[0]['lr']}", log_file)

    return {
        'loss': epoch_loss / len(train_dataloader),
        'wer': epoch_wer / len(train_dataloader),
        'ser': epoch_ser / len(train_dataloader)
    }


def valid_net(model, valid_dataloader):
    model.eval()
    
    epoch_loss, epoch_wer, epoch_ser = 0.0, 0.0, 0.0
    for features, tokens, labels in tqdm(valid_dataloader):
        features, tokens, labels = features.to(configs.device), tokens.to(configs.device), labels.to(configs.device)

        with torch.no_grad():
            loss, output = model.loss(features, tokens, labels)

            loss_val = loss.item()
            assert_loss(loss_val)

            wer, ser = compute_error_rate(output, labels)
            epoch_loss += loss_val
            epoch_wer += wer
            epoch_ser += ser

        torch.cuda.empty_cache()

    return {
        'loss': epoch_loss / len(valid_dataloader),
        'wer': epoch_wer / len(valid_dataloader),
        'ser': epoch_ser / len(valid_dataloader)
    }