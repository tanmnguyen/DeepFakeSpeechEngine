import sys 
sys.path.append('../')

import math
import torch 
import configs 

from tqdm import tqdm
from utils.io import log
from utils.metrics import compute_error_rate


def assert_loss(loss_val):
    if math.isnan(loss_val) or math.isinf(loss_val):
        raise ValueError(f'Loss value is {loss_val}')

def train_net(model, train_dataloader, optimizer, scheduler, log_file):
    model.train()

    epoch_loss, epoch_wer, epoch_ser = 0.0, 0.0, 0.0
    for i, (mfcc, text) in enumerate(tqdm(train_dataloader)):
        mfcc, text = mfcc.to(configs.device), text.to(configs.device)

        optimizer.zero_grad()
        loss, output = model.loss(mfcc, text)

        loss_val = loss.item()
        assert_loss(loss_val)

        loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            wer, ser = compute_error_rate(output, text)
            epoch_loss += loss_val
            epoch_wer += wer
            epoch_ser += ser

        torch.cuda.empty_cache()

        if i % 500 == 0:
            log(f"Loss: {epoch_loss / (i+1)} | WER: {epoch_wer / (i+1)} | SER: {epoch_ser / (i+1)}", log_file)

    return {
        'loss': epoch_loss / len(train_dataloader),
        'wer': epoch_wer / len(train_dataloader),
        'ser': epoch_ser / len(train_dataloader)
    }


def valid_net(model, valid_dataloader):
    model.eval()
    
    epoch_loss, epoch_wer, epoch_ser = 0.0, 0.0, 0.0
    for mfcc, text in tqdm(valid_dataloader):
        mfcc, text = mfcc.to(configs.device), text.to(configs.device)

        with torch.no_grad():
            loss, output = model.loss(mfcc, text)

            loss_val = loss.item()
            assert_loss(loss_val)

            wer, ser = compute_error_rate(output, text)
            epoch_loss += loss_val
            epoch_wer += wer
            epoch_ser += ser

        torch.cuda.empty_cache()

    return {
        'loss': epoch_loss / len(valid_dataloader),
        'wer': epoch_wer / len(valid_dataloader),
        'ser': epoch_ser / len(valid_dataloader)
    }