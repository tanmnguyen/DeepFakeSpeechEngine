import sys 
sys.path.append('../')

import torch 
import configs 

from tqdm import tqdm
from utils.metrics import compute_error_rate


def assert_loss(loss):
    assert torch.isnan(loss).item() == False, "Loss is NaN!"
    assert torch.isinf(loss).item() == False, "Loss is Inf!"

def train_net(model, train_dataloader, optimizer, scheduler):
    model.train()

    epoch_loss, epoch_wer, epoch_ser = 0.0, 0.0, 0.0
    for mfcc, text in tqdm(train_dataloader):
        mfcc, text = mfcc.to(configs.device), text.to(configs.device)

        optimizer.zero_grad()
        loss, output = model.loss(mfcc, text)
        assert_loss(loss)

        loss.backward()
        optimizer.step()
        scheduler.step()

    # evaluating the model 
    train_metrics = valid_net(model, train_dataloader)
    return train_metrics

def valid_net(model, valid_dataloader):
    model.eval()
    
    epoch_loss, epoch_wer, epoch_ser = 0.0, 0.0, 0.0
    for mfcc, text in tqdm(valid_dataloader):
        mfcc, text = mfcc.to(configs.device), text.to(configs.device)

        with torch.no_grad():
            loss, output = model.loss(mfcc, text)
            assert_loss(loss)

            wer, ser = compute_error_rate(output, text)
            epoch_loss += loss.item()
            epoch_wer += wer
            epoch_ser += ser

    return {
        'loss': epoch_loss / len(valid_dataloader),
        'wer': epoch_wer / len(valid_dataloader),
        'ser': epoch_ser / len(valid_dataloader)
    }