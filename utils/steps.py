import sys 
sys.path.append('../')

import math
import torch 
import configs 

from tqdm import tqdm
from utils.io import log
from utils.metrics import compute_error_rate
from utils.batch import process_mel_spectrogram

tokenizer = configs.speech_recognition_cfg['tokenizer']

def assert_loss(loss_val):
    if math.isnan(loss_val) or math.isinf(loss_val):
        raise ValueError(f'Loss value is {loss_val}')

def train_asr_net(model, train_dataloader, scheduler, optimizer, log_file):

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


def valid_asr_net(model, valid_dataloader):
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


# train speaker recognition model
def train_spk_net(model, train_dataloader, accuracy, criterion, scheduler, optimizer, log_file):
    model.train()

    epoch_loss, epoch_acc = 0.0, 0.0
    for i, (features, labels) in enumerate(tqdm(train_dataloader)):
        features, labels = features.to(configs.device), labels.to(configs.device)

        optimizer.zero_grad()
        preds = model(features)

        loss = criterion(preds, labels)

        loss.backward()
        optimizer.step()

        if optimizer.param_groups[0]['lr'] >= configs.speaker_recognition_cfg['min_lr']:
            scheduler.step()

        with torch.no_grad():
            acc = accuracy(preds, labels)
            epoch_loss += loss.item()
            epoch_acc += acc

        torch.cuda.empty_cache()

        if i % 100 == 0:
            log(f"Loss: {epoch_loss / (i+1)} " + \
                f"| Accuracy: {epoch_acc / (i+1)} " + \
                f"| LR: {optimizer.param_groups[0]['lr']}", log_file)


    return {
        'loss': epoch_loss / len(train_dataloader),
        'accuracy': epoch_acc / len(train_dataloader)
    }

def valid_spk_net(model, valid_dataloader, accuracy, criterion):
    model.eval()

    epoch_loss, epoch_acc = 0.0, 0.0
    for i, (features, labels) in enumerate(tqdm(valid_dataloader)):
        with torch.no_grad():
            features, labels = features.to(configs.device), labels.to(configs.device)
            preds = model(features)
            loss = criterion(preds, labels)
            acc = accuracy(preds, labels)
            epoch_loss += loss.item()
            epoch_acc += acc

        torch.cuda.empty_cache()

    return {
        'loss': epoch_loss / len(valid_dataloader),
        'accuracy': epoch_acc / len(valid_dataloader)
    }

def train_gen_net(model, train_dataloader, scheduler, optimizer, accuracy, spk_model, asr_model, log_file, stage="mimic"):
    model.train() 

    epoch_loss, epoch_wer, epoch_ser, epoch_spk_acc = 0.0, 0.0, 0.0, 0.0
    for i, (melspectrogram_features, tokens, labels, speaker_labels) in enumerate(tqdm(train_dataloader)):

        # melspec = process_mel_spectrogram(melspectrogram_features)
 
        melspectrogram_features, tokens, labels, speaker_labels = \
            melspectrogram_features.to(configs.device), \
            tokens.to(configs.device), \
            labels.to(configs.device), \
            speaker_labels.to(configs.device)

        optimizer.zero_grad()
        spk_model.zero_grad()
        asr_model.zero_grad()

        gen_melspec = model(melspectrogram_features)

        if stage == "deepfake":
            gen_melspec = process_mel_spectrogram(gen_melspec)

            loss_spk, spk_output = spk_model.loss(gen_melspec, speaker_labels)
            loss_asr, asr_output = asr_model.loss(gen_melspec, tokens, labels)

            # we want to minimize the loss of the speech recognition model 
            # while maximizing the loss of the speaker recognition model
            loss = loss_asr / loss_spk

        elif stage == "mimic":
            # here the criterion should be mse 
            loss = torch.nn.MSELoss()(gen_melspec.view(tokens.shape[0], -1), melspectrogram_features.view(tokens.shape[0], -1))

        loss.backward()
        optimizer.step()

        if optimizer.param_groups[0]['lr'] >= configs.mel_generator_cfg['min_lr']:
            scheduler.step()

        with torch.no_grad():
            wer, ser = compute_error_rate(asr_output, labels)
            acc = accuracy(spk_output, speaker_labels)

            epoch_loss += loss.item()
            epoch_wer += wer
            epoch_ser += ser
            epoch_spk_acc += acc


        torch.cuda.empty_cache()
        if i % 100 == 0:
            log(f"Loss: {epoch_loss / (i+1)} " + \
                f"| WER: {epoch_wer / (i+1)} " + \
                f"| SER: {epoch_ser / (i+1)} " + \
                f"| Speaker Accuracy: {epoch_spk_acc / (i+1)} " + \
                f"| LR: {optimizer.param_groups[0]['lr']}", log_file)

    return {
        'loss': epoch_loss / len(train_dataloader),
        'wer': epoch_wer / len(train_dataloader),
        'ser': epoch_ser / len(train_dataloader),
        'speaker_accuracy': epoch_spk_acc / len(train_dataloader)
    }

def valid_gen_net():
    pass 