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

def train_gen_net(model, train_dataloader, accuracy, log_file, train_spk=True):
    epoch_loss, epoch_wer, epoch_ser, epoch_spk_acc, \
        epoch_mel_mse, epoch_loss_spk, epoch_loss_asr, epoch_disc_acc = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    for i, (melspectrogram_features, tokens, labels, speaker_labels) in enumerate(tqdm(train_dataloader)):
        melspectrogram_features, tokens, labels, speaker_labels = \
            melspectrogram_features.to(configs.device), \
            tokens.to(configs.device), \
            labels.to(configs.device), \
            speaker_labels.to(configs.device)

        loss, spk_output, asr_output, mel_mse, loss_spk, loss_asr, d_acc = model.train_generator(
            melspectrogram_features, tokens, labels, speaker_labels
        )

        if train_spk:
            loss_spk, _ = model.train_speaker_recognizer(melspectrogram_features, speaker_labels)

        with torch.no_grad():
            wer, ser = compute_error_rate(asr_output, labels)
            acc = accuracy(spk_output, speaker_labels)

            epoch_mel_mse += mel_mse.item()
            epoch_loss += loss.item()
            epoch_wer += wer
            epoch_ser += ser
            epoch_spk_acc += acc
            epoch_loss_spk += loss_spk.item()
            epoch_loss_asr += loss_asr.item()
            epoch_disc_acc += d_acc

        torch.cuda.empty_cache()
        if i % 1 == 0:
            log(f"Loss: {epoch_loss / (i+1):.4f} " + \
                f"| WER: {epoch_wer / (i+1):.4f} " + \
                f"| SER: {epoch_ser / (i+1):.4f} " + \
                f"| Speaker Accuracy: {epoch_spk_acc / (i+1):.4f} " + \
                f"| Mel MSE: {epoch_mel_mse / (i+1):.4f}" + \
                f"| SPK Loss: {epoch_loss_spk / (i+1):.4f}" + \
                f"| ASR Loss: {epoch_loss_asr / (i+1):.4}" + \
                f"| Disc Avg Acc: {epoch_disc_acc / (i+1):.4}" + \
                f"| LR: {model.gen_optimizer.param_groups[0]['lr']:.4f}", log_file)

    return {
        'loss': epoch_loss / len(train_dataloader),
        'wer': epoch_wer / len(train_dataloader),
        'ser': epoch_ser / len(train_dataloader),
        'mel_mse': epoch_mel_mse / len(train_dataloader), 
        'speaker_accuracy': epoch_spk_acc / len(train_dataloader)
    }

def valid_gen_net(model, valid_dataloader, accuracy, log_file):
    model.generator.eval()
    model.asr_model.eval()
    model.spk_model.eval()

    epoch_loss, epoch_wer, epoch_ser, epoch_spk_acc, \
        epoch_mel_mse, epoch_loss_spk, epoch_loss_asr = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 
    for i, (melspectrogram_features, tokens, labels, speaker_labels) in enumerate(tqdm(valid_dataloader)):
        melspectrogram_features, tokens, labels, speaker_labels = \
            melspectrogram_features.to(configs.device), \
            tokens.to(configs.device), \
            labels.to(configs.device), \
            speaker_labels.to(configs.device)

        with torch.no_grad():
            loss, spk_output, asr_output, mel_mse, loss_spk, loss_asr = model.valid_generator(
                melspectrogram_features, tokens, labels, speaker_labels
            )

            wer, ser = compute_error_rate(asr_output, labels)
            acc = accuracy(spk_output, speaker_labels)

            epoch_mel_mse += mel_mse.item()
            epoch_loss += loss.item()
            epoch_wer += wer
            epoch_ser += ser
            epoch_spk_acc += acc
            epoch_loss_spk += loss_spk.item()
            epoch_loss_asr += loss_asr.item()

        torch.cuda.empty_cache()

    log(f"Loss: {epoch_loss / (i+1):.4f} " + \
                f"| WER: {epoch_wer / (i+1):.4f} " + \
                f"| SER: {epoch_ser / (i+1):.4f} " + \
                f"| Speaker Accuracy: {epoch_spk_acc / (i+1):.4f} " + \
                f"| Mel MSE: {epoch_mel_mse / (i+1):.4f}" + \
                f"| SPK Loss: {epoch_loss_spk / (i+1):.4f}" + \
                f"| ASR Loss: {epoch_loss_asr / (i+1):.4}" + \
                f"| LR: {model.gen_optimizer.param_groups[0]['lr']:.4f}", log_file)

    return {
        'loss': epoch_loss / len(valid_dataloader),
        'wer': epoch_wer / len(valid_dataloader),
        'ser': epoch_ser / len(valid_dataloader),
        'mel_mse': epoch_mel_mse / len(valid_dataloader), 
        'speaker_accuracy': epoch_spk_acc / len(valid_dataloader)
    }
    