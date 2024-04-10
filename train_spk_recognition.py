# This file train speaker recognition model 
# This program splits the train directory into train and dev set.

import os 
import time 
import torch
import configs 
import argparse 
import torch.optim as optim

from torchmetrics import Accuracy
from torch.utils.data import DataLoader
from models.SPKWhisper import SPKWhisper
from models.whisper.model import ModelDimensions
from datasets.SpeakerRecognitionDataset import SpeakerRecognitionDataset

from utils.io import log
from utils.steps import train_spk_net, valid_spk_net
from utils.batch import speaker_recognition_collate_fn

# create a directory to save the results
str_time = time.strftime("%m-%d-%Y-%H-%M-%S")
result_dir = os.path.join(configs.speaker_recognition_cfg["result_dir"], f'{str_time}')
os.makedirs(result_dir, exist_ok=True)

# define log file 
log_file = os.path.join(result_dir, 'log.txt')

torch.manual_seed(3001)
def main(args):

    dataset = SpeakerRecognitionDataset(os.path.join(args.data, "train"))
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=configs.speaker_recognition_cfg['batch_size'],
        collate_fn=speaker_recognition_collate_fn, 
        shuffle=True
    )

    valid_dataloader = DataLoader(
        valid_dataset, 
        batch_size=configs.speaker_recognition_cfg['batch_size'],
        collate_fn=speaker_recognition_collate_fn, 
        shuffle=False
    )

    dims = ModelDimensions(
        n_mels=80, 
        n_audio_ctx=1500, 
        n_audio_state=384, 
        n_audio_head=6, 
        n_audio_layer=4, 
        n_vocab=51864, 
        n_text_ctx=448, 
        n_text_state=384, 
        n_text_head=6, 
        n_text_layer=4
    )

    model = SPKWhisper(
        dims, 
        num_classes = dataset.num_classes,
        whisper_model_weight = "weights/asr/tiny_whisper_model.pth"
    ).to(configs.device)

    log(model, log_file)
    log(f"Device: {configs.device}", log_file)
    log(f"Number of parameters: {sum(p.numel() for p in model.parameters())}", log_file)

    optimizer = optim.AdamW(model.parameters(), 
        lr=configs.speaker_recognition_cfg['learning_rate'], 
        eps=1e-8
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=len(train_dataloader) * 5, 
        gamma=configs.speaker_recognition_cfg['scheduler_gamma']
    )

    criterion = torch.nn.CrossEntropyLoss()
    accuracy = Accuracy(task="multiclass", num_classes=dataset.num_classes)

    # train_spk_net(model, train_dataloader, accuracy, criterion, scheduler, optimizer, log_file)
    for epoch in range(configs.speaker_recognition_cfg['epochs']):
        train_history = train_spk_net(model, train_dataloader, accuracy, criterion, scheduler, optimizer, log_file)
        log(
            f"[Train] Epoch: {epoch+1}/{configs.speech_recognition_cfg['epochs']} - " + 
            f"Loss: {train_history['loss']} | " + 
            f"Accuracy: {train_history['accuracy']}",
            log_file
        )

        valid_history = valid_spk_net(model, valid_dataloader, accuracy, criterion)
        log(
            f"[Valid] Epoch: {epoch+1}/{configs.speech_recognition_cfg['epochs']} - " + 
            f"Loss: {valid_history['loss']} | " + 
            f"Accuracy: {valid_history['accuracy']}",
            log_file
        )
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-data',
                        '--data',
                        required=True,
                        help="path to kaldi data format directory. This should contains dev, test, and train folders")


    args = parser.parse_args()
    main(args)