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
from datasets.SpeakerRecognitionDataset import SpeakerRecognitionDataset

from models.SPKConv import SPKConv
from models.SPKWhisper import SPKWhisper
from models.whisper.model import ModelDimensions

from utils.io import log
from utils.steps import train_spk_net, valid_spk_net
from utils.batch import speaker_recognition_collate_fn


from models.SPKTDNN import SPKTDNN

# create a directory to save the results
str_time = time.strftime("%m-%d-%Y-%H-%M-%S")
result_dir = os.path.join(configs.speaker_recognition_cfg["result_dir"], f'{str_time}')
os.makedirs(result_dir, exist_ok=True)

# define log file 
log_file = os.path.join(result_dir, 'log.txt')

torch.manual_seed(3001)
def main(args):
    configs.speaker_recognition_cfg['train_option']['spk2idx'] = configs.get_json(
        f"json/tedlium_{args.set}_spks.json",
        start_index=args.start_spk_idx,
        num_keys=50,
        shuffle=False
    )
    configs.speaker_recognition_cfg['speaker_ids'] = os.path.join(args.set, "utt2spk")
    log(configs.speaker_recognition_cfg, log_file)

    dataset = SpeakerRecognitionDataset(os.path.join(args.data, args.set), configs.speaker_recognition_cfg['train_option'])
    
    if len(dataset) <= 500:
        train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [0.5, 0.5])
    else:
        train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])

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

    model = SPKTDNN(num_classes=dataset.num_classes).to(configs.device)

    log(model, log_file)
    log(f"Device: {configs.device}", log_file)
    log(f"Train set size: {len(train_dataset)}", log_file)
    log(f"Valid set size: {len(valid_dataset)}", log_file)
    log(f"Number of parameters: {sum(p.numel() for p in model.parameters())}", log_file)

    optimizer = optim.Adam(model.parameters(), 
        lr=configs.speaker_recognition_cfg['learning_rate'], 
        eps=1e-8
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=len(train_dataloader) * 3, 
        gamma=configs.speaker_recognition_cfg['scheduler_gamma']
    )

    criterion = torch.nn.CrossEntropyLoss()
    accuracy = Accuracy(task="multiclass", num_classes=dataset.num_classes).to(configs.device)

    for epoch in range(configs.speaker_recognition_cfg['epochs']):
        train_history = train_spk_net(model, train_dataloader, accuracy, criterion, scheduler, optimizer, log_file)
        log(
            f"[Train] Epoch: {epoch+1}/{configs.speaker_recognition_cfg['epochs']} - " + 
            f"Loss: {train_history['loss']} | " + 
            f"Accuracy: {train_history['accuracy']}",
            log_file
        )

        valid_history = valid_spk_net(model, valid_dataloader, accuracy, criterion)
        log(
            f"[Valid] Epoch: {epoch+1}/{configs.speaker_recognition_cfg['epochs']} - " + 
            f"Loss: {valid_history['loss']} | " + 
            f"Accuracy: {valid_history['accuracy']}",
            log_file
        )

        torch.save(model.state_dict(), os.path.join(result_dir, f"spk_model_epoch_{epoch+1}.pt"))
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-data',
                        '--data',
                        required=True,
                        help="path to kaldi data format directory. This should contains dev, test, and train folders")
    

    parser.add_argument('-set',
                        '--set',
                        type=str,
                        default="train",
                        required=False,
                        help="[train/test/dev] set")
    
    parser.add_argument('-start_spk_idx',
                        '--start_spk_idx',
                        type=int,
                        default=0,
                        required=False,
                        help="Start speaker index")


    args = parser.parse_args()
    main(args)