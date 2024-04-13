# This file generate a new spectrogram from the original spectrogram 
import os 
import time 
import torch
import configs 
import argparse
import torch.optim as optim

from utils.io import log
from torchmetrics import Accuracy
from torch.utils.data import DataLoader
from utils.networks import load_asr, load_spk
from utils.steps import train_gen_net, valid_gen_net
from utils.batch import spectrogram_generation_collate_fn
from datasets.SpectrogramGenerationDataset import SpectrogramGenerationDataset

from models.MelGenerator import MelGenerator

# create a directory to save the results
str_time = time.strftime("%m-%d-%Y-%H-%M-%S")
result_dir = os.path.join(configs.mel_generator_cfg["result_dir"], f'{str_time}')
os.makedirs(result_dir, exist_ok=True)

# define log file 
log_file = os.path.join(result_dir, 'log.txt')

torch.manual_seed(3001)
def main(args):
    dataset = SpectrogramGenerationDataset(
        os.path.join(args.data, args.set), 
        configs.speaker_recognition_cfg['train_option']
    )

    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=configs.mel_generator_cfg['batch_size'],
        collate_fn=spectrogram_generation_collate_fn, 
        shuffle=True
    )

    valid_dataloader = DataLoader(
        valid_dataset, 
        batch_size=configs.mel_generator_cfg['batch_size'],
        collate_fn=spectrogram_generation_collate_fn, 
        shuffle=False
    )
    
    gen_model = MelGenerator(input_channels=80).to(configs.device)
    asr_model = load_asr(configs.mel_generator_cfg['asr_weight'])
    spk_model = load_spk(configs.mel_generator_cfg['spk_weight'], num_classes=dataset.num_classes)

    # freeze the asr and spk model
    # asr_model.requires_grad_(False)
    # spk_model.requires_grad_(False)


    log(gen_model, log_file)
    log(f"Device: {configs.device}", log_file)
    log(f"Train set size: {len(train_dataset)}", log_file)
    log(f"Valid set size: {len(valid_dataset)}", log_file)
    log(f"Number of parameters: {sum(p.numel() for p in gen_model.parameters())}", log_file)

    optimizer = optim.Adam(gen_model.parameters(), 
        lr=configs.mel_generator_cfg['learning_rate'], 
        eps=1e-8
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=len(train_dataloader) * 2, 
        gamma=configs.mel_generator_cfg['scheduler_gamma']
    )

    accuracy = Accuracy(task="multiclass", num_classes=dataset.num_classes).to(configs.device)

    for epoch in range(configs.speaker_recognition_cfg['epochs']):
        train_history = train_gen_net(gen_model, train_dataloader, scheduler, optimizer, accuracy, spk_model, asr_model, log_file)
        log(
            f"[Train] Epoch: {epoch+1}/{configs.mel_generator_cfg['epochs']} - " +
            f"Loss: {train_history['loss']} | " +
            f"WER: {train_history['wer']} | " +
            f"SER: {train_history['ser']} | " +
            f"Speaker Accuracy: {train_history['speaker_accuracy']}",
            log_file
        )

        # valid_history = valid_gen_net(model, valid_dataloader, accuracy, criterion)
        # log(
        #     f"[Valid] Epoch: {epoch+1}/{configs.mel_generator_cfg['epochs']} - " + 
        #     f"Loss: {valid_history['loss']} | " + 
        #     f"Accuracy: {valid_history['accuracy']}",
        #     log_file
        # )

        torch.save(gen_model.state_dict(), os.path.join(result_dir, f"gen_model_epoch_{epoch+1}.pt"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-data',
                        '--data',
                        required=True,
                        help="path to kaldi data format directory")
    
    parser.add_argument('-set',
                        '--set',
                        type=str,
                        default="train",
                        required=False,
                        help="[train/test/dev] set")


    args = parser.parse_args()
    main(args)