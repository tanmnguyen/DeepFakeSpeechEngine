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
    # update the speaker recognition configuration
    configs.speaker_recognition_cfg['train_option']['spk2idx'] = configs.get_json(
        f"json/tedlium_{args.set}_spks.json",
        start_index=args.start_spk_idx,
        num_keys=50,
        shuffle=False
    )
    configs.speaker_recognition_cfg['speaker_ids'] = os.path.join(args.set, "utt2spk")
    log(configs.speaker_recognition_cfg, log_file)

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

    asr_model = load_asr(configs.mel_generator_cfg['asr_weight'])
    spk_model = load_spk(configs.mel_generator_cfg['spk_weight'], num_classes=dataset.num_classes)
    
    gen_model = MelGenerator(
        asr_model=asr_model, 
        spk_model=spk_model,
        step_size = len(train_dataloader) * 2,
    ).to(configs.device)

    if args.resume is not None:
        gen_model.load_state_dict(torch.load(args.resume))

    log(gen_model, log_file)
    log(f"Device: {configs.device}", log_file)
    log(f"Train set size: {len(train_dataset)}", log_file)
    log(f"Valid set size: {len(valid_dataset)}", log_file)
    log(f"Number of parameters: {sum(p.numel() for p in gen_model.parameters())}", log_file)

    accuracy = Accuracy(task="multiclass", num_classes=dataset.num_classes).to(configs.device)

    # initialize the network optimal solution first by learning the identity function
    for epoch in range(configs.mel_generator_cfg['epochs']):
        train_history = train_gen_net(
            gen_model, 
            train_dataloader, 
            accuracy, 
            log_file, 
            train_spk=1 if epoch < 1 else 0.005, 
            beta=(0.2, 0.2, 20) if epoch < 1 else (7, 5, 1),
        )
        log(
            f"[Train] Epoch: {epoch+1}/{configs.mel_generator_cfg['epochs']} - " +
            f"Loss: {train_history['loss']} | " +
            f"WER: {train_history['wer']} | " +
            f"SER: {train_history['ser']} | " +
            f"Speaker Accuracy: {train_history['speaker_accuracy']} | " +
            f"Mel MSE: {train_history['mel_mse']} | " + 
            f"SPK Loss: {train_history['spk_loss']} | " + 
            f"ASR Loss: {train_history['asr_loss']} | " +
            f"Disc Avg acc: {train_history['disc_avg_acc']}",
            log_file
        )

        valid_history = valid_gen_net(gen_model, valid_dataloader, accuracy, log_file, beta=(1,1,1))
        log(
            f"[Valid] Epoch: {epoch+1}/{configs.mel_generator_cfg['epochs']} - " +
            f"Loss: {valid_history['loss']} | " +
            f"WER: {valid_history['wer']} | " +
            f"SER: {valid_history['ser']} | " +
            f"Speaker Accuracy: {valid_history['speaker_accuracy']} | " +
            f"Mel MSE: {valid_history['mel_mse']} | " + 
            f"SPK Loss: {valid_history['spk_loss']} | " + 
            f"ASR Loss: {valid_history['asr_loss']} | " +
            f"Disc Avg acc: {valid_history['disc_avg_acc']}",
            log_file
        )

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
    
    parser.add_argument('-start_spk_idx',
                        '--start_spk_idx',
                        type=int,
                        default=0,
                        required=False,
                        help="Start speaker index")
    
    parser.add_argument('-resume',
                        '--resume',
                        type=str,
                        required=False,
                        help="Path to a weight file to resume training from")


    args = parser.parse_args()
    main(args)