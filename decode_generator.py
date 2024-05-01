# UNI: tmn2134
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

    # log configurations
    log(configs.speaker_recognition_cfg, log_file)
    log(configs.mel_generator_cfg, log_file)

    dataset = SpectrogramGenerationDataset(
        os.path.join(args.data, args.set), 
        configs.speaker_recognition_cfg['train_option']
    )

    _, valid_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])

    valid_dataloader = DataLoader(
        valid_dataset, 
        batch_size=configs.mel_generator_cfg['batch_size'],
        collate_fn=spectrogram_generation_collate_fn, 
        shuffle=False,
        drop_last=True,
    )

    asr_model = load_asr(configs.mel_generator_cfg['asr_weight'])
    spk_model = load_spk(configs.mel_generator_cfg['spk_weight'], num_classes=dataset.num_classes)
    
    gen_model = MelGenerator(
        asr_model=asr_model, 
        spk_model=spk_model,
        step_size = 0,
    ).to(configs.device)

    gen_model.load_state_dict(torch.load(args.weight))

        
    log(gen_model, log_file)
    log(f"Device: {configs.device}", log_file)
    log(f"Valid set size: {len(valid_dataset)}", log_file)
    log(f"Number of parameters: {sum(p.numel() for p in gen_model.parameters())}", log_file)

    accuracy = Accuracy(task="multiclass", num_classes=dataset.num_classes).to(configs.device)

    valid_history = valid_gen_net(gen_model, valid_dataloader, accuracy, log_file, beta=(1,1,1))
    log(
        f"[Valid] - " +
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
    
    parser.add_argument('-weight',
                        '--weight',
                        type=str,
                        required=False,
                        help="Path to a pretrained weight of the mel-generation model")


    args = parser.parse_args()
    main(args)