# This file generate a new spectrogram from the original spectrogram 
import os 
import time 
import torch
import librosa
import configs 
import argparse
import soundfile as sf
import torch.optim as optim

from utils.io import log
from torchmetrics import Accuracy
from torch.utils.data import DataLoader
from utils.networks import load_asr, load_spk
from utils.steps import train_gen_net, valid_gen_net
from utils.batch import spectrogram_generation_collate_fn
from datasets.SpectrogramGenerationDataset import SpectrogramGenerationDataset

from models.MelGenerator import MelGenerator


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
    # asr_model = load_asr(configs.mel_generator_cfg['asr_weight'])
    # spk_model = load_spk(configs.mel_generator_cfg['spk_weight'], num_classes=dataset.num_classes)

    gen_model.eval()
    for i, (melspectrogram_features, tokens, labels, speaker_labels) in enumerate(train_dataloader):
        melspectrogram_features, tokens, labels, speaker_labels = \
            melspectrogram_features.to(configs.device), \
            tokens.to(configs.device), \
            labels.to(configs.device), \
            speaker_labels.to(configs.device)
        output = gen_model(melspectrogram_features)

        inv_audio = librosa.feature.inverse.mel_to_audio(output[0].numpy(), sr=16000, n_fft=400, hop_length=160, window="hann")
        sf.write("inverted.wav", inv_audio, 16000)

        print(output.shape)
        break

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