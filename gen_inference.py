# This file generate a new spectrogram from the original spectrogram 
import os 
import time 
import torch
import librosa
import configs 
import argparse
import numpy as np
import soundfile as sf
import torch.optim as optim
import matplotlib.pyplot as plt

from utils.io import log
from utils.networks import load_asr
from torch.utils.data import DataLoader
from utils.networks import load_state_dict
from utils.batch import process_mel_spectrogram
from utils.batch import spectrogram_generation_collate_fn
from datasets.SpectrogramGenerationDataset import SpectrogramGenerationDataset

from models.MelGenerator import MelGenerator

def compute_fft(audio, threshold=600, constant=1000):
    # Compute the FFT
    fft = np.fft.fft(audio)
    
    # # Compute the absolute values of the FFT
    # fft_abs = np.abs(fft)
    
    # # Check if the values are high and replace them with a constant
    # high_values_indices = np.where(fft_abs > threshold)
    # fft[high_values_indices] = 0

    # t = int(1e5 + 2e4)
    # fft[t:-t] = 0
    
    return fft


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

    gen_model = MelGenerator(
        asr_model=None, 
        spk_model=None
    ).to(configs.device)

    asr_model = load_asr(configs.mel_generator_cfg['asr_weight'])
    
    gen_model = load_state_dict(gen_model, args.weight)
    gen_model.eval()
    index = 0
    for i, (melspectrogram_features, tokens, labels, speaker_labels) in enumerate(train_dataloader):
        melspectrogram_features, tokens, labels, speaker_labels = \
            melspectrogram_features.to(configs.device), \
            tokens.to(configs.device), \
            labels.to(configs.device), \
            speaker_labels.to(configs.device)
        
        output = gen_model(melspectrogram_features)


        gen_melspec = output[index].squeeze(0).detach().numpy()
        ori_melspec = melspectrogram_features[index].squeeze(0).detach().numpy()

        # save_melspec_comparison_plot(gen_melspec, ori_melspec, 'melspec_comparison.png')
        # return 
        # output = gen_model(melspectrogram_features)

        inv_audio = librosa.feature.inverse.mel_to_audio(
            gen_melspec, sr=16000, n_fft=400, hop_length=160, window="hann"
        )

        ori_audio = librosa.feature.inverse.mel_to_audio(
            ori_melspec, sr=16000, n_fft=400, hop_length=160, window="hann"
        ) 
        fft_inv = compute_fft(inv_audio)
        fft_ori = np.fft.fft(ori_audio)

        plt.figure(figsize=(10, 6))
        plt.plot(np.abs(fft_inv), label="inverted audio") 
        plt.plot(np.abs(fft_ori), label="original audio")
        plt.title('FFT Result from Audio Files')
        plt.xlabel('Frequency (bin)')
        plt.ylabel('Magnitude')
        plt.legend()
        plt.grid(True)
        plt.show()

        # invert fft 
        inv_audio = np.fft.ifft(fft_inv).real
        ori_audio = np.fft.ifft(fft_ori).real

        sf.write("inverted.wav", inv_audio, 16000)
        sf.write("original.wav", ori_audio, 16000)
        return

        processed_mel = process_mel_spectrogram(output[index].unsqueeze(0))
        # decode with asr model 
        asr_output = asr_model.decode(processed_mel)
        print(asr_output)

        return


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
    
    parser.add_argument('-weight',
                        '--weight',
                        type=str,
                        required=True,
                        help="path to a weight file of the generator model")


    args = parser.parse_args()
    main(args)