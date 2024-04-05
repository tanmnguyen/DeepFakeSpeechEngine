import os 
import math
import argparse
import librosa
import soundfile as sf

import sys 
sys.path.append('../../')

from utils.io import read_ark

import logging
import os
import scipy 
import numpy as np
from packaging.version import parse as V
from scipy.io.wavfile import write

EPS = 1e-10

# TODO: checkout torch audio.kaldi for mel spectrogram computation
def main(args):
    segment_dict = {}
    with open(os.path.join(args.data, "segments")) as f:
        for line in f:
            utterance_id, audio_id, from_, to_ = line.strip().split()
            segment_dict[utterance_id] = (audio_id, from_, to_)

    # sph dict 
    sph_dict = {}
    with open(os.path.join(args.data, "wav.scp")) as f:
        for line in f:
            content = line.strip().split()
            utterance_id = content[0]
            audio_path = content[-2]
            sph_dict[utterance_id] = audio_path

    with open(os.path.join(args.data, "spectrogram_feats.scp")) as f:
        for line in f:
            content = line.strip().split()

            utterance_id = content[0]
            feature_path = content[1]

            melspec = read_ark(feature_path)
            melspec = melspec.T

            print(melspec.shape)

            # audio id 
            audio_id, from_, to_ = segment_dict[utterance_id]

            # fbank_feat = read_ark(feature_path)

            # print(utterance_id, fbank_feat.shape)

            # spc = logmelspc_to_linearspc(
            #     fbank_feat,
            #     fs=16000,
            #     n_mels=40,
            #     n_fft=512,
            # )

            # print("spc", spc.shape)

            audio, sr = librosa.load(sph_dict[audio_id], sr=None, mono=True)
            segment = audio[int(float(from_)*sr): int(float(to_)*sr)]

            print("sr", sr)
            # melspec = librosa.feature.melspectrogram(
            #     y=segment, 
            #     sr=sr, 
            #     n_fft=512,
            #     hop_length=160,
            #     n_mels=257,
            #     fmin=0,
            #     window="hamm",
            # )

            print(melspec.shape, melspec.max(), melspec.min())

            converted_audio = librosa.feature.inverse.mel_to_audio(
                melspec, sr=sr, n_fft=512, hop_length=160, window="hamm")
            
            sf.write('converted_audio.wav', converted_audio, sr)
            sf.write('original_audio.wav', segment, sr)
    
            return 
            # tranpose spc  
            spc = spc.T
                
            print("spc", spc.shape)

            # y = librosa.feature.inverse.mel_to_audio(spc)
            y = griffin_lim(
                spc,
                n_fft=512,
                n_shift=160,
                win_length=512,
                window="hann",
                n_iters=20,
            )

            print("y", y.shape, max(y), min(y))

            y = (y * np.iinfo(np.int16).max).astype(np.int16)
            write(
                "audio-output-test.wav",
                16000,
                y,
            )
            print(max(y), np.iinfo(np.int16).max)
            return
    # collect audio file path
    # id_to_audio_path = {}
    # with open(os.path.join(args.data, "wav.scp")) as f:
    #     for line in f:
    #         content = line.strip().split()
    #         id_to_audio_path[content[0]] = content[-2]

    # # read scp feature file
    # id_to_feature_path = {}
    # with open(os.path.join(args.data, "feats.scp")) as f:
    #     for line in f:
    #         content = line.strip().split()
    #         id_to_feature_path[content[0]] = content[1]

    # # ark_file = "/app/egs/tedlium/data/train/data/raw_mfcc_train.1.ark:33"
    # # features = read_ark(ark_file)
    # # print("mfcc features", features.shape)
    # # read information from the segments 
    # with open(os.path.join(args.data, "segments")) as f:
    #     cnt = 0
    #     for line in f:
    #         utterance_id, audio_id, from_, to_ = line.strip().split()
    #         # get audio file 
    #         audio_file = id_to_audio_path[audio_id]
    #         # read audio data 
    #         audio, sr = librosa.load(audio_file, sr=None, mono=True)
    #         # get segment of audio
    #         segment = audio[int(float(from_)*sr): int(float(to_)*sr)]
    #         # print(utterance_id, audio_id, from_, to_, len(segment))
    #         # write audio to mp3 file
    #         sf.write("output_wav_file.wav", segment, sr)

    #         mel_spectrogram = librosa.feature.melspectrogram(
    #             y=segment, 
    #             sr=sr, 
    #             n_fft=int(25 / 1000 * sr),
    #             hop_length=int(10 / 1000 * sr),
    #             n_mels=40,
    #             fmin=0,
    #         )

    #         print("mel spectrogram", mel_spectrogram.shape)
    #         mfcc = read_ark(id_to_feature_path[utterance_id])
    #         print("mfcc", mfcc.shape)

    #         # n_mfcc = 20  # Number of MFCC coefficients
    #         # mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel_spectrogram), n_mfcc=n_mfcc)

    #         # print("-> mfcc", mfcc.shape)
    #         cnt += 1
    #         if cnt > 10:
    #             return
            # print(id_to_audio_path[audio_id], from_, to_)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-data',
                        '--data',
                        required=True,
                        help="path to data set")

    args = parser.parse_args()
    main(args)



import os 
import h5py
import librosa
import argparse
from tqdm import tqdm

# hdf5_path = 'data/test/data/mel_spectrograms.hdf5'
def compute_spectrogram(data_path: str, segment_dict, wav_dict):
    print("Computing mel spectrogram features for", data_path)
    
    # Path to the HDF5 file
    hdf5_path = os.path.join(data_path, "data", "mel_spectrograms.hdf5")
    os.makedirs(os.path.dirname(hdf5_path), exist_ok=True)

    with h5py.File(hdf5_path, 'w') as hf:
        for wav_id in tqdm(segment_dict):
            # Read audio file
            audio, sr = librosa.load(wav_dict[wav_id], sr=16000)

            for utterance_id, from_, to_ in segment_dict[wav_id]:
                # Compute segment
                segment = audio[int(float(from_) * sr): int(float(to_) * sr)]

                # Compute mel-spectrogram
                melspectrogram = librosa.feature.melspectrogram(
                    y=segment, 
                    sr=sr, 
                    n_fft=512,
                    hop_length=160,
                    fmin=0,
                    window="hann",
                )

                hf.create_dataset(utterance_id, data=melspectrogram)
    # with h5py.File(hdf5_path, 'w') as hf:
    #     for utterance_id in segment_dict:
    #         wav_id, from_, to_ = segment_dict[utterance_id]

    #         # Read audio file
    #         audio, sr = librosa.load(wav_dict[wav_id], sr=16000)

    #         print(sr)
    #         # Compute segment
    #         segment = audio[int(float(from_) * sr): int(float(to_) * sr)]

    #         # Compute mel-spectrogram
    #         melspectrogram = librosa.feature.melspectrogram(
    #             y=segment, 
    #             sr=sr, 
    #             n_fft=512,
    #             hop_length=160,
    #             fmin=0,
    #             window="hann",
    #         )

    #         converted_audio = librosa.feature.inverse.mel_to_audio(melspectrogram, sr=sr, n_fft=512, hop_length=160, window="hann")
    #         sf.write('converted_audio.wav', converted_audio, sr)


    #         hf.create_dataset(utterance_id, data=melspectrogram)

    #         return 

    # # Path to the extended ark file
    # ark_file = 'data/test/data/raw_spectrogram_test.ark'
    # # Path to the corresponding .scp file
    # scp_file = 'data/test/raw_spectrogram_test.scp'

    # os.makedirs(os.path.dirname(ark_file), exist_ok=True)
    # os.makedirs(os.path.dirname(scp_file), exist_ok=True)

    # with open(ark_file, 'wb') as ark_file, open(scp_file, 'w') as scp_file:
    #     # Initialize the starting position for writing features in Ark file
    #     start_position = 0
        
    #     # Iterate through each utterance in segment_dict
        

    #         # Convert mel-spectrogram to log-scale
    #         log_melspectrogram = librosa.power_to_db(melspectrogram, ref=np.max)

    #         # Write features to Ark file
    #         ark_file.write(log_melspectrogram.T.astype(np.float32).tobytes())

    #         # Calculate the end position after writing the features
    #         end_position = start_position + len(log_melspectrogram.T) * len(log_melspectrogram.T[0]) * 4  # 4 bytes per float32
            
    #         # Write SCP entry with Ark file path and start position
    #         scp_file.write(f'{utterance_id} {os.path.abspath(ark_file)}:{start_position}:{end_position}\n')

    #         # Update the start position for the next feature
    #         start_position = end_position   

    #         return 
            

# def read_data(utterance_id):
#     with h5py.File(hdf5_path, 'r') as hf:
#         if utterance_id not in hf:
#             raise ValueError(f"utterance_id {utterance_id} not found in the HDF5 file")
#         return hf[utterance_id][:]
    
def main(args):
    # build segment dict 
    segment_dict = {} 
    with open(os.path.join(args.data, "segments"), 'r') as f:
        for line in f:
            utterance_id, wav_id, from_, to_ = line.strip().split()
            if wav_id not in segment_dict:
                segment_dict[wav_id] = []
            segment_dict[wav_id].append((utterance_id, from_, to_))

    # build wav dict 
    wav_dict = {}
    with open(os.path.join(args.data, "wav.scp"), 'r') as f:
        for line in f:
            content = line.strip().split()
            utterance_id = content[0]
            audio_path = content[-2]
            wav_dict[utterance_id] = audio_path

    compute_spectrogram(args.data, segment_dict, wav_dict)
    # return 
    # data = read_ark("data/test/data/raw_spectrogram_test.ark:0")
    # data = read_data("AimeeMullins_2009P-0001782-0002881")
    # print(data.shape)

    # converted_audio = librosa.feature.inverse.mel_to_audio(data, sr=16000, n_fft=512, hop_length=160, window="hamm")
    # sf.write('converted_audio.wav', converted_audio, 16000)

    return 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-data',
                        '--data',
                        required=True,
                        help="path to data set. Provide data/train, data/dev, or data/test explicitly.")

    args = parser.parse_args()
    main(args)



sr = 16000
melspectrogram = read_data("/app/egs/tedlium/data/test/data/melspectrogram_9.hdf5", "RobertGupta_2010U-0033863-0035117")

converted_audio = librosa.feature.inverse.mel_to_audio(melspectrogram, sr=sr, n_fft=512, hop_length=160, window="hann")
sf.write('converted_audio.wav', converted_audio, sr)
return 
