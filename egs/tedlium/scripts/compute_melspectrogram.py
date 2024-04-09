import os 
import time 
import h5py
import librosa
import argparse
import soundfile as sf
import multiprocessing as mp

def compute_spectrogram(hdf5_path: str, segment_dict, wav_dict):
    # Path to the HDF5 file
    os.makedirs(os.path.dirname(hdf5_path), exist_ok=True)

    with h5py.File(hdf5_path, 'w') as hf:
        for wav_id in segment_dict:
            # Read audio file
            audio, sr = librosa.load(wav_dict[wav_id], sr=16000)

            for utterance_id, from_, to_ in segment_dict[wav_id]:
                # Compute segment
                segment = audio[int(float(from_) * sr): int(float(to_) * sr)]

                # Compute mel-spectrogram
                melspectrogram = librosa.feature.melspectrogram(
                    y=segment, 
                    sr=sr, 
                    n_mels=80,
                    n_fft=512,
                    hop_length=160,
                    fmin=0,
                    window="hann",
                )

                hf.create_dataset(utterance_id, data=melspectrogram)

    return f"Done computing mel spectrogram features - {hdf5_path}"

def split_segment_dict(segment_dict, nprocs):
    n_dicts = len(segment_dict)
    n_dicts_per_proc = n_dicts // nprocs

    segment_dict_per_proc = []
    for i in range(nprocs):
        start = i * n_dicts_per_proc
        end = n_dicts if i == nprocs - 1 else (i + 1) * n_dicts_per_proc

        segment_dict_per_proc.append({k: segment_dict[k] for k in list(segment_dict.keys())[start:end]})
  
    return segment_dict_per_proc


def read_data(hdf5_path, utterance_id):
    with h5py.File(hdf5_path, 'r') as hf:
        if utterance_id not in hf:
            raise ValueError(f"utterance_id {utterance_id} not found in the HDF5 file")
        return hf[utterance_id][:]

def main(args):

    print("Compute Mel Spectrogram features for", args.data)

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

    args.nprocs = min(args.nprocs, len(segment_dict))

    # split the segments into multiple processes
    segment_dict_per_proc = split_segment_dict(segment_dict, args.nprocs)
    
    start_time = time.time()

    # Compute mel spectrogram features in parallel
    processes = []
    for i in range(args.nprocs):
        processes.append(
            mp.Process(
                target=compute_spectrogram, 
                args=(os.path.join(args.data, f"data/melspectrogram_{i}.hdf5"), segment_dict_per_proc[i], wav_dict)
            )
        )
        processes[-1].start()

    for p in processes:
        p.join()
        
    print(f"Time taken: {time.time() - start_time:.2f}s")

    # write utterance to hdf5 mapping 
    with open(os.path.join(args.data, "melspectrogram_utterance_mapping"), 'w') as f:
        for i in range(args.nprocs):
            segment_dict = segment_dict_per_proc[i]
            for wav_id in segment_dict:
                for utterance_id, _, _ in segment_dict[wav_id]:
                    # get absolute path to the hdf5 file
                    hdf5_path = os.path.join(args.data, f"data/melspectrogram_{i}.hdf5")
                    f.write(f"{utterance_id} {hdf5_path}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-data',
                        '--data',
                        required=True,
                        help="path to data set. Provide data/train, data/dev, or data/test explicitly.")
    
    parser.add_argument('-nprocs',
                        '--nprocs',
                        type=int,
                        default=1,
                        required=True,
                        help="Number of processes to use for parallel computation. Default is 1.")

    args = parser.parse_args()
    main(args)