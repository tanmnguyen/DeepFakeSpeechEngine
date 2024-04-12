import h5py

def read_melspectrogram_from_batch(batch, max_length=None):
    # this function optimize reading by minimizing file access operations

    # read all melspectrogram paths and utterance ids
    file2utterance_id = {} 
    for item in batch:
        melspec_path, utterance_id = item['melspec_path'], item['utterance_id']
        if melspec_path not in file2utterance_id:
            file2utterance_id[melspec_path] = []
        file2utterance_id[melspec_path].append(utterance_id)
   
    # read melspectrogram features
    melspectrogram_features = []
    for melspec_path in file2utterance_id:
        with h5py.File(melspec_path, 'r') as hf:
            for utterance_id in file2utterance_id[melspec_path]:
                if utterance_id not in hf:
                    raise ValueError(f"utterance_id {utterance_id} not found in the HDF5 file {melspec_path}!")
                mel = hf[utterance_id][:]
                if max_length is not None:
                    mel = mel[:, :max_length]
                melspectrogram_features.append(mel)

    return melspectrogram_features


def log(message, log_file):
    print(message)
    with open(log_file, 'a') as f:
        f.write(f"{message}\n")