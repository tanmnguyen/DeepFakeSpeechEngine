import os 
from torch.utils.data import Dataset

class SpeakerRecognitionDataset(Dataset):
    def __init__(self, data_path: str):
        self.data = dict() 
        self.spk2idx = {"unknown": 0}

        # read mel-spectrogram features from hdf5 file 
        with open(os.path.join(data_path, "melspectrogram_utterance_mapping"), 'r') as f:
            for line in f:
                utterance_id, path = line.split()
                self.data[utterance_id] = {
                    "utterance_id": utterance_id,
                    "melspec_path": os.path.join(data_path, "data", path.split('/')[-1]),
                    "speaker_id": self.spk2idx["unknown"],
                }

        # read speaker id 
        with open(os.path.join(data_path, "utt2spk"), 'r') as f:
            for line in f:
                utterance_id, speaker_id = line.split()
                # check if the utterance id is in the data
                if speaker_id not in self.spk2idx:
                    self.spk2idx[speaker_id] = len(self.spk2idx)

                if utterance_id in self.data:
                    self.data[utterance_id]["speaker_id"] = self.spk2idx[speaker_id]

        # check if all the utterance has speaker id
        for utterance_id, data in self.data.items():
            if data["speaker_id"] == self.spk2idx["unknown"]:
                print(f"\033[91m[ERROR] No speaker id for utterance: {utterance_id}\033[0m")

        # convert data dict to list 
        self.data = list(self.data.values())

        # set num class
        self.num_classes = len(self.spk2idx)

    def __len__(self):  
        return 100
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]