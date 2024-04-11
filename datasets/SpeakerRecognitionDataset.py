import os 
from torch.utils.data import Dataset

# This dataset contains the melspectrogram feature, uterrance id and binary class label
class SpeakerRecognitionDataset(Dataset):
    def __init__(self, data_path: str, train_option):
        self.data = dict() 
        self.spk2idx = dict()
        self.utt2spkid = dict()

        # read speaker id 
        with open(os.path.join(data_path, "utt2spk"), 'r') as f:
            for line in f:
                utterance_id, speaker_id = line.split()
                speaker_id = speaker_id.split("-")[0].split("_")[0]

                # speaker to speaker train option
                if train_option[0] == "spk2spk":
                    if speaker_id in train_option[1:]:
                        if speaker_id not in self.spk2idx:
                            self.spk2idx[speaker_id] = len(self.spk2idx)
                        self.utt2spkid[utterance_id] = self.spk2idx[speaker_id]

                        

        # read mel-spectrogram features from hdf5 file 
        with open(os.path.join(data_path, "melspectrogram_utterance_mapping"), 'r') as f:
            for line in f:
                utterance_id, path = line.split()

                if utterance_id in self.utt2spkid:
                    self.data[utterance_id] = {
                        "utterance_id": utterance_id,
                        "melspec_path": os.path.join(data_path, "data", path.split('/')[-1]),
                        "speaker_id": self.utt2spkid[utterance_id],
                    }

        # convert data dict to list 
        self.data = list(self.data.values())

        # determine num class 
        self.num_classes = len(self.spk2idx)

    def __len__(self):  
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]