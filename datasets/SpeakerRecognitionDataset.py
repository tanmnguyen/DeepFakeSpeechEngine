import os 
from torch.utils.data import Dataset

# This dataset contains the melspectrogram feature, uterrance id and binary class label
class SpeakerRecognitionDataset(Dataset):
    def __init__(self, data_path: str, train_option: dict):
        self.data = dict() 
        self.utt2spkid = dict()

        if train_option["option"] == "spk2spk":
            self.spk2idx = train_option["spk2idx"]

        # read speaker id 
        with open(os.path.join(data_path, "utt2spk"), 'r') as f:
            for line in f:
                utterance_id, speaker_id = line.split()
                speaker_id = speaker_id.split("-")[0].split("_")[0]

                # speaker to speaker train option
                if train_option['option'] == "spk2spk" and speaker_id in self.spk2idx:
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

        self.idx_to_utteranceid = {i: utterance_id for i, utterance_id in enumerate(self.data.keys())}

        # determine num class 
        self.num_classes = len(self.spk2idx)

    def __len__(self):  
        return len(self.data)
    
    def __getitem__(self, idx):
        utterance_id = self.idx_to_utteranceid[idx]
        return self.data[utterance_id]