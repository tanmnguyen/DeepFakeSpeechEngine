import os 
from torch.utils.data import Dataset

class ASRMelSpecDataset(Dataset):
    def __init__(self, data_path: str):
        super().__init__()
            
        self.data = dict() 

        # read mel-spectrogram features from hdf5 file 
        with open(os.path.join(data_path, "melspectrogram_utterance_mapping"), 'r') as f:
            for line in f:
                utterance_id, path = line.split()
                self.data[utterance_id] = {
                    "utterance_id": utterance_id,
                    "melspec_path": os.path.join(data_path, "data", path.split('/')[-1]),
                    "text": "",
                }

        # text transcription file 
        with open(os.path.join(data_path, "text"), 'r') as f:
            text_transcriptions = f.readlines()
            for line in text_transcriptions:
                try:
                    utterance_id, text = line.split(maxsplit=1)
                    if utterance_id in self.data:
                        self.data[utterance_id]["text"] = text.strip()
                except:
                    self.data[utterance_id]["text"] = "" 
                    print(f"\033[91m[ERROR PARSING - SET TEXT TO EMPTY]\033[0m {line}")

        self.idx_to_utteranceid = {i: utterance_id for i, utterance_id in enumerate(self.data.keys())}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        utterance_id = self.idx_to_utteranceid[idx]
        return self.data[utterance_id]
    

