import os 

from torch.utils.data import Dataset

class TedliumDataset(Dataset):
    def __init__(self, data_path: str):
        super().__init__()
            
        self.data = dict() 

        # features file 
        with open(os.path.join(data_path, "feats.scp"), 'r') as f:
            feature_paths = f.readlines()
            for line in feature_paths:
                id, path = line.split()
                self.data[id] = {"feature_path": path}

        # text transcription file 
        with open(os.path.join(data_path, "text"), 'r') as f:
            text_transcriptions = f.readlines()
            for line in text_transcriptions:
                try:
                    id, text = line.split(maxsplit=1)
                    if id in self.data:
                        self.data[id]["text"] = text.strip()
                except:
                    self.data[id]["text"] = "" 
                    print(f"\033[91m[ERROR PARSING - SET TEXT TO EMPTY]\033[0m {line}")

        self.idx_to_id = {i: id for i, id in enumerate(self.data.keys())}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        id = self.idx_to_id[idx]
        return self.data[id]
    

