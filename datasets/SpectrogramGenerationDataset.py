# UNI: tmn2134
import os 
from torch.utils.data import Dataset

from datasets.ASRMelSpecDataset import ASRMelSpecDataset
from datasets.SpeakerRecognitionDataset import SpeakerRecognitionDataset

class SpectrogramGenerationDataset(Dataset):
    def __init__(self, data_path: str, train_option: dict):
        asr_dataset = ASRMelSpecDataset(data_path)
        spk_dataset = SpeakerRecognitionDataset(data_path, train_option)
        
        # combine dataset 
        self.data = dict() 
        for utterance_id in asr_dataset.data:
            if utterance_id in spk_dataset.data:
                self.data[utterance_id] = {
                    "utterance_id": utterance_id,
                    "melspec_path": asr_dataset.data[utterance_id]["melspec_path"],
                    "text": asr_dataset.data[utterance_id]["text"],
                    "speaker_id": spk_dataset.data[utterance_id]["speaker_id"]
                }
                
        self.num_classes = spk_dataset.num_classes
        self.idx_to_utteranceid = {i: utterance_id for i, utterance_id in enumerate(self.data.keys())}
        self.len = len(self.data)

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        utterance_id = self.idx_to_utteranceid[idx]
        return self.data[utterance_id]
    