import os 
import time 
import torch 
import configs 
import argparse 

from utils.io import log
from torch.utils.data import DataLoader
from utils.batch import speech_recognition_collate_fn
from datasets.ASRMelSpecDataset import ASRMelSpecDataset

from models.whisper.model import ModelDimensions
from models.ASRWhisper import ASRWhisper

from utils.steps import train_net, valid_net

import torch.optim as optim

# create a directory to save the results
str_time = time.strftime("%m-%d-%Y-%H-%M-%S")
result_dir = os.path.join(configs.result_dir, f'{str_time}')
os.makedirs(result_dir, exist_ok=True)

# define log file 
log_file = os.path.join(result_dir, 'log.txt')

def main(args):
    train_dataset = ASRMelSpecDataset(os.path.join(args.data, "test"))
    valid_dataset = ASRMelSpecDataset(os.path.join(args.data, "dev"))

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=configs.speech_recognition_cfg['batch_size'],
        collate_fn=speech_recognition_collate_fn, 
        shuffle=True
    )

    valid_dataloader = DataLoader(
        valid_dataset, 
        batch_size=configs.speech_recognition_cfg['batch_size'],
        collate_fn=speech_recognition_collate_fn, 
        shuffle=False
    )

    dims = ModelDimensions(
        n_mels=80, 
        n_audio_ctx=1500, 
        n_audio_state=384, 
        n_audio_head=6, 
        n_audio_layer=4, 
        n_vocab=51864, 
        n_text_ctx=448, 
        n_text_state=384, 
        n_text_head=6, 
        n_text_layer=4
    )

    model = ASRWhisper(
        dims, 
        pad_token=configs.speech_recognition_cfg['tokenizer'].eot,
        whisper_model_weight = "weights/asr/tiny_whisper_model.pth"
    ).to(configs.device)

    log(model, log_file)
    log(f"Device: {configs.device}", log_file)
    log(f"Number of parameters: {sum(p.numel() for p in model.parameters())}", log_file)


    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() 
                        if not any(nd in n for nd in configs.speech_recognition_cfg['no_decay'])],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() 
                        if any(nd in n for nd in configs.speech_recognition_cfg['no_decay'])],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = optim.AdamW(optimizer_grouped_parameters, 
        lr=configs.speech_recognition_cfg['learning_rate'], 
        eps=1e-8
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=len(train_dataloader) * 5, 
        gamma=configs.speech_recognition_cfg['scheduler_gamma']
    )

    for epoch in range(configs.speech_recognition_cfg['epochs']):
        train_history = train_net(model, train_dataloader, scheduler, optimizer, log_file)
        log(
            f"[Train] Epoch: {epoch+1}/{configs.speech_recognition_cfg['epochs']} - " + 
            f"Loss: {train_history['loss']} | " + 
            f"WER: {train_history['wer']} | " + 
            f"SER: {train_history['ser']}",
            log_file
        )

        valid_history = valid_net(model, valid_dataloader)
        log(
            f"[Valid] Epoch: {epoch+1}/{configs.speech_recognition_cfg['epochs']} - " + 
            f"Loss: {valid_history['loss']} | " + 
            f"WER: {valid_history['wer']} | " + 
            f"SER: {valid_history['ser']}",
            log_file
        )
        
        torch.save(model.state_dict(), os.path.join(result_dir, f"asr_model_epoch_{epoch+1}.pt"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-data',
                        '--data',
                        required=True,
                        help="path to kaldi data format directory. This should contains dev, test, and train folders")


    args = parser.parse_args()
    main(args)