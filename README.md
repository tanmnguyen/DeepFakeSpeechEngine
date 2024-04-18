# DeepFakeSpeechEngine
Starting of a Deep Learning engine for Speech to Speech. 

**Note**: This is an ongoing project, please expect changes and updates to occur on the regular basis. 

## Installation 
Download this repository and install the Docker image.
```
git clone https://github.com/tanmnguyen/DeepFakeSpeechEngine.git
cd DeepFakeSpeechEngine
docker-compose up --build
```
To enter the interactive docker environment
```
docker-compose run .
```

When the data preparation is ready, we can only use conda virtual environment instead of docker with the following dependencies (lighter):
```bash 
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install git+https://github.com/openai/whisper.git
pip install -r requirements.txt
```
## Train 
Train speech recognition model 
```bash 
CUDA_VISIBLE_DEVICES=0 python train_asr.py --data egs/tedlium/data 
```

Train Speaker Recognition model 
```bash 
CUDA_VISIBLE_DEVICES=1 python train_spk_recognition.py --data egs/tedlium/data/
```

Train Spectrogram Generation model 
```bash 
CUDA_VISIBLE_DEVICES=1 python train_generator.py --data egs/tedlium/data/ --set train
```

kaldi/src/featbin/compute-spectrogram-feats --output-format=kaldi "scp:egs/tedlium/data/train/wav.scp" "ark:egs/tedlium/data/train/spectrogram_feats.ark"

## Inference 
Inference generator 
```bash 
python gen_inference.py --data egs/tedlium/data/ --set train --weight weights/gen/gen_model.pt
```