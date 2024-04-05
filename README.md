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
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```
## Train 
Train ASR model 
```bash 
python train_asr.py --data egs/tedlium/data 
```

kaldi/src/featbin/compute-spectrogram-feats --output-format=kaldi "scp:egs/tedlium/data/train/wav.scp" "ark:egs/tedlium/data/train/spectrogram_feats.ark"
