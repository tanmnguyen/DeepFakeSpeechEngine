# S2SGAN: Non-Parallel Speech-to-Speech GAN with Independent Speaker and Speech Recognition Networks

**Name**: Tan Nguyen | **UNI**: tmn2134 

**Abstract:**

*The foundation for a multitude of downstream tasks lies in the flexibility of generating genuine speech from speech that can meet a wide array of needs for different applications. In this work, we introduce a non-parallel speech-to-speech GAN network designed for flexible, genuine speech generation across various applications. Our network features independent speaker identification and speech recognition modules, allowing for effective adaptation to new speech domains without full network retraining—only the GAN's generative and discriminative components require updates. We enhance our system with a joint adversarial fine-tuning of the speaker recognition during generation training, promoting robust synthetic speech output. Additionally, our framework supports easy integration of the latest advancements in speaker and speech recognition models to improve generation performance. We explore the inclusion of multi-speaker identification networks to generate anonymized mel-spectrogram outputs, which are then used to reconstruct authentic-sounding human speech. Our approach's effectiveness is validated quantitatively using the TED-LIUM release 3 corpora*

## Tools 
We list the tools and dependencies used in this repository below. 
- Whisper
- Pytorch 2.2.1
- Torchvision 0.17.1
- Torchaudio 2.2.1
- librosa 
- tqdm
- h5py
- torchmetrics
- einops
- deep_phonemizer
- matplotlib

For installation of these tools, please refer to the instalation section in this readme file. 


## Installation 
1. Download the repository and change to the project folder.
```bash
git clone https://github.com/tanmnguyen/DeepFakeSpeechEngine.git
cd DeepFakeSpeechEngine
```
2. We offer a Docker image equipped with all necessary dependencies specifically for utilizing the Kaldi data preparation recipe. Note: The processes of training, testing, and decoding (inference) do not require this Docker environment. Therefore, if your data is already prepared, you may choose to bypass this Docker setup to save time.
```bash
docker-compose up --build
docker-compose run .
```
3. Install the requirement dependencies 
```bash
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install git+https://github.com/openai/whisper.git
pip install -r requirements.txt
```

## Important Directories and Executables
We provide the training scripts for all three models presented in our paper. To make it simple, we provide a main script for training the Mel-Generator Model for all 4 partition sets.  
```bash 
bash train.sh 
```
This script specifies the arguments for training and test the performance of our model as specified in our paper. Furthermore, we also provide details and training steps for each speech recognition, speaker recognition, and mel generator model separatedly below. 

In addition, to compute the decoding, we also provide a script for decoding for all 4 partitions. Execute the following to start the decoding
```bash
bash decode.sh
```
### Speech Recognition Model
To fine-tune the Whisper speech recognition model 
```bash 
CUDA_VISIBLE_DEVICES=0 python train_asr.py --data egs/tedlium/data 
```
### Speaker Recognition Model
To train Speaker Recognition model. Here, we arbitrarily choose start speaker index for our partition set A, B, C, and D in the paper. 
```bash 
python train_spk_recognition.py --data egs/tedlium/data/ --set train --start_spk_idx 0
python train_spk_recognition.py --data egs/tedlium/data/ --set train --start_spk_idx 150
python train_spk_recognition.py --data egs/tedlium/data/ --set train --start_spk_idx 732
python train_spk_recognition.py --data egs/tedlium/data/ --set train --start_spk_idx 800
```

### Mel Generator Model
To train Spectrogram Generation model 
```bash 
python train_generator.py --data egs/tedlium/data/ --set train --start_spk_idx 0 --spk weights/spk/spk_model_A.pt
python train_generator.py --data egs/tedlium/data/ --set train --start_spk_idx 150 --spk weights/spk/spk_model_B.pt
python train_generator.py --data egs/tedlium/data/ --set train --start_spk_idx 732 --spk weights/spk/spk_model_C.pt
python train_generator.py --data egs/tedlium/data/ --set train --start_spk_idx 800 --spk weights/spk/spk_model_D.pt
```


<!-- ## Inference 
Inference generator 
```bash 
python gen_inference.py --data egs/tedlium/data/ --set train --start_spk_idx 0 --weight weights/gen/gen_model_A.pt
``` -->