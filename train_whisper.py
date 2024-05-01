# UNI: tmn2134
import os 
import numpy as np
import torch.nn.functional as F
import torch 
import whisper 
import soundfile as sf
import librosa

from models.whisper.audio import log_mel_spectrogram
EPSILON = 1e-10
def inv_log_mel_spectrogram(log_mel_spec, n_fft, hop_length, device=None):
    # Inverse the normalization applied in log_mel_spectrogram function
    log_spec = log_mel_spec * 4.0 - 4.0
    mel_spec = 10 ** log_spec  # Exponentiate to undo the logarithm

    # Invert the Mel spectrogram to obtain magnitudes
    filters = torch.pinverse(mel_filters(device, mel_spec.shape[0]))  # Compute the pseudo-inverse of the Mel filterbank
    magnitudes = filters @ mel_spec

    # Add a small epsilon to ensure magnitudes remain non-negative
    magnitudes = magnitudes.clamp_min(EPSILON)

    # Pad magnitudes to match the original STFT shape
    magnitudes = F.pad(magnitudes, (0, 1))

    # Reconstruct the complex spectrogram
    stft_complex = torch.sqrt(magnitudes) * torch.exp(1j * torch.zeros_like(magnitudes))

    # Compute the inverse STFT
    audio = torch.istft(stft_complex, n_fft, hop_length, window=torch.hann_window(n_fft).to(stft_complex.device))

    return audio


def inv_mel_spectrogram(mel_spec, n_fft, hop_length, device=None):
    # Invert the Mel spectrogram to obtain magnitudes
    filters = torch.pinverse(mel_filters(device, mel_spec.shape[0]))  # Compute the pseudo-inverse of the Mel filterbank
    magnitudes = filters @ mel_spec

    # Add a small epsilon to ensure magnitudes remain non-negative
    magnitudes = magnitudes.clamp_min(EPSILON)

    # Pad magnitudes to match the original STFT shape
    magnitudes = F.pad(magnitudes, (0, 1))

    # Reconstruct the complex spectrogram
    stft_complex = torch.sqrt(magnitudes) * torch.exp(1j * torch.zeros_like(magnitudes))

    # Compute the inverse STFT
    audio = torch.istft(stft_complex, n_fft, hop_length, window=torch.hann_window(n_fft).to(stft_complex.device))

    return audio

def inv_stft_spectrogram(magnitudes, n_fft, hop_length, device=None):
    recovered_magnitudes = torch.sqrt(magnitudes)
    recovered_phase = torch.zeros_like(stft)
    recovered_stft = recovered_magnitudes * torch.exp(1j * recovered_phase)
    # Compute the inverse STFT
    audio = torch.istft(recovered_stft, n_fft, hop_length, window=torch.hann_window(n_fft).to(recovered_stft.device))

    return audio

def mel_filters(device, n_mels: int) -> torch.Tensor:
    # Load the mel filterbank matrix
    filters_path = os.path.join("assets", "mel_filters.npz")
    with np.load(filters_path, allow_pickle=False) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)
    
def process_mel_spectrogram(mel_spec):
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0

    return log_spec 

# load base model
model = whisper.load_model("base.en")
# load audio 
audio = whisper.load_audio("egs/tedlium/db/TEDLIUM_release-3/legacy/train/sph/RobinChase_2012G.sph")
# pad or trim audio
audio = whisper.pad_or_trim(audio.flatten())
sf.write("original.wav", audio, 16000)

# compute mel 
mel, mel_spec = log_mel_spectrogram(audio, n_mels=128)

mel_spec = mel_spec.numpy()
inv_audio = librosa.feature.inverse.mel_to_audio(mel_spec, sr=16000, n_fft=400, hop_length=160, window="hann")
# inv_audio = inv_stft_spectrogram(stft, 400, 160, None)
# inv_audio = inv_mel_spectrogram(melspec, 400, 160, None)

# inv_audio = inv_log_mel_spectrogram(mel, 400, 160, None)
# print(inv_audio)
sf.write("inverted.wav", inv_audio, 16000)

mel_spec = librosa.feature.melspectrogram(
    y=audio, 
    n_mels=80,
    sr=16000, 
    n_fft=400,
    hop_length=160,
    fmin=0,
    window="hann",
)
# trim audio
mel_spec = mel_spec[:, :3000]
mel_spec = process_mel_spectrogram(torch.from_numpy(mel_spec))
print("checking", mel_spec.shape)
options = whisper.DecodingOptions(language="en", without_timestamps=True)
results = model.decode(mel_spec.unsqueeze(0), options)
print(results)
