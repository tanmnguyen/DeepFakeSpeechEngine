import librosa 
import soundfile as sf

def melspec2wav(melspectrogram, sr=16000, wav_path="output.wav"):
   converted_audio = librosa.feature.inverse.mel_to_audio(melspectrogram, sr=sr, n_fft=512, hop_length=160, window="hann")
   sf.write(wav_path, converted_audio, sr)
