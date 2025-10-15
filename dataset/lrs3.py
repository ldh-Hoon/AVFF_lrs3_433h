import os
import json
import random
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
import torchaudio, torchvision
from torchaudio.compliance import kaldi
import imageio
import csv
from scipy.io import wavfile

N_MFCC = 128
SIZE = 224
BATCH = 2


from tqdm import tqdm
import os
import json
import random
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
import torchaudio, torchvision
from torchaudio.compliance import kaldi
import imageio
import csv
from scipy.io import wavfile


import librosa
from pydub import AudioSegment


class LRS3Dataset(Dataset):
   def __init__(self, root_dir, split="pretrain", transform=None, noise=False, snr=5, count=0):
       self.root_dir = root_dir
       self.transform = transform
       self.data = []
       self.split = split
       self.noise = noise
       self.snr = snr
       self.count = count
       # pretrain trainval test


       self.load_tsv_metadata(os.path.join(root_dir, f"{split}.tsv"))


       print(f"{split}::{len(self.data)}")


   def load_metadata(self, metadata_file):
       with open(metadata_file, 'r') as f:
           metadata = json.load(f)


       for item in metadata:
           video_path = os.path.join(self.root_dir, item['path'])
           audio_path = video_path.replace(".mp4", ".wav")
           self.data.append((video_path, audio_path))
  
   def load_tsv_metadata(self, tsv_file_path):
        if 'train' in self.split:
            split = 'trainval'
        if 'test' in self.split:
            split = 'test'

        with open(tsv_file_path, 'r', encoding='utf-8') as f: 
            reader = csv.reader(f, delimiter='\t')
            next(reader)
            for row in reader:
                if not row:
                    continue
                video_path = row[1].replace('/video','')

                audio_path = row[2]

                video_path = video_path.split('lip_reading')[-1]


                noise = self.noise
                noise_type = "babble"
                snr = self.snr

                if not noise:
                    audio_path = audio_path.split('lip_reading')[-1]

                else:
                    
                    audio_path = audio_path.replace('/home/youngchae/Desktop/datasets/lip_reading/lrs3/audio/', 
                                                    f"{self.root_dir.split('lip_reading')[0]}/lip_reading/lrs3/noise_audio/").replace(
                                                        f'/{split}/',
                                                        f'/{noise_type}/{split}/snr_{snr}/') # noise
                    audio_path = audio_path.replace('/home/youngchae/Desktop/datasets', self.root_dir.split('lip_reading')[0]) 

                video_frames = row[3]
                if int(video_frames) < 150: # 최소 프레임 수 조건, 3.2s
                    continue

                self.data.append((video_path, audio_path))
        if self.count != 0:
            self.data = self.data[:self.count]


   def __len__(self):
       return len(self.data)
  
   def __getitem__(self, idx):
       video_path, audio_path = self.data[idx]


       video = self.load_video_and_audio(video_path)


       # torchaudio.load 사용
       data, sample_rate = torchaudio.load(audio_path)
      
       # ⚡️ 수정: DC offset 제거 및 1D numpy 배열로 변환 (librosa 입력 형식)
       if data.ndim > 1 and data.shape[0] == 1: # (1, Samples) 형태라면
           data = data.squeeze(0) # (Samples,) 형태로 변환


       # ⚡️ 핵심 수정: Torch Tensor를 NumPy 배열로 변환
       audio_data_np = data.cpu().numpy()


       # NumPy 배열을 librosa 기반 함수에 전달
       audio_mel = self.mel_preprocessing_new(audio_data_np)
      
       return video, audio_mel




   def collate_fn(self, samples):
       min_frames = 16
       min_wav_length = 768


       padded_frames = []
       padded_wavs = []


       for frames, wav in samples:
           if frames.shape[0] < min_frames:
               padding = min_frames - frames.shape[0]
               pad_frames = torch.zeros((padding, frames.shape[1], frames.shape[2], frames.shape[3]), dtype=frames.dtype)
               padded_frames.append(torch.cat((frames, pad_frames), dim=0))
           else:
               padded_frames.append(frames[:min_frames])
              
           padded_wavs.append(wav[:min_wav_length])


       return torch.stack(padded_frames), torch.stack(padded_wavs)
  
   def load_video_and_audio(self, video_path):
       reader = imageio.get_reader(video_path)
       frames = []
       W = SIZE
       H = SIZE


       for i, frame in enumerate(reader):
           if i % 5 == 0:  # 5fps로 조정
               frame = np.array(frame)
               frame = cv2.resize(frame, (H, W))
               frames.append(frame)


       if len(frames) == 0:
           raise ValueError(f"No frames found in video: {video_path}")


       frames = np.array(frames).transpose((0, 3, 1, 2))  # (T, H, W, C) -> (T, C, H, W)
       frames = frames.astype(np.float32) / 255.0
       num_frames = frames.shape[0]
       self.target_frames = num_frames


       return torch.tensor(frames, dtype=torch.float32)
  
   def load_audio(self, video_path):
       audio_segment = AudioSegment.from_file(video_path)


       audio_array = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)


       return audio_array
  
   def load_audio_new(self, video_path):
       waveform, sr = torchaudio.load(video_path)
       waveform = waveform - waveform.mean()
       return waveform




   def mel_preprocessing_new(self, audio_data_np, sr=16000):
      
       # ⚡️ 논문 설정 준수: 16ms window, 4ms shift (LRS3/AV-Sync 논문 표준)
       win_length = int(sr * 0.016) # 16ms window (256 samples)
       hop_length = int(sr * 0.004) # 4ms shift (64 samples)
      
       # ⚡️ Mel Band 안정화를 위해 FFT 크기는 512로 설정 (128 Mel Bins 커버)
       n_fft = 512


       # 1. Mel Spectrogram 계산
       mel_spectrogram = librosa.feature.melspectrogram(
           y=audio_data_np,
           sr=sr,
           n_mels=128,
           n_fft=n_fft,
           hop_length=hop_length,
           win_length=win_length,
           fmin=20.0
       )      


       # 2. 로그 파워 변환
       log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)


       # 3. (F, T) -> (T, F) 형태로 변환 및 텐서로 변환
       fbank = torch.tensor(log_mel_spectrogram.T, dtype=torch.float32)


       fbank = self.norm_fbank(fbank)

       # 4. Padding/Cut Logic (기존 로직 유지)
       n_frames = fbank.shape[0]
       TARGET_LEN = 768
       p = TARGET_LEN - n_frames
      
       if p > 0:
           m = torch.nn.ZeroPad2d((0, 0, 0, p))
           fbank = m(fbank)
       elif p < 0:
           fbank = fbank[0:TARGET_LEN, :]
          
       return fbank


   def norm_fbank(self, fbank):
       norm_mean= -4.2677393
       norm_std= 4.5689974
       fbank = (fbank - norm_mean) / (norm_std * 2)
       return fbank



