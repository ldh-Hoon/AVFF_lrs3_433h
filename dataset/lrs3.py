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

                self.data.append((video_path, audio_path))
        if self.count != 0:
            self.data = self.data[:self.count]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        video_path, audio_path = self.data[idx]

        video = self.load_video_and_audio(video_path)

        sample_rate, wav_data = wavfile.read(audio_path)
        data = torch.from_numpy(wav_data).float()
        audio_mel = self.mel_preprocessing_new(data.unsqueeze(0))
        return video, audio_mel


    def collate_fn(self, samples):
        min_frames = 16
        min_wav_length = min(sample[1].shape[0] for sample in samples)

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

    def mel_preprocessing(self, audio_data, sr=16000, n_mels=128, n_fft=1024, hop_length=512):
        n_fft = int(sr * 0.016)  # 16ms
        hop_length = int(sr * 0.004)  # 4ms
        
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio_data, 
            sr=sr, 
            n_mels=n_mels, 
            n_fft=n_fft, 
            hop_length=hop_length
        )       
        # hopping window 16ms, every 4ms, 128 bins

        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        return torch.tensor(log_mel_spectrogram.T, dtype=torch.float32)
    
    def mel_preprocessing_new2(self, audio_data, sr=16000):
        # kaldi.fbank 설정 수정: frame_shift=10으로 변경
        fbank = torchaudio.compliance.kaldi.fbank(
            audio_data,
            htk_compat=True,
            sample_frequency=sr,
            use_energy=False,
            window_type='hanning',
            num_mel_bins=128,
            dither=0.0,
            frame_shift=10  # <-- _wav2fbank와 동일하게 수정
        )

        # 길이 조절 방식 수정: 선형 보간으로 변경
        TARGET_LEN = 768
        fbank = torch.nn.functional.interpolate(
            fbank.unsqueeze(0).transpose(1, 2),
            size=(TARGET_LEN,),
            mode='linear',
            align_corners=False
        ).transpose(1, 2).squeeze(0)

        return fbank
    
    def mel_preprocessing_new(self, audio_data, sr=16000):
        fbank = kaldi.fbank(audio_data, htk_compat=True, sample_frequency=sr, use_energy=False, 
        window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=4 ,frame_length= 16)
        n_frames = fbank.shape[0]
        TARGET_LEN = 768
        p = TARGET_LEN - n_frames
        # cut and pad
        
        TARGET_LEN = 768
        fbank = torch.nn.functional.interpolate(
            fbank.unsqueeze(0).transpose(1, 2),
            size=(TARGET_LEN,),
            mode='linear',
            align_corners=False
        ).transpose(1, 2).squeeze(0)
        
        return fbank

    def _wav2fbank(self, filename):
        waveform, sr = torchaudio.load(filename)
        waveform = waveform - waveform.mean()

        try:
            fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False, window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)
        except:
            fbank = torch.zeros([512, 128]) + 0.01
            print('there is a loading error')

        target_length = self.target_length
        # n_frames = fbank.shape[0]

        # p = target_length - n_frames

        # # cut and pad
        # if p > 0:
        #     m = torch.nn.ZeroPad2d((0, 0, 0, p))
        #     fbank = m(fbank)
        # elif p < 0:
        #     fbank = fbank[0:target_length, :]

        fbank = torch.nn.functional.interpolate(fbank.unsqueeze(0).transpose(1,2), size=(target_length, ), mode='linear', align_corners=False).transpose(1,2).squeeze(0)

        return fbank

    def norm_fbank(self, fbank):
        norm_mean= -4.2677393
        norm_std= 4.5689974
        fbank = (fbank - norm_mean) / (norm_std * 2)
        return fbank


if __name__ == '__main__':
    from vis import visualize_batch

    import torch
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # root_path = "/media/ldh/sda1/lrs3"
    root_path = "/lrs3/30h_data"

    train_dataset = LRS3Dataset(root_path, split="train")

    num_workers = 20
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=num_workers, collate_fn=train_dataset.collate_fn)

    for video, audio in train_loader:
        print(f"Train Video shape: {video.shape}, {audio.shape}")
        visualize_batch(video, audio)
        break