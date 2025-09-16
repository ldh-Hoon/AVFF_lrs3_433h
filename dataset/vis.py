# vis_lrs3_save.py
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

def display_mel_spectrogram(mel_tensor, save_path, title="Mel Spectrogram", figsize=(12,4)):
    """
    mel_tensor: (time, n_mels) 또는 (n_mels, time)
    save_path: 저장할 파일 경로
    """
    if torch.is_tensor(mel_tensor):
        mel = mel_tensor.detach().cpu().numpy()
    else:
        mel = mel_tensor

    # (time, n_mels) -> (n_mels, time)
    if mel.shape[0] != 128:  # n_mels이 128이 아니라면 transpose
        mel = mel.T

    plt.figure(figsize=figsize)
    plt.imshow(mel, origin='lower', aspect='auto', interpolation='nearest', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.xlabel("Frames")
    plt.ylabel("Mel Bins")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def display_video_frames(video_tensor, save_path, title="Video Frames", max_frames=16):
    """
    video_tensor: (B, T, C, H, W)
    save_path: 저장할 파일 경로
    """
    video = video_tensor[0]  # 배치 0번째
    T, C, H, W = video.shape
    n_frames = min(T, max_frames)
    
    fig, axes = plt.subplots(1, n_frames, figsize=(2*n_frames, 2))
    for i in range(n_frames):
        frame = video[i]  # (C,H,W)
        frame = frame.permute(1,2,0).detach().cpu().numpy()  # (H,W,C)
        frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-6)
        #if C == 3:
            #frame = frame[..., ::-1]  # BGR -> RGB
        axes[i].imshow(frame)
        axes[i].axis('off')
        axes[i].set_title(f"Frame {i+1}")
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_batch(video, audio, save_dir='vis/', v_rec=None, a_rec=None):
    """
    video: (B,T,C,H,W)
    audio: (B, T_mel, n_mels)
    save_dir: 시각화 파일 저장 디렉토리
    v_rec, a_rec: optional reconstruction tensors
    """
    os.makedirs(save_dir, exist_ok=True)

    display_mel_spectrogram(audio[0], save_path=os.path.join(save_dir, "original_audio_mel.png"), title="Original Audio Mel")
    display_video_frames(video, save_path=os.path.join(save_dir, "original_video_frames.png"), title="Original Video Frames")

    if a_rec is not None:
        display_mel_spectrogram(a_rec[0], save_path=os.path.join(save_dir, "reconstructed_audio_mel.png"), title="Reconstructed Audio Mel")
    if v_rec is not None:
        display_video_frames(v_rec, save_path=os.path.join(save_dir, "reconstructed_video_frames.png"), title="Reconstructed Video Frames")


import matplotlib.pyplot as plt
import os


# -----------------------------
# Reconstruction 저장 함수
# -----------------------------
def save_reconstruction(video, audio, v_rec, a_rec, save_dir, step_or_epoch):
   os.makedirs(save_dir, exist_ok=True)
  
   # --- 오디오 스펙트로그램 ---
   mel_audio = audio[0].detach().cpu().numpy().T  # (T, F) -> (F, T) ?
   mel_rec = a_rec[0].detach().cpu().numpy().T
  
   plt.figure(figsize=(10,4))
   plt.imshow(mel_audio, aspect='auto', origin='lower')
   plt.colorbar()
   plt.title(f'Original Audio - Step/Epoch {step_or_epoch}')
   plt.savefig(os.path.join(save_dir, f'audio_original_{step_or_epoch}.png'))
   plt.close()


   plt.figure(figsize=(10,4))
   plt.imshow(mel_rec, aspect='auto', origin='lower')
   plt.colorbar()
   plt.title(f'Reconstructed Audio - Step/Epoch {step_or_epoch}')
   plt.savefig(os.path.join(save_dir, f'audio_rec_{step_or_epoch}.png'))
   plt.close()
  
   # --- 비디오 프레임 이미지 ---
   num_frames = video.size(2)
   frame_dir = os.path.join(save_dir, f'video_frames_{step_or_epoch}')
   os.makedirs(frame_dir, exist_ok=True)
  
   for i in range(num_frames):
       frame_orig = video[0, :, i, :, :].permute(1,2,0).detach().cpu().numpy()
       frame_orig = (frame_orig - frame_orig.min()) / (frame_orig.max() - frame_orig.min() + 1e-6)
       plt.imsave(os.path.join(frame_dir, f'orig_{i:03d}.png'), frame_orig)
      
       frame_rec = v_rec[0, :, i, :, :].permute(1,2,0).detach().cpu().numpy()
       frame_rec = (frame_rec - frame_rec.min()) / (frame_rec.max() - frame_rec.min() + 1e-6)
       plt.imsave(os.path.join(frame_dir, f'rec_{i:03d}.png'), frame_rec)





