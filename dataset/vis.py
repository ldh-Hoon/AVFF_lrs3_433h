import matplotlib.pyplot as plt
import os
import torch
import numpy as np


def save_reconstruction(video, audio, v_rec, a_rec, save_dir, step_or_epoch, log_db=False, max_frames=16):
   """
   영상 + 오디오 복원 결과 저장 함수
   - 오디오는 원본/복원을 한 그림에 위아래로 비교
   - 비디오는 프레임별 원본/복원을 2행 N열로 한 장에 저장
   """


   os.makedirs(save_dir, exist_ok=True)


   # --- 오디오 Mel ---
   mel_orig = audio[0].detach()
   mel_rec = a_rec[0].detach()


   if mel_orig.ndim == 3:
       mel_orig = mel_orig.squeeze(0)
   if mel_rec.ndim == 3:
       mel_rec = mel_rec.squeeze(0)


   mel_orig = mel_orig.T
   mel_rec = mel_rec.T


   if log_db:
       mel_orig = 20 * torch.log10(mel_orig + 1e-5)
       mel_rec = 20 * torch.log10(mel_rec + 1e-5)


   mel_orig_np = mel_orig.cpu().numpy()
   mel_rec_np = mel_rec.cpu().numpy()


   vmin = min(mel_orig_np.min(), mel_rec_np.min())
   vmax = max(mel_orig_np.max(), mel_rec_np.max())


   fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
   im0 = axes[0].imshow(mel_orig_np, aspect='auto', origin='lower', cmap='magma', vmin=vmin, vmax=vmax)
   axes[0].set_title(f'Original Audio - {step_or_epoch}')
   fig.colorbar(im0, ax=axes[0])


   im1 = axes[1].imshow(mel_rec_np, aspect='auto', origin='lower', cmap='magma', vmin=vmin, vmax=vmax)
   axes[1].set_title(f'Reconstructed Audio - {step_or_epoch}')
   fig.colorbar(im1, ax=axes[1])


   axes[1].set_xlabel("Time Frame")
   axes[1].set_ylabel("Mel Bin")
   plt.tight_layout()
   mel_path = os.path.join(save_dir, f'audio_comparison_{step_or_epoch}.png')
   plt.savefig(mel_path)
   plt.close(fig)
   print(f"[Saved] {mel_path}")


   # --- 비디오 (프레임 16개 비교) ---
   num_frames = video.size(2)
   n_select = min(max_frames, num_frames)


   frame_indices = np.linspace(0, num_frames-1, n_select, dtype=int)


   fig, axes = plt.subplots(2, n_select, figsize=(n_select*2, 4))


   for i, frame_index in enumerate(frame_indices):
       # 원본
       frame_orig = video[0, :, frame_index, :, :].permute(1, 2, 0).detach().cpu().numpy()
       frame_orig = (frame_orig - frame_orig.min()) / (frame_orig.max() - frame_orig.min() + 1e-6)
       axes[0, i].imshow(frame_orig)
       axes[0, i].axis('off')
       axes[0, i].set_title(f'F{frame_index}')


       # 복원
       frame_rec = v_rec[0, :, frame_index, :, :].permute(1, 2, 0).detach().cpu().numpy()
       frame_rec = (frame_rec - frame_rec.min()) / (frame_rec.max() - frame_rec.min() + 1e-6)
       axes[1, i].imshow(frame_rec)
       axes[1, i].axis('off')


   axes[0,0].set_ylabel("Original")
   axes[1,0].set_ylabel("Reconstructed")


   plt.tight_layout()
   video_path = os.path.join(save_dir, f'video_comparison_{step_or_epoch}.png')
   plt.savefig(video_path)
   plt.close(fig)
   print(f"[Saved] {video_path}")





