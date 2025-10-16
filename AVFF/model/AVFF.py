import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from torch import Tensor
from torch.nn import Linear, Module


# COMBINE_SIZE, LSTM_DIM, N_MFCC, SIZE 등은 기존과 동일
COMBINE_SIZE = 1024
LSTM_DIM = 512
N_MFCC = 128
SIZE = 224


# Marlin 및 audioMAE 모델 임포트 (기존과 동일)
from .marlin_pytorch.config import resolve_config, Downloadable
from .marlin_pytorch.model.marlin import Marlin
from .marlin_pytorch.model.modules import MLP
from .audioMAE.models_mae import *


def create_mask(slice_size):
   """exactly 50% masked"""
   mask = torch.zeros(slice_size)
   mask[:slice_size // 2] = 1
   mask = mask[torch.randperm(slice_size)]
   return mask.reshape(slice_size, 1)


class S2SNetwork(nn.Module):
   def __init__(self, feature_dim, in_len, out_len):
       super(S2SNetwork, self).__init__()
       self.in_len = in_len
       self.out_len = out_len
       self.transformer_block = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=8)
       self.mlp = nn.Linear(in_len, out_len)


   def forward(self, x):
       B, S, N, F = x.shape
       x = x.permute(0, 1, 3, 2)
       x = self.mlp(x)
       x = x.permute(0, 1, 2, 3)


       x = x.reshape(B*S, self.out_len, -1)
       x = x.permute(1, 0, 2)
       x = self.transformer_block(x)
       x = x.reshape(self.out_len, B, S, -1)
       return x.permute(1, 2, 0, 3)


class AVFF(nn.Module):
   def __init__(self,
               num_slices=8,
               v_emb_dim=768,
               v_num_patches=196,
               a_num_patches=48,
               marlin_path = "/home/ldh/avsr/AvHubert/av_hubert/avhubert/AVFF/pretrained_models/MARLIN/marlin_vit_base_ytf.full.pt",
               audioMAE_path = '/home/ldh/avsr/AvHubert/av_hubert/avhubert/AVFF/pretrained_models/AudioMAE/pretrained.pth'):
       super(AVFF, self).__init__()
      
       self.marlin_path = marlin_path
       self.audioMAE_path = audioMAE_path
       self.num_slices = num_slices


       self.video_encoder = None
       self.video_decoder = None
       self.audio_encoder = None
       self.audio_decoder = None
       self.audio_mae = None


       a_emb_dim = 256


       self.v_discriminator = MLP([1536, int(1536 / 8), 1], build_activation=nn.LeakyReLU)
       self.a_discriminator = MLP([a_emb_dim, int(a_emb_dim / 8), 1], build_activation=nn.LeakyReLU)
      
       self.A2V = S2SNetwork(v_emb_dim, a_num_patches, v_num_patches)
       self.V2A = S2SNetwork(v_emb_dim, v_num_patches, a_num_patches)


       self.init_from_pretrained(marlin_path, audioMAE_path)
      
   def init_from_pretrained(self, marlin_path, audioMAE_path):
       marlin = Marlin.from_file("marlin_vit_base_ytf", marlin_path)
       self.marlin = marlin
       self.video_encoder = self.marlin.encoder
       self.video_decoder = self.marlin.decoder


       audio_MAE = mae_vit_base_patch16(in_chans=1, audio_exp=True, img_size=(1024, 128), decoder_mode=1)
       checkpoint = torch.load(audioMAE_path, map_location='cuda')
       audio_MAE.load_state_dict(checkpoint['model'], strict=False)


       self.audio_mae = audio_MAE
       self.audio_encoder = self.audio_mae.blocks
       self.audio_decoder = self.audio_mae.decoder_blocks


   @property
   def device(self):
       return self.video_encoder.norm.weight.device


   def encoding(self, v, a):
       # video embedding
       v = self.marlin.extract_features(v)
       b, _, _ = v.shape
       v = v.reshape(b, self.num_slices, -1, v.shape[-1])


       # audio embedding
       a = a.unsqueeze(1)
       a = torch.nn.functional.pad(a, (0, 0, 0, 1024-768))
       a = self.audio_mae.forward_encoder_no_mask(a)
       a = a[:, 1:, :]
       a = a[:, :384, :]
       a = a.reshape(b, self.num_slices, -1, a.shape[-1])


       return v, a


   def patchify_video(self, x, patch_size=(2,16,16)):
       B, C, T, H, W = x.shape
       pt, ph, pw = patch_size
       assert T % pt == 0 and H % ph == 0 and W % pw == 0
       x = x.reshape(B, C, T//pt, pt, H//ph, ph, W//pw, pw)
       x = x.permute(0, 2, 4, 6, 1, 3, 5, 7)  # [B, T//pt, H//ph, W//pw, C, pt, ph, pw]
       x = x.reshape(B, -1, C*pt*ph*pw)
       return x


   def cross_modality_fusion(self, v, a):
       slice_size = v.shape[1]
       m_v = create_mask(slice_size)
       m_a = 1 - m_v


       mask_v = m_v.reshape(1, slice_size, 1, 1).to(self.device)
       mask_a = m_a.reshape(1, slice_size, 1, 1).to(self.device)


       v_vis = v * mask_v
       a_vis = a * mask_a


       v_vis = v_vis[v_vis != 0].reshape(v_vis.shape[0], -1, v_vis.shape[2], v_vis.shape[3])
       a_vis = a_vis[a_vis != 0].reshape(a_vis.shape[0], -1, a_vis.shape[2], a_vis.shape[3])
      
       a_v = self.A2V(a_vis)
       v_a = self.V2A(v_vis)
      
       i_v = (m_v.squeeze() == 0).nonzero(as_tuple=True)[0].reshape(-1)
       i_a = (m_a.squeeze() == 0).nonzero(as_tuple=True)[0].reshape(-1)


       v_fused = v.clone()
       a_fused = a.clone()
       v_fused[:, i_v] = a_v
       a_fused[:, i_a] = v_a
      
       # 원본 마스크된 인덱스를 저장
       m_v_indices = (m_v.squeeze() == 0).nonzero().squeeze(1)
       m_a_indices = (m_a.squeeze() == 0).nonzero().squeeze(1)


       return v_fused, a_fused, v_a, a_v, m_v_indices, m_a_indices
  
   def decoding(self, v, a):
       B, S, N_v, F_v = v.shape
       B, S, N_a, F_a = a.shape
      
       v = v.reshape(B, S*N_v, F_v)
       a = a.reshape(B, S*N_a, F_a)


       a = torch.nn.functional.pad(a, (0, 0, 0, 512-384))


       v_dec = self.marlin.enc_dec_proj(v)
       v_dec = self.video_decoder.forward_no_mask(v_dec)


       v_rec = self.video_decoder.unpatch_to_img(v_dec)


       a_dec, _, _ = self.audio_mae.forward_decoder_no_mask(a)
       a_rec = self.audio_mae.unpatchify(a_dec)
       a_rec = a_rec[:, :, :768, :]


       return v_rec, a_rec
  


   def patchify_img(self, x: Tensor, pt=2,ph=16,pw=16) -> Tensor:
       """
       x: (B, C, T, H, W)
       returns: (B, num_patches, patch_volume * C)
       """
       # T, H, W를 patch 단위로 나누고 patch 차원을 C*pt*ph*pw로 합치기
       x = rearrange(
           x,
           "b c (t pt) (h ph) (w pw) -> b (t h w) (c pt ph pw)",
           pt=pt,
           ph=ph,
           pw=pw
       )
       return x




   def forward_loss(self, v, a, v_encoded, a_encoded, v_rec, a_rec, i_v, i_a,
                lambda_c=0.1, lambda_rec=1.0, lambda_adv=0.1, lambda_gp=1.0):


       B = v.size(0)


       # ---------------- Contrastive Loss ----------------
       # v_encoded, a_encoded는 이미 encoding된 텐서
       v_encoded = v_encoded.reshape(B, self.num_slices, -1, 768)
       a_encoded = a_encoded.reshape(B, self.num_slices, -1, 768)


       masked_v_encoded = v_encoded[:, i_v, :, :]  # (B, num_mask_slices, N_patch, D)
       masked_a_encoded = a_encoded[:, i_a, :, :]  # (B, num_mask_slices, N_patch, D)


       # contrastive loss 계산
       L_c = self.contrastive_loss(masked_v_encoded, masked_a_encoded)


       # ---------------- Patchify & Masked Indices ----------------
       # Video
       pt, ph, pw = 2, 16, 16


       # v, v_rec를 patchify
       v_patches = self.patchify_img(v).reshape(B, self.num_slices, -1, 1536)        # [B, N_patch, 768]
       v_rec_patches = self.patchify_img(v_rec).reshape(B, self.num_slices, -1, 1536)


       # masking index i_v 사용
       masked_v = v_patches[:, i_v]          # [B, num_masked_patch, 1536]
       masked_v_rec = v_rec_patches[:, i_v] # 동일 shape


       # Audio
       a_padded = torch.nn.functional.pad(a, (0, 0, 0, 1024 - 768), value=0).unsqueeze(1)
       a_rec_padded = torch.nn.functional.pad(a_rec, (0, 0, 0, 1024 - 768), value=0)


       a_patches = self.audio_mae.patchify(a_padded)[:, :384, :]  # (B, 384, 256)
       a_rec_patches = self.audio_mae.patchify(a_rec_padded)[:, :384, :]  # (B, 384, 256)
      
       a_patches = a_patches.reshape(B, self.num_slices, -1, 256)
       a_rec_patches = a_rec_patches.reshape(B, self.num_slices, -1, 256)
      
       masked_a = a_patches[:, i_a]
       masked_a_rec = a_rec_patches[:, i_a]


       # ---------------- Reconstruction Loss ----------------
       loss_rec = self.reconstruction_loss(masked_a_rec, masked_a, masked_v_rec, masked_v)


       # ---------------- Adversarial Loss ----------------
       L_G_adv, L_D_adv = self.adv_loss(masked_v_rec, masked_v, masked_a_rec, masked_a)


       # ---------------- Gradient Penalty ----------------
       gp = self.gradient_penalty_fn(masked_v, masked_v_rec, masked_a, masked_a_rec)


       # ---------------- 최종 손실 ----------------
       L_G = L_c * lambda_c + loss_rec * lambda_rec + L_G_adv * lambda_adv
       L_D = L_D_adv + gp * lambda_gp


       return L_G, L_D, (L_c, loss_rec, L_G_adv)








   def forward_with_mask(self, video, audio):
       v_encoded, a_encoded = self.encoding(video, audio)


       v_fused, a_fused, v_a, a_v, i_v, i_a = self.cross_modality_fusion(v_encoded, a_encoded)
      
       v_rec, a_rec = self.decoding(v_fused, a_fused)


       # forward_loss에 L_c와 인코딩된 텐서들을 전달
       L_G, L_D, (L_c, loss_rec, L_G_adv) = self.forward_loss(video, audio, v_fused, a_fused, v_rec, a_rec, i_v, i_a)
      
       return v_rec, a_rec, v_a, a_v, i_v, i_a, (L_G, L_D, L_c, loss_rec, L_G_adv, v_encoded, a_encoded)
  


   def forward_no_mask(self, video, audio):
       # 1. 인코딩
       v_encoded, a_encoded = self.encoding(video, audio)


       v_rec, a_rec = self.decoding(v_encoded, a_encoded)


       i_v = torch.arange(v_encoded.shape[1], device=self.device)
       i_a = torch.arange(a_encoded.shape[1], device=self.device)


       L_G, L_D, (L_c, loss_rec, L_G_adv) = self.forward_loss(
           video, audio, v_encoded, a_encoded, v_rec, a_rec, i_v, i_a
       )


       return v_rec, a_rec, None, None, i_v, i_a, (L_G, L_D, L_c, loss_rec, L_G_adv, v_encoded, a_encoded)


  
   # 그 외 forward 함수들 제거 또는 수정 (forward_2, forward, forward_no_masking 등)
   # WGAN-GP 학습을 위해서는 forward_with_mask 함수만 사용하면 됩니다.


   # gradient_penalty_fn, contrastive_loss, reconstruction_loss, adv_loss 함수는 기존과 동일
   def gradient_penalty_fn(self, v, v_rec, a, a_rec, chunk_size=1) -> Tensor:
       B = v.size(0)
       gp_total = 0.0
       for i in range(0, B, chunk_size):
           b = min(chunk_size, B-i)
           alpha_v = torch.rand(b, 1, 1, 1, 1, device=v.device)
           alpha_a = torch.rand(b, 1, 1, 1, device=a.device)
          
           interpolates_v = alpha_v * v[i:i+b] + (1-alpha_v) * v_rec[i:i+b]
           interpolates_a = alpha_a * a[i:i+b] + (1-alpha_a) * a_rec[i:i+b]


           interpolates_v.requires_grad_(True)
           interpolates_a.requires_grad_(True)


           D_interpolates_v = self.v_discriminator(interpolates_v)
           D_interpolates_a = self.a_discriminator(interpolates_a)


           gradients_v = torch.autograd.grad(
               outputs=D_interpolates_v,
               inputs=interpolates_v,
               grad_outputs=torch.ones_like(D_interpolates_v),
               create_graph=True, retain_graph=True
           )[0].reshape(b, -1)


           gradients_a = torch.autograd.grad(
               outputs=D_interpolates_a,
               inputs=interpolates_a,
               grad_outputs=torch.ones_like(D_interpolates_a),
               create_graph=True, retain_graph=True
           )[0].reshape(b, -1)


           gp_total += ((gradients_v.norm(2, dim=1)-1)**2).mean() + ((gradients_a.norm(2, dim=1)-1)**2).mean()
       return gp_total / (B / chunk_size)




   def contrastive_loss(self, v, a, tau=0.1):
       B, num_slices, N_v, _ = v.shape
      
       # Patch 평균 (논문에서 bar 의미)
       v_mean = v.mean(dim=2).reshape(B * num_slices, -1)  # (B*num_slices, D)
       a_mean = a.mean(dim=2).reshape(B * num_slices, -1)  # (B*num_slices, D)


       # 정규화 (cosine 유사도와 동일하게)
       v_mean = F.normalize(v_mean, dim=-1)
       a_mean = F.normalize(a_mean, dim=-1)


       # Similarity matrix
       logits = torch.matmul(v_mean, a_mean.T) / tau
       labels = torch.arange(v_mean.size(0)).to(v_mean.device)


       # v→a, a→v 둘 다 계산
       loss_v2a = F.cross_entropy(logits, labels)
       loss_a2v = F.cross_entropy(logits.T, labels)


       return 0.5 * (loss_v2a + loss_a2v)




  
   def reconstruction_loss(self, masked_a_rec, masked_a, masked_v_rec, masked_v):


       loss_a = F.mse_loss(masked_a_rec, masked_a)
       loss_v = F.mse_loss(masked_v_rec, masked_v)
       return loss_a + loss_v






      
   def adv_loss(self, masked_v_rec, masked_v, masked_a_rec, masked_a):
       Dv_rec = self.v_discriminator(masked_v_rec)
       Dv = self.v_discriminator(masked_v)
       Da_rec = self.a_discriminator(masked_a_rec)
       Da = self.a_discriminator(masked_a)
       L_G_adv = -torch.mean(Dv_rec) - torch.mean(Da_rec)
       L_D_adv = torch.mean(Dv_rec) - torch.mean(Dv) + torch.mean(Da_rec) - torch.mean(Da)
       return L_G_adv, L_D_adv


   def _crop_face(self, v):
       return self.marlin._crop_face(v)
  
   def augmentation(self, v):
       return None



