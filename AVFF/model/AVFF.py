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
   return mask.view(slice_size, 1)


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


       x = x.view(B*S, self.out_len, -1)
       x = x.permute(1, 0, 2)
       x = self.transformer_block(x)
       x = x.view(self.out_len, B, S, -1)
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


       self.v_discriminator = MLP([v_emb_dim, int(v_emb_dim / 4), 1], build_activation=nn.LeakyReLU)
       self.a_discriminator = MLP([v_emb_dim, int(v_emb_dim / 4), 1], build_activation=nn.LeakyReLU)
      
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
       v = v.view(b, self.num_slices, -1, v.shape[-1])


       # audio embedding
       a = a.unsqueeze(1)
       a = torch.nn.functional.pad(a, (0, 0, 0, 256))
       a = self.audio_mae.forward_encoder_no_mask(a)
       a = a[:, 1:, :]
       a = a[:, :384, :]
       a = a.view(b, self.num_slices, -1, a.shape[-1])


       return v, a


   def cross_modality_fusion(self, v, a):
       slice_size = v.shape[1]
       m_v = create_mask(slice_size)
       m_a = 1 - m_v


       mask_v = m_v.view(1, slice_size, 1, 1).to(self.device)
       mask_a = m_a.view(1, slice_size, 1, 1).to(self.device)


       v_vis = v * mask_v
       a_vis = a * mask_a


       v_vis = v_vis[v_vis != 0].view(v_vis.shape[0], -1, v_vis.shape[2], v_vis.shape[3])
       a_vis = a_vis[a_vis != 0].view(a_vis.shape[0], -1, a_vis.shape[2], a_vis.shape[3])
      
       a_v = self.A2V(a_vis)
       v_a = self.V2A(v_vis)
      
       i_v = (m_v.squeeze() == 0).nonzero(as_tuple=True)[0].view(-1)
       i_a = (m_a.squeeze() == 0).nonzero(as_tuple=True)[0].view(-1)


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
      
       v = v.view(B, S*N_v, F_v)
       a = a.view(B, S*N_a, F_a)
       a = torch.nn.functional.pad(a, (0, 0, 0, 128))


       v_dec = self.marlin.enc_dec_proj(v)
       v_dec = self.video_decoder.forward_no_mask(v_dec)
       v_rec = self.video_decoder.unpatch_to_img(v_dec)
      
       a_dec, _, _ = self.audio_mae.forward_decoder_no_mask(a)
       a_rec = self.audio_mae.unpatchify(a_dec)
       a_rec = a_rec[:, :, :768, :]


       return v_rec, a_rec


   def forward_loss(self, L_c, v, a, v_encoded, a_encoded, v_rec, a_rec, i_v, i_a,
                   #lambda_c=0.0, lambda_rec=1.0, lambda_adv=0.0, lambda_gp=0.0):
                   lambda_c=0.01, lambda_rec=1.0, lambda_adv=0.1, lambda_gp=1.0):
      
       B = v_encoded.size(0)
      
       # 인코딩된 원본 텐서에서 마스크된 부분 추출
       masked_v = v_encoded[:, i_v]
       masked_a = a_encoded[:, i_a]


       # 재구성된 텐서를 인코더 패치 형태로 변환 후 마스크된 부분 추출
       # 이 부분이 핵심적인 수정 사항입니다.
       # 디코딩된 텐서(v_rec)를 다시 인코딩 패치 형태(B, 8, -1, 768)로 변환해야 합니다.
       v_rec_patches = self.marlin.extract_features(v_rec).view(B, self.num_slices, -1, 768)
       a_rec_padded = torch.nn.functional.pad(a_rec, (0, 0, 0, 1024 - 768), mode='constant', value=0)


       # 1. 패딩된 오디오를 audioMAE 인코더에 전달
       a_rec_encoded = self.audio_mae.forward_encoder_no_mask(a_rec_padded)
      
       # 2. `cls_token` (첫 번째 토큰) 제거
       # audioMAE는 1024x128 입력에 대해 512개의 패치를 생성하므로,
       # 출력은 (B, 512 + 1, 768) 형태입니다.
       a_rec_encoded = a_rec_encoded[:, 1:, :] # (B, 512, 768)
      
       # 3. 패치 개수를 명시적으로 계산하여 view에 적용
       # 512 패치를 8개 슬라이스로 나누면, 슬라이스당 64개 패치
       num_patches_per_slice_a = 512 // self.num_slices # 512 / 8 = 64
      
       a_rec_patches = a_rec_encoded.view(B, self.num_slices, num_patches_per_slice_a, 768)
       a_rec_patches_sliced = a_rec_patches[:, :, :48, :]


       masked_v_rec = v_rec_patches[:, i_v]
       masked_a_rec = a_rec_patches_sliced[:, i_a]


       # Contrastive Loss (이미 계산되어 전달됨)
       loss_c = L_c
      
       # Reconstruction Loss
       loss_rec = self.reconstruction_loss(a, a_rec, v, v_rec, mask_a=i_a, mask_v=i_v)
      
       # Adversarial Loss
       L_G_adv, L_D_adv = self.adv_loss(masked_v_rec, masked_v, masked_a_rec, masked_a)
      
       # Gradient Penalty
       gp = self.gradient_penalty_fn(masked_v, masked_v_rec, masked_a, masked_a_rec)
      
       # 최종 생성자 손실
       L_G = loss_c * lambda_c + loss_rec * lambda_rec + L_G_adv * lambda_adv
      
       # 최종 판별자 손실 (WGAN-GP 공식)
       L_D = L_D_adv + gp * lambda_gp
      
       return L_G, L_D, (loss_c, loss_rec, L_G_adv)


   def forward_with_mask(self, video, audio):
       v_encoded, a_encoded = self.encoding(video, audio)
       L_c = self.contrastive_loss(v_encoded, a_encoded)
      
       v_fused, a_fused, v_a, a_v, i_v, i_a = self.cross_modality_fusion(v_encoded, a_encoded)
      
       v_rec, a_rec = self.decoding(v_fused, a_fused)


       # forward_loss에 L_c와 인코딩된 텐서들을 전달
       L_G, L_D, _ = self.forward_loss(L_c, video, audio, v_encoded, a_encoded, v_rec, a_rec, i_v, i_a)
      
       return v_rec, a_rec, v_a, a_v, i_v, i_a, (L_G, L_D, L_c, v_encoded, a_encoded)
  
   # 그 외 forward 함수들 제거 또는 수정 (forward_2, forward, forward_no_masking 등)
   # WGAN-GP 학습을 위해서는 forward_with_mask 함수만 사용하면 됩니다.


   # gradient_penalty_fn, contrastive_loss, reconstruction_loss, adv_loss 함수는 기존과 동일
   def gradient_penalty_fn(self, v, v_rec, a, a_rec) -> Tensor:
       B = v.size(0)
      
       alpha = torch.rand(B, 1, 1, 1).to(self.device)
       interpolates_v = alpha * v + (1 - alpha) * v_rec
       interpolates_a = alpha * a + (1 - alpha) * a_rec
      
       interpolates_v.requires_grad_(True)
       interpolates_a.requires_grad_(True)


       D_interpolates_v = self.v_discriminator(interpolates_v)
       D_interpolates_a = self.a_discriminator(interpolates_a)


       gradients_v = torch.autograd.grad(
           outputs=D_interpolates_v,
           inputs=interpolates_v,
           grad_outputs=torch.ones(D_interpolates_v.size(), device=self.device),
           create_graph=True,
           retain_graph=True
       )[0]
       gradients_a = torch.autograd.grad(
           outputs=D_interpolates_a,
           inputs=interpolates_a,
           grad_outputs=torch.ones(D_interpolates_a.size(), device=self.device),
           create_graph=True,
           retain_graph=True
       )[0]
       gradients_v = gradients_v.view(B, -1)
       gradients_a = gradients_a.view(B, -1)
       gp_v = ((gradients_v.norm(2, dim=1) - 1) ** 2).mean()
       gp_a = ((gradients_a.norm(2, dim=1) - 1) ** 2).mean()
       return gp_v + gp_a


   def contrastive_loss(self, v, a, tau=0.1):
       B, num_slices, N_v, _ = v.shape
      
       # Patch 평균 (논문에서 bar 의미)
       v_mean = v.mean(dim=2).view(B * num_slices, -1)  # (B*num_slices, D)
       a_mean = a.mean(dim=2).view(B * num_slices, -1)  # (B*num_slices, D)


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




   def reconstruction_loss(self, x_a, x_a_rec, x_v, x_v_rec, mask_a=None, mask_v=None, num_slices=8):
       # Audio


       if x_a_rec.shape != x_a.shape:
           x_a_rec = x_a_rec.squeeze(1)


       B_a, T_a, F_a = x_a.shape
       loss = 0.0


       if mask_a is not None and len(mask_a) > 0:
           mask_bool_a = torch.zeros(B_a, T_a, device=x_a.device, dtype=torch.bool)
           mask_a = torch.tensor(mask_a, device=x_a.device)
           slice_len = T_a // num_slices
           for b in range(B_a):
               for s in mask_a:
                   start = s * slice_len
                   end = start + slice_len
                   mask_bool_a[b, start:end] = True
           mask_bool_a = mask_bool_a.unsqueeze(-1)  # (B, T, 1)
           x_a_masked = x_a[mask_bool_a.expand_as(x_a)]
           x_a_rec_masked = x_a_rec[mask_bool_a.expand_as(x_a_rec)]
           loss += F.mse_loss(x_a_rec_masked, x_a_masked)


       # Video
       B_v, C_v, F_v, H_v, W_v = x_v.shape
       if mask_v is not None and len(mask_v) > 0:
           mask_bool_v = torch.zeros(B_v, F_v, device=x_v.device, dtype=torch.bool)
           mask_v = torch.tensor(mask_v, device=x_v.device)
           slice_len = F_v // num_slices
           for b in range(B_v):
               for s in mask_v:
                   start = s * slice_len
                   end = start + slice_len
                   mask_bool_v[b, start:end] = True
           mask_bool_v = mask_bool_v[:, None, :, None, None]  # (B,1,F,1,1)
           mask_bool_v = mask_bool_v.expand(B_v, C_v, F_v, H_v, W_v)
           x_v_masked = x_v[mask_bool_v]
           x_v_rec_masked = x_v_rec[mask_bool_v]
           loss += F.mse_loss(x_v_rec_masked, x_v_masked)


       return loss




      
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



