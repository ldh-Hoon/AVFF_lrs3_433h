
import os


import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


from einops import rearrange
from torch import Tensor
from torch.nn import Linear, Module

COMBINE_SIZE = 1024
LSTM_DIM = 512
N_MFCC = 128
SIZE = 224

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

        self.video_encoder = None
        self.video_decoder = None
        
        self.audio_encoder = None
        self.audio_decoder = None
        self.audio_mae = None

        self.v_discriminator = MLP([768, int(v_emb_dim / 4), 1],
                                 build_activation=nn.LeakyReLU)
        self.a_discriminator = MLP([128, int(128 / 4), 1],
                                 build_activation=nn.LeakyReLU)
        
        self.num_slices = num_slices

        self.A2V = S2SNetwork(v_emb_dim, a_num_patches, v_num_patches)
        self.V2A = S2SNetwork(v_emb_dim, v_num_patches, a_num_patches)

        self.init_from_pretrained(marlin_path, audioMAE_path)
        
    def init_from_pretrained(self, marlin_path, audioMAE_path):
        # load MARLIN
        marlin = Marlin.from_file("marlin_vit_base_ytf", marlin_path)

        self.marlin = marlin
        self.video_encoder = self.marlin.encoder
        self.video_decoder = self.marlin.decoder

        # load audio-MAE
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
        """
        v: (Batch, F_v, C=3, H=224, W=224)
        a: (Batch, F_a, L=128)
        
        L: mel_bins
        F_v: 16 frames, F_a: 768 frames

        video is 5 fps, and should be cropped (with self._crop_face)
        
        audio sampled 16kHz to Mel spectrogram
        hopping window size 16ms, every 4ms
        """
        #v = self._crop_face(v) # crop face with faceX-zoo
        
        # video embedding
        # v: (B, F, C, H, W)
        # v = v.permute(0, 1, 2, 3, 4) # (B, C, F, H, W) 

        v = self.marlin.extract_features(v)
        #x_v = self.video_encoder.pos_embedding(x_v) # <- SinCosPositionalEmbedding이라 사용 x
        # AVFF 논문상으로는 2x16x16 patch
        # YTF에서 학습된 MARLIN pretrained model
        b, _, _ = v.shape
        # torch.Size([1, 1568, 768])


        # audio embedding
        a = a.unsqueeze(1) # to (B, C=1, F, L)
        a = torch.nn.functional.pad(a, (0, 0, 0, 256))
    
        a = self.audio_mae.forward_encoder_no_mask(a)
        #a, mask, ids_restore, _ = self.audio_mae.forward_encoder_(a.float(), 0.001)
        a = a[:, 1:, :]
        #a = self.audio_mae.norm(a)
        # N: cls token 제거 513 -> 512 [:, 1:, :]
        # torch.Size([1, 513, 768])
        a = a[:, :384, :]
        # 1024 => 512(64*8) seq len, make 384(48*8) 
        # make 8 slices

        v = v.view(b, self.num_slices, -1, v.shape[-1])  # (B, num_slices, N, C)
        a = a.view(b, self.num_slices, -1, a.shape[-1])  # (B, num_slices, N, C)
        # print("slices:", v.shape, a.shape)
        # slices: torch.Size([1, 8, 196, 768]) torch.Size([1, 8, 64, 768])
        # a: torch.Size([1, 8, 48, 768])


        return v, a


    def cross_modality_fusion(self, v, a):
        
        # masking
        v_msk, a_msk, v_vis, a_vis, m_v, m_a = self.complementary_masking(v, a)

        # v_vis, a_vis: torch.Size([1, 4, 197, 768]) torch.Size([1, 4, 64, 768])
        # (B, slice/2, seq, dim), slice/2 = 4 
        # slice 개수는 경험적으로 설정했다고 함

        # V2A
        a_v = self.A2V(a_vis)
        v_a = self.V2A(v_vis)

        # print("v_a, a_v:", v_a.shape, a_v.shape)

        i_v = (m_v.squeeze() == 0).nonzero(as_tuple=True)[0].view(1, -1).squeeze(0)
        i_a = (m_a.squeeze() == 0).nonzero(as_tuple=True)[0].view(1, -1).squeeze(0)
        # (batch, 8) shape mask로부터 0인 index만 얻어서 (4) 형태로 [[2, 4, 6, 7]]와 같이 생성
        # batch내에서 mask는 같다고 가정
        v[:, i_v] = a_v
        a[:, i_a] = v_a
        
        # print(v.shape, a.shape)
        # 다른 modality slice를 삽입

        return v, a, v_a, a_v, i_v, i_a
    
    # complementary_masking
    def complementary_masking(self, v, a):
        slice_size = v.shape[1] 

        m_v = create_mask(slice_size)
        m_a = 1 - m_v

        mask_v = m_v.view(1, slice_size, 1, 1).to(self.device) 
        mask_a = m_a.view(1, slice_size, 1, 1).to(self.device) 

        v_msk = v * (1 - mask_v)
        a_msk = a * (1 - mask_a)

        v_vis = v * mask_v
        a_vis = a * mask_a

        v_vis = v_vis[v_vis != 0].view(v_vis.shape[0], -1, v_vis.shape[2], v_vis.shape[3]) 
        a_vis = a_vis[a_vis != 0].view(v_vis.shape[0], -1, a_vis.shape[2], a_vis.shape[3]) 

        # print("v_msk, a_msk, v_vis, a_vis, m_v, m_a:", v_msk.shape, a_msk.shape, v_vis.shape, a_vis.shape, m_v, m_a)

        return v_msk, a_msk, v_vis, a_vis, m_v, m_a
    
    
    def decoding(self, v, a):
        B, S, N_v, F_v = v.shape

        B, S, N_a, F_a = a.shape

        v = v.view(B, S*N_v, F_v)
        a = a.view(B, S*N_a, F_a)
        a = torch.nn.functional.pad(a, (0, 0, 0, 128)) # 128
        

        v = self.marlin.enc_dec_proj(v)
        v = self.video_decoder.forward_no_mask(v)
        v_rec = self.video_decoder.unpatch_to_img(v)
        

        a, _, _ = self.audio_mae.forward_decoder_no_mask(a) # pred, _, _
        a_rec = self.audio_mae.unpatchify(a)
        a_rec = a_rec[:, :, :768, :]

        return v, a, v_rec, a_rec

    def forward_no_masking(self, video, audio):
        #video = self._crop_face(video).to(self.device)
        v, a = self.encoding(video, audio)
        L_c = self.contrastive_loss(v, a)
        
        # v_msk, a_msk, v_vis, a_vis, m_v, m_a = self.complementary_masking(x_v, x_a)
        
        #v, a, v_a, a_v, i_v, i_a = self.cross_modality_fusion(v, a)
        
        v, a, v_rec, a_rec = self.decoding(v, a)
        v_rec_ = v_rec
        a_rec_ = a_rec

        loss_rec = self.reconstruction_loss(v_rec, video, a_rec, audio)
        B = v_rec.size(0)
        video = video.reshape(B, 8, -1, 768)
        v_rec = v_rec.reshape(B, 8, -1, 768)
        audio = audio.reshape(B, 8, -1, 128)        
        a_rec = a_rec.reshape(B, 8, -1, 128)

        loss_rec = self.reconstruction_loss(v_rec, video, a_rec, audio)
        L_adv_g, L_adv_d = self.adv_loss(v_rec, video, a_rec, audio)        
        lambda_c = 0.01
        lambda_rec = 1
        lambda_adv = 0.1
        lambda_gp=1
        gp = self.gradient_penalty_fn(video, v_rec, audio, a_rec)
        L = (L_c * lambda_c + loss_rec * lambda_rec + L_adv_g * lambda_adv + gp * lambda_gp), L_adv_d
    
        return v_rec_, a_rec_, L
    
    

    def forward_no_fusion(self, video, audio):
        #video = self._crop_face(video).to(self.device)
        v, a = self.encoding(video, audio)
        L_c = self.contrastive_loss(v, a)
        
        # v_msk, a_msk, v_vis, a_vis, m_v, m_a = self.complementary_masking(x_v, x_a)
        
        #v, a, v_a, a_v, i_v, i_a = self.cross_modality_fusion(v, a)
        
        v, a, v_rec, a_rec = self.decoding(v, a)

        # L = self.forward_loss(L_c, video, audio, v_rec, a_rec, i_v, i_a)

        return v_rec, a_rec, 0
    
    def forward(self, video, audio):
        #video = self._crop_face(video).to(self.device)
        v, a = self.encoding(video, audio)
        L_c = self.contrastive_loss(v, a)
        
        # v_msk, a_msk, v_vis, a_vis, m_v, m_a = self.complementary_masking(x_v, x_a)
        
        v, a, v_a, a_v, i_v, i_a = self.cross_modality_fusion(v, a)
        
        v, a, v_rec, a_rec = self.decoding(v, a)

        L = self.forward_loss(L_c, video, audio, v_rec, a_rec, i_v, i_a)

        return v_rec, a_rec, L
    
    def forward_with_mask(self, video, audio):
        #video = self._crop_face(video)
        v, a = self.encoding(video, audio)
        L_c = self.contrastive_loss(v, a)
        
        # v_msk, a_msk, v_vis, a_vis, m_v, m_a = self.complementary_masking(x_v, x_a)
        
        v, a, v_a, a_v, i_v, i_a = self.cross_modality_fusion(v, a)
        
        v, a, v_rec, a_rec = self.decoding(v, a)

        L = self.forward_loss(L_c, video, audio, v_rec, a_rec, i_v, i_a)

        return v_rec, a_rec, v_a, a_v, i_v, i_a, L
    
    def forward_2(self, video, audio):
        #video = self._crop_face(video).to(self.device)
        v, a = self.encoding(video, audio)
        L_c = self.contrastive_loss(v, a)
        
        # v_msk, a_msk, v_vis, a_vis, m_v, m_a = self.complementary_masking(x_v, x_a)
        
        v, a, v_a, a_v, i_v, i_a = self.cross_modality_fusion(v, a)
        
        v, a, v_rec, a_rec = self.decoding(v, a)

        (Lg1, Ld1) = self.forward_loss(L_c, video, audio, v_rec, a_rec, i_v, i_a)


        v, a = self.encoding(v_rec, a_rec.squeeze(1))
        L_c2 = self.contrastive_loss(v, a)
        
        # v_msk, a_msk, v_vis, a_vis, m_v, m_a = self.complementary_masking(x_v, x_a)
        
        v, a, v_a, a_v, i_v, i_a = self.cross_modality_fusion(v, a)
        
        v, a, v_rec, a_rec = self.decoding(v, a)

        (Lg2, Ld2) = self.forward_loss(L_c2, video, audio, v_rec, a_rec, i_v, i_a)

        return v_rec, a_rec, (Lg1+Lg2, Ld1+Ld2)
    
    def forward_loss(self, L_c, v, a, v_rec, a_rec, i_v, i_a, 
                     lambda_c = 0.01, lambda_rec = 1, lambda_adv = 0.1, lambda_gp=1):
        """
        using the AdamW optimizer [40] with a learning rate of 1.5e-4
        with a cosine decay [39]. The weights of the losses are as follows:
        λc = 0.01, λrec = 1.0, and λadv = 0.1
        """
        B = v_rec.size(0)
        
        v = v.reshape(B, 8, -1, 768)
        masked_v = v[:, i_v]

        v_rec = v_rec.reshape(B, 8, -1, 768)
        masked_v_rec = v_rec[:, i_v]

        a = a.reshape(B, 8, -1, 128)
        masked_a = a[:, i_a]
        
        a_rec = a_rec.reshape(B, 8, -1, 128)
        masked_a_rec = a_rec[:, i_a]

        loss_c = L_c
        loss_rec = self.reconstruction_loss(masked_v_rec, masked_v, masked_a_rec, masked_a)

        L_adv_g, L_adv_d = self.adv_loss(masked_v_rec, masked_v, masked_a_rec, masked_a)

        gp = self.gradient_penalty_fn(masked_v, masked_v_rec, masked_a, masked_a_rec)

        return (loss_c * lambda_c + loss_rec * lambda_rec + L_adv_g * lambda_adv + gp * lambda_gp), L_adv_d

    def gradient_penalty_fn(self, v, v_rec, a, a_rec) -> Tensor:
        B = v.size(0)
        
        alpha = torch.rand(B, 1, 1, 1).to(self.device)  # B x 1 x 1 x 1
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

    # loss functions

    def contrastive_loss(self, v, a, tau=0.1):
        """
        bi-directional audio-visual 
        contrastive loss

        v: video embeddings (B, 8, N_v, 768) # num_slices=8
        a: audio embeddings (B, 8, N_a, 768)
        tau: temperature
        
        Returns:
        L_c: contrastive loss (B,)
        """
        B, num_slices, N_v, _ = v.shape
        _, _, N_a, _ = a.shape
        
        # Mean latent vector across the patch dimension
        video_mean = v.mean(dim=2)  # Shape: (B, num_slices, 768)
        audio_mean = a.mean(dim=2)  # Shape: (B, num_slices, 768)
        
        # Reshape for computation
        video_mean = video_mean.view(B * num_slices, -1)  # Shape: (B * num_slices, 768)
        audio_mean = audio_mean.view(B * num_slices, -1)  # Shape: (B * num_slices, 768)
        
        # Compute similarity scores
        logits = torch.matmul(video_mean, audio_mean.T) / tau  # Shape: (B * num_slices, B * num_slices)
        
        # Create labels for contrastive loss
        labels = torch.arange(B * num_slices).to(video_mean.device)
        
        # Compute the contrastive loss
        loss = F.cross_entropy(logits, labels)
        return loss  # (B,)

    def reconstruction_loss(self, masked_v_rec, masked_v, masked_a_rec, masked_a):
        """
        before unpatchfy:
        (torch.Size([1, 1568, 1536]), torch.Size([1, 128, 768]))
        
        after unpatchfy
        v_rec, a_rec: (torch.Size([1, 3, 16, 224, 224]), torch.Size([1, 1, 768, 128]))
        = same with each v, a input shape
        """

        loss_fn = nn.MSELoss(reduction='mean')

        # Compute MSE loss for masked video and audio
        loss_v = loss_fn(masked_v_rec.squeeze(), masked_v.squeeze())
        loss_a = loss_fn(masked_a_rec.squeeze(), masked_a.squeeze())

        return loss_v + loss_a 
        
    def adv_loss(self, masked_v_rec, masked_v, masked_a_rec, masked_a):
        """
        Using Wasserstein GAN loss for adversarial training.
        """

        Dv_rec = self.v_discriminator(masked_v_rec)
        Dv = self.v_discriminator(masked_v)

        Da_rec = self.a_discriminator(masked_a_rec)
        Da = self.a_discriminator(masked_a)

        L_G_adv = -torch.mean(Dv_rec) - torch.mean(Da_rec)  # Generator loss
        L_D_adv = torch.mean(Dv_rec) - torch.mean(Dv) + torch.mean(Da_rec) - torch.mean(Da)  # Discriminator loss

        return L_G_adv, L_D_adv


    # utils 
    # _crop_face :
    def _crop_face(self, v):
        return self.marlin._crop_face(v)
    
    def augmentation(self, v):
        # augmentation horizonal flip and random drayscaling
        # 언급된 두 가지 증강

        return None