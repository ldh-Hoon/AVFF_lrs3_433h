import torch
import torch.nn as nn
import torch.nn.functional as F

from .AVFF import *
from torchaudio.compliance import kaldi

FIX = False

class classifier(nn.Module):
    def __init__(self, model_name, target_dim=10, 
                marlin_path = "/home/ldh/avsr/AvHubert/av_hubert/avhubert/AVFF/pretrained_models/MARLIN/marlin_vit_base_ytf.full.pt",
                audioMAE_path = '/home/ldh/avsr/AvHubert/av_hubert/avhubert/AVFF/pretrained_models/AudioMAE/pretrained.pth'):
        super(classifier, self).__init__()

        self.model_trained = AVFF(marlin_path=marlin_path, audioMAE_path=audioMAE_path)
        self.model_trained.load_state_dict(torch.load(f'{model_name}.pth'))
        self.target_dim = target_dim


        
        self.init()

    def init(self):        
        for param in self.model_trained.parameters():
            param.requires_grad = False
        
        self.model_trained.eval()


    def pred_forward(self, v, a):
        if not v == None:
            B, F_v, C, H, W = v.shape
            v = v.permute(0, 2, 1, 3, 4)
            pred_modal = 'audio'

            v = self.model_trained.marlin.extract_features(v)

            b, _, _ = v.shape
            v = v.reshape(b, self.model_trained.num_slices, -1, v.shape[-1]) 
            x_v = v
        else:
            b, F_a, M = a.shape
            pred_modal = 'video'
        
            a = a.unsqueeze(1) # to (B, C=1, F, L)

            a = torch.nn.functional.pad(a, (0, 0, 0, 256))
            a = self.model_trained.audio_mae.forward_encoder_no_mask(a.float())
            a = a[:, 1:, :]
            a = a[:, :384, :]

            a = a.reshape(b, self.model_trained.num_slices, -1, a.shape[-1])
            x_a = a


        if pred_modal == 'audio':
            a = self.model_trained.V2A(x_v)

            B, S, N_a, F_a = a.shape
            a = a.reshape(B, S*N_a, F_a)
            a = torch.nn.functional.pad(a, (0, 0, 0, 128)) # 128
            
            a, _, _ = self.model_trained.audio_mae.forward_decoder_no_mask(a) # pred, _, _
            a_rec = self.model_trained.audio_mae.unpatchify(a)
            a_rec = a_rec[:, :, :768, :].squeeze()

            return a_rec

        if pred_modal == 'video':
            v = self.model_trained.A2V(x_a)
            
            v = self.model_trained.marlin.enc_dec_proj(v)

            B, S, N_v, F_v = v.shape
            v = v.reshape(B, S*N_v, F_v)

            v = self.model_trained.video_decoder.forward_no_mask(v)
            v_rec = self.model_trained.video_decoder.unpatch_to_img(v)

            return v_rec.permute(0, 2, 1, 3, 4)
        
    
    def get_features(self, v, a):
        B, F_v, C, H, W = v.shape
        _, F_a, M = a.shape

        v = v.permute(0, 2, 1, 3, 4)
        #v = self.model_trained._crop_face(v).to(self.model_trained.device)
        x_v, x_a = self.model_trained.encoding(v, a)

        v_a = self.model_trained.A2V(x_a)
        a_v = self.model_trained.V2A(x_v)

        return x_v, x_a, v_a, a_v


    def forward(self, v, a, mode=0):
        B, F_v, C, H, W = v.shape
        _, F_a, M = a.shape
        v_s, a_s = self.get_slices(v, a)
        
        n_s = v_s.size(1) // B

        out_v = []
        out_a = []

        for i in range(B):
            x_v, x_a = self.forward_slice(v_s[i], a_s[i])

            out_v.append(x_v)
            out_a.append(x_a)

        out_v = torch.stack(out_v).view(B, -1, 768)
        out_a = torch.stack(out_a).view(B, -1, 768)

        return out_v, out_a

    def forward_slice(self, v, a):
        B = v.size(0)
        v = v.permute(0, 2, 1, 3, 4)
        #v = self.model_trained._crop_face(v).to(self.model_trained.device)
        x_v, x_a = self.model_trained.encoding(v, a)

        v_a = self.model_trained.A2V(x_a)
        a_v = self.model_trained.V2A(x_v)

        v_diff = x_v-v_a
        a_diff = x_a-a_v
    
                
        return v_diff, a_diff
        
    def get_slices(self, video, audio):
        B, F, C, H, W = video.shape
        fps = 5
        n_v = video.size(1)
        n_a = audio.size(1)

        video_slice_length = int(3.2 * fps)
        audio_slice_length = int(768)

        video_slices = []
        audio_slices = []

        for b in range(B):
            for start in range(0, n_v, int(3.2 * fps)):
                if start + video_slice_length <= n_v:
                    video_slices.append(video[b, start:start + video_slice_length])
                else:
                    pad_size = (start + video_slice_length) - n_v
                    padded_slice = torch.cat((video[b, start:], torch.zeros((pad_size, C, H, W), device=video.device)))
                    video_slices.append(padded_slice)

        v_s = torch.stack(video_slices).view(B, -1, video_slice_length, C, H, W)

        for b in range(B):
            for start in range(0, n_a, int(768)):
                if start + audio_slice_length <= n_a:
                    audio_slices.append(audio[b, start:start + audio_slice_length])
                else:
                    pad_size = (start + audio_slice_length) - n_a
                    padded_slice = torch.cat((audio[b, start:], torch.zeros((pad_size, 128), device=audio.device)))
                    audio_slices.append(padded_slice)

        a_s = torch.stack(audio_slices).view(B, -1, audio_slice_length, 128)

        return v_s, a_s

    def min_max_norm(self, tensor):
        min_val = tensor.min(dim=1, keepdim=True)[0]
        max_val = tensor.max(dim=1, keepdim=True)[0]
        normalized_tensor = (tensor - min_val) / (max_val - min_val + 0.001)
        return normalized_tensor

    def reduction(self, x):
        x = torch.mean(x, dim=-1)

        return x
    
    def interpolation(self, x):
        x = F.interpolate(x.unsqueeze(1), size=self.target_dim , mode='linear', align_corners=False).squeeze(1)

        return x
    
    def mel_preprocessing_new(self, audio_data, sr=16000):

        if audio_data.dim() == 1:
            audio_data = audio_data.unsqueeze(0) 
        fbank = kaldi.fbank(audio_data, htk_compat=True, sample_frequency=sr, use_energy=False, 
        window_type='hanning', num_mel_bins=128, dither=0.0,frame_shift=4 ,frame_length= 16)
        n_frames = fbank.shape[0]
        TARGET_LEN = 768
        p = TARGET_LEN - n_frames
        # cut and pad
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
    
    def data_preprocessing(self, data, type):
        # no batch dim
        if type == 'audio':
            audio = self.mel_preprocessing_new(data)
            return audio
        
        if type == "video":
            video = data

            if video.dtype == torch.uint8:
                video = video.float() / 255.0

            step = float(16) / 5
            if step.is_integer():
                step = int(step)
                idxs = slice(None, None, step)
            else:
                num_frames = max(int(len(video) / step), 1)
                idxs = torch.arange(num_frames, dtype=torch.float32) * step
                idxs = idxs.floor().to(torch.int64)
            video = video[idxs]

            video = F.interpolate(video, size=(224, 224), mode='bilinear', align_corners=False)
            if video.size(0) < 16:
                padding_size = 16 - video.size(0)
                padding = torch.zeros(padding_size, video.size(1), 224, 224, device=video.device)
                video = torch.cat((video, padding), dim=0)  # dim=0으로 변경하여 프레임 축에 패딩 추가
            else:
                video = video[:16, :, :, :]
            
            return video