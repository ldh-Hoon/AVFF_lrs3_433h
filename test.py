import os
import torch
from dataset.lrs3 import LRS3Dataset
from dataset.vis import save_reconstruction
from AVFF.model.AVFF import AVFF

# ---------- 설정 ----------
BATCH = 2
root_path = "/lrs3/433h_data"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = "trained/final_ema_433h_epoch17.pth"  # 불러올 모델

os.makedirs("reconstructions", exist_ok=True)

# ---------- Dataset ----------
dataset = LRS3Dataset(root_path, split="train")
loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH, shuffle=True, collate_fn=dataset.collate_fn)

# ---------- Model ----------
model = AVFF().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ---------- Inference ----------
with torch.no_grad():
    for step, batch in enumerate(loader):
        v, a = batch
        if v is None:
            continue

        video, audio = v.to(device), a.to(device)
        video = video.permute(0, 2, 1, 3, 4)  # B, C, T, H, W
        
        v_encoded, a_encoded = model.encoding(video, audio)

        v_fused, a_fused, v_a, a_v, i_v, i_a = model.cross_modality_fusion(v_encoded, a_encoded)

        v_rec, a_rec = model.decoding(v_fused, a_fused)

        # 저장
        save_dir = f"reconstructions/sample"
        save_reconstruction(video, audio, v_rec, a_rec, save_dir, step_or_epoch=step)

        # 배치 하나만 테스트
        break
