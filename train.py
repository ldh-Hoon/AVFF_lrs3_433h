import os
import gc
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.multiprocessing import spawn
from torch.cuda.amp import GradScaler, autocast  # FP16

# ---------- 설정 ----------
BATCH = 1
NUM_EPOCH = 5
LR = 1.5e-4
warmup_epochs = int(NUM_EPOCH * 0.1)
logging_step = 200
save_every = 1
model_name = "lrs3_5epoch"
root_path = "/lrs3/433h_data"
accum_steps = 2  # Gradient Accumulation

os.makedirs("logs", exist_ok=True)
os.makedirs("trained", exist_ok=True)
os.makedirs("reconstructions", exist_ok=True)
log_file_path = os.path.join("logs", f"{model_name}.jsonl")  # JSON Lines

# ---------- Train 함수 ----------
def train_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{rank}")

    # heavy import
    from dataset.lrs3 import LRS3Dataset
    from dataset.vis import save_reconstruction
    from AVFF.model.AVFF import AVFF

    # Dataset & Loader
    train_dataset = LRS3Dataset(root_path, split="train")
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH,
        sampler=train_sampler,
        num_workers=22,
        collate_fn=train_dataset.collate_fn,
        pin_memory=True
    )

    # Model
    model = AVFF().to(device)
    model = DDP(model, device_ids=[rank])

    # Discriminators DDP
    model.module.v_discriminator = DDP(model.module.v_discriminator.to(device), device_ids=[rank])
    model.module.a_discriminator = DDP(model.module.a_discriminator.to(device), device_ids=[rank])

    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    vd_optimizer = optim.AdamW(model.module.v_discriminator.parameters(), lr=LR)
    ad_optimizer = optim.AdamW(model.module.a_discriminator.parameters(), lr=LR)
    scheduler = LambdaLR(
        optimizer, 
        lr_lambda=lambda epoch: float(epoch+1)/warmup_epochs if epoch < warmup_epochs else 1.0
    )

    # FP16 scaler
    scaler = GradScaler()

    step_logs = []
    global_step = 0

    for epoch in range(NUM_EPOCH):
        model.train()
        train_sampler.set_epoch(epoch)
        pbar = tqdm(train_loader, desc=f'Rank {rank} | Epoch {epoch}/{NUM_EPOCH}', disable=(rank != 0))

        optimizer.zero_grad()
        vd_optimizer.zero_grad()
        ad_optimizer.zero_grad()

        for step, batch in enumerate(pbar):
            global_step += 1
            v, a = batch
            if v is None:
                continue
            video, audio = v.to(device), a.to(device)
            video = video.permute(0, 2, 1, 3, 4)

            # -------------------- Forward + Loss --------------------
            with autocast():  # FP16
                with model.no_sync() if ((step + 1) % accum_steps != 0) else torch.enable_grad():
                    if epoch < 1:  # 초반 1 epoch
                        v_rec, a_rec, _, _, i_v, i_a, (loss_g, loss_d, loss_c, loss_rec, loss_adv, v_encoded, a_encoded) = model.module.forward_with_nomask(video, audio)


                    else:
                        v_rec, a_rec, _, _, i_v, i_a, (loss_g, loss_d, loss_c, loss_rec, loss_adv, v_encoded, a_encoded) = \
                            model.module.forward_with_mask(video, audio)

            
            # -------------------- Backward --------------------
            scaler.scale((loss_g + loss_d) / accum_steps).backward()

            # -------------------- Optimizer Step --------------------
            if (step + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.step(vd_optimizer)
                scaler.step(ad_optimizer)
                scaler.update()

                optimizer.zero_grad()
                vd_optimizer.zero_grad()
                ad_optimizer.zero_grad()

            # -------------------- Logging --------------------
            if rank == 0 and (global_step % logging_step == 0 or global_step == 1):
                step_log = {
                    "epoch": epoch,
                    "step": global_step,
                    "loss_g": loss_g.item(),
                    "loss_d": loss_d.item(),
                    "loss_c": loss_c.item(),
                    "loss_rec": loss_rec.item(),
                    "loss_adv": loss_adv.item(),
                }
                step_logs.append(step_log)
                pbar.set_postfix({
                    'loss_g': f'{loss_g.item():.4f}',
                    'loss_d': f'{loss_d.item():.4f}',
                    'loss_c': f'{loss_c.item():.4f}',
                    'loss_rec': f'{loss_rec.item():.4f}',
                    'loss_adv': f'{loss_adv.item():.4f}',
                })
                with open(log_file_path, "a") as f:
                    f.write(json.dumps(step_log) + "\n")

            # 메모리 관리
            gc.collect()
            torch.cuda.empty_cache()

        # -------------------- Epoch 저장 --------------------
        if rank == 0:
            torch.save(model.state_dict(), f'trained/{model_name}_epoch{epoch}.pth')

            if (epoch % save_every == 0):
                sample_video, sample_audio = next(iter(train_loader))
                sample_video = sample_video.to(device).permute(0, 2, 1, 3, 4)
                sample_audio = sample_audio.to(device)
                with torch.no_grad():
                    v_rec, a_rec, *_ = model.module.forward_with_mask(sample_video, sample_audio)
                save_dir = f"reconstructions/epoch_{epoch}"
                save_reconstruction(sample_video, sample_audio, v_rec, a_rec, save_dir, step_or_epoch=epoch)

        scheduler.step()

    torch.distributed.destroy_process_group()


# ---------- Entry point ----------
if __name__ == '__main__':
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    world_size = 4
    spawn(train_ddp, args=(world_size,), nprocs=world_size, join=True)
