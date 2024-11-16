import torch
import yaml
from ddsp.model import DDSP
from effortless_config import Config
from os import path
import os
from preprocess import Dataset
from ddsp.core import  mean_std_loudness
from ddsp.trainer import Trainer
from ddsp.utils import verify_adjust_stop_lr

class args(Config):
    CONFIG = "config.yaml"
    NAME = "debug"
    ROOT = "runs"
    STEPS = 500000
    BATCH = 16
    LR = 1e-3
    STOP_LR = 1e-4
    DECAY_OVER = 400000

args.parse_args()

with open(args.CONFIG, "r") as config:
    config = yaml.safe_load(config)

device      = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model      = DDSP(**config["model"])
dataset    = Dataset(config["preprocess"]["out_dir"])
dataloader = torch.utils.data.DataLoader(dataset, args.BATCH, True, drop_last=True)
mean_loudness, std_loudness = mean_std_loudness(dataloader)
args.STOP_LR = verify_adjust_stop_lr(args.LR, args.STOP_LR)

config["data"]["mean_loudness"] = mean_loudness
config["data"]["std_loudness"] = std_loudness

if not path.exists(path.join(args.ROOT, args.NAME)):
    os.makedirs(path.join(args.ROOT, args.NAME), exist_ok=True)

with open(path.join(args.ROOT, args.NAME, "config.yaml"), "w") as out_config:
    yaml.safe_dump(config, out_config)

trainer = Trainer(dataloader, model, config, args, device)
trainer.train()

#opt = torch.optim.Adam(model.parameters(), lr=args.START_LR)
#
#schedule = get_scheduler(
#    len(dataloader),
#    args.START_LR,
#    args.STOP_LR,
#    args.DECAY_OVER,
#)
#
# scheduler = torch.optim.lr_scheduler.LambdaLR(opt, schedule)
#
#def spectrogram_loss(x, y, config):
#    scales = config["train"]["scales"]
#    overlap = config["train"]["overlap"]
#
#    original_stft = multiscale_fft( x, scales, overlap)
#    reconstruct_stft = multiscale_fft( y, scales, overlap)
#
#    loss = torch.tensor(0.0).to(device)
#
#    for s_x, s_y in zip(original_stft, reconstruct_stft):
#        lin_loss = (s_x - s_y).abs().mean()
#        log_loss = (safe_log(s_x) - safe_log(s_y)).abs().mean()
#        loss = loss + lin_loss + log_loss
#
#    return loss
#
#def evaluate_audio(x, y, config):
#        sr = config["preprocess"]["sampling_rate"]
#        x_audio = x.reshape(-1).detach().cpu()
#        y_audio = y.reshape(-1).detach().cpu()
#
#        writer.add_audio("Real",  x_audio, step, sample_rate=sr)
#        writer.add_audio("Real",  y_audio, step, sample_rate=sr)
#
#        eval_audio = torch.cat([x, y], -1).reshape(-1).detach().cpu().numpy()
#        sf.write( path.join(args.ROOT, args.NAME, f"eval_{epoch:06d}.wav"), eval_audio, sr)
#
#
#
#best_loss = float("inf")
#mean_loss = 0
#n_element = 0
#step = 0
#epochs = int(np.ceil(args.STEPS / len(dataloader)))
#
#for epoch in tqdm(range(epochs)):
#    epoch_loss = 0 
#    last_signal = None
#    last_y = None
#
#    for signal, pitch, loudness in dataloader:
#        signal = signal.to(device)
#        pitch = pitch.unsqueeze(-1).to(device)
#        loudness = loudness.unsqueeze(-1).to(device)
#
#        loudness = (loudness - mean_loudness) / std_loudness
#        y = model(pitch, loudness).squeeze(-1)
#       
#        loss = spectrogram_loss(signal, y, config)
#        opt.zero_grad()
#        loss.backward()
#        opt.step()
#
#        writer.add_scalar("loss", loss.item(), step)
#
#        step += 1
#
#        n_element += 1
#        mean_loss += (loss.item() - mean_loss) / n_element
#        epoch_loss += loss.item()
#        last_signal = signal
#        last_y = y
#
#        if step % 100 == 0:
#            writer.add_scalar("epoch_loss", epoch_loss, step)
#
#
#    if not epoch % 10:
#        writer.add_scalar("mean_loss", mean_loss, epoch)
#        writer.add_scalar("lr", schedule(epoch), epoch)
#        writer.add_scalar("reverb_decay", model.reverb.decay.item(), epoch)
#        writer.add_scalar("reverb_wet", model.reverb.wet.item(), epoch)
#
#        if mean_loss < best_loss:
#            best_loss = mean_loss
#            torch.save(
#                model.state_dict(),
#                path.join(args.ROOT, args.NAME, "state.pth"),
#            )
#
#        mean_loss = 0
#        n_element = 0
#
#        evaluate_audio(last_signal, last_y, config)
#       
