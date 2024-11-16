import torch
from torch.utils.tensorboard.writer import SummaryWriter
from ddsp.model import DDSP
from os import path
from ddsp.core import multiscale_fft, safe_log
from tqdm import tqdm
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np

class Trainer:
    def __init__(
        self,
        dataloader: torch.utils.data.DataLoader,
        model: DDSP,
        config: dict,
        args,
        device: torch.device,
    ):

        self.config = config
        self.model = model.to(device)
        self.device = device
        self.dataloader = dataloader
        self.path = path.join(args.ROOT, args.NAME)
        self.writer = SummaryWriter(self.path, flush_secs=20)
        self.mean_loudness = config["data"]["mean_loudness"]
        self.std_loudness = config["data"]["std_loudness"]
        self.best_loss = float("inf")
        self.mean_loss = 0
        self.n_element = 0
        self.step = 0
        self.epochs = int(np.ceil(args.STEPS / len(dataloader)))
        self.opt = torch.optim.Adam(model.parameters(), lr=args.LR)

        total_steps = args.DECAY_OVER / len(dataloader)
        gamma = (args.STOP_LR / args.LR) ** (1 / total_steps)
        self.lr_scheduler = ExponentialLR(self.opt, gamma=gamma)

    def train(self) -> None:
        for epoch in tqdm(range(self.epochs)):
            last_signal = None
            last_y = None

            for signal, pitch, loudness in self.dataloader:
                signal   =  signal.to(self.device)
                pitch    =  pitch.unsqueeze(-1).to(self.device)
                loudness =  loudness.unsqueeze(-1).to(self.device)
                loudness =  (loudness - self.mean_loudness) / self.std_loudness
                y        =  self.model(pitch, loudness).squeeze(-1)
                loss     =  self.spectrogram_loss(signal, y)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                self.writer.add_scalar("loss", loss.item(), self.step)

                self.step += 1
                self.n_element += 1
                self.mean_loss += (loss.item() - self.mean_loss) / self.n_element

                last_signal = signal
                last_y = y

            if not epoch % 10:
                self.evaluate_epoch(epoch)

                if last_signal is not None and last_y is not None:
                    self.evaluate_audio(last_signal, last_y)

                if self.mean_loss < self.best_loss:
                        self.best_loss = self.mean_loss

                self.mean_loss = 0
                self.n_element = 0


    def spectrogram_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        scales = self.config["train"]["scales"]
        overlap = self.config["train"]["overlap"]

        original_stft = multiscale_fft(x, scales, overlap)
        reconstruct_stft = multiscale_fft(y, scales, overlap)

        loss = torch.tensor(0.0).to(self.device)

        for s_x, s_y in zip(original_stft, reconstruct_stft):
            lin_loss = (s_x - s_y).abs().mean()
            log_loss = (safe_log(s_x) - safe_log(s_y)).abs().mean()
            loss = loss + lin_loss + log_loss

        return loss

    def evaluate_epoch(self, epoch: int) -> None:
        current_lr = self.opt.param_groups[0]['lr']

        self.writer.add_scalar("mean_loss", self.mean_loss, epoch)
        self.writer.add_scalar("lr", current_lr, epoch)
        self.writer.add_scalar("reverb_decay", self.model.reverb.decay.item(), epoch)
        self.writer.add_scalar("reverb_wet", self.model.reverb.wet.item(), epoch)

        if self.mean_loss < self.best_loss:
            torch.save( self.model.state_dict(), path.join(self.path, "state.pth"))

    def evaluate_audio(self, x: torch.Tensor, y: torch.Tensor) -> None:
        sr = self.config["preprocess"]["sampling_rate"]
        x_audio = x.reshape(-1).detach().cpu()
        y_audio = y.reshape(-1).detach().cpu()

        self.writer.add_audio("Last/Real", x_audio, self.step, sample_rate=sr)
        self.writer.add_audio("Last/Fake", y_audio, self.step, sample_rate=sr)

        if self.mean_loss < self.best_loss:
            self.writer.add_audio( f"Best/Real", x_audio, self.step, sample_rate=sr)
            self.writer.add_audio( f"Best/Fake", y_audio, self.step, sample_rate=sr)

            

