import torch
import nn_tilde
from ddsp.model import DDSP
from ddsp.resampler import Resampler

def loudness_stats(c):
    return (c["data"]["mean_loudness"], c["data"]["std_loudness"])

class ScriptDDSP(torch.nn.Module):
    def __init__(self, ddsp: DDSP, config: dict, realtime: bool):
        super().__init__()
        self.ddsp = ddsp
        self.ddsp.gru.flatten_parameters()

        mean_loudness, std_loudness = loudness_stats(config)

        self.register_buffer("mean_loudness", torch.tensor(mean_loudness))
        self.register_buffer("std_loudness", torch.tensor(std_loudness))
        self.realtime = realtime

    def forward(self, pitch, loudness):
        loudness = (loudness - self.mean_loudness) / self.std_loudness

        if self.realtime:
            pitch = pitch[:, :: self.ddsp.block_size]
            loudness = loudness[:, :: self.ddsp.block_size]
            return self.ddsp.realtime_forward(pitch, loudness)

        return self.ddsp(pitch, loudness)

class ScriptNNTildeDDSP(nn_tilde.Module):
    def __init__(self, ddsp: DDSP, config: dict, inference_sr: int | None = None):
        super().__init__()
        self.ddsp = ddsp
        self.ddsp.gru.flatten_parameters()
        self.resampler = None

        mean_loudness, std_loudness = loudness_stats(config)
        model_sr = config["model"]["sampling_rate"]

        self.register_buffer("mean_loudness", torch.tensor(mean_loudness))
        self.register_buffer("std_loudness", torch.tensor(std_loudness))

        if inference_sr is not None and model_sr != inference_sr:
            self.resampler = Resampler(model_sr, inference_sr)

        else:
            print(f"Using model sampling rate: {model_sr}hz for inference")

        model_block_size = config["model"]["block_size"]

        self.register_method(
            "forward",
            in_channels=2,
            in_ratio=model_block_size,
            out_channels=1,
            out_ratio=1,
            input_labels=[f"(signal) pitch", f"(signal) loudness"],
            output_labels=[f"(signal) audio out"],
            test_buffer_size=model_block_size,
        )

    @torch.jit.export
    def forward(self, x):
        # in shape of x: (batch_size, (pitch, loudness), block_size)
        bach_size = x.shape[0]
        # only one batch (1, (pitch, loudness), block_size)
        # TODO: handle batches more gracefully
        x = x[:1]

        # pitch (1, 1, block_size), loudness (1, 1, block_size)
        pitch, loudness = torch.split(x, 1, dim=1)

        # to (1, block_size)
        pitch = pitch.squeeze(1)
        loudness = loudness.squeeze(1)

        # as column vector (1, block_size, 1)
        pitch = pitch.reshape(1, -1, 1)
        loudness = loudness.reshape(1, -1, 1)

        loudness = (loudness - self.mean_loudness) / self.std_loudness

        # out shape: (batch, block_size,  1) 
        out = self.ddsp.realtime_forward(pitch, loudness)

        # out shape to (1, 1, block_size)
        out = out.reshape(1, 1, -1)

        # repeat the output to match the batch size
        # TODO: handle batches more gracefully
        out = out.repeat(bach_size, 1, 1)

        if self.resampler is None:
            return out

        return self.resampler(out)

