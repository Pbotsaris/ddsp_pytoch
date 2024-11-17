import torch
import torch.nn as nn
import numpy as np
import cached_conv as cc
from ddsp.core import kaiser_filter

class Resampler(nn.Module):
    def __init__(self, original_sr: int, target_sr: int):
        super().__init__()

        self.original_sr = original_sr
        self.target_sr = target_sr

        if original_sr == target_sr:
            raise ValueError(self.error("same_sr"))

        ratio = target_sr / original_sr

        if int(ratio) != ratio:
            raise ValueError(self.error("multiple"))

        if ratio % 2 and cc.USE_BUFFER_CONV:
            raise ValueError(self.error("power_of_two"))

        ratio = int(ratio)
        wc = np.pi / ratio

        filter = kaiser_filter(wc, 140)
        filter = torch.from_numpy(filter).float()
        padding = cc.get_padding(len(filter), ratio)

        self.downsampler = cc.Conv1d( 1, 1, len(filter), stride=ratio, padding=padding, bias=False)
        self.downsampler.weight.data.copy_(filter.reshape(1, 1, -1))

        pad = len(filter) % ratio
        filter = nn.functional.pad(filter, (0, pad))
        filter = filter.reshape(-1, ratio).permute(1, 0)

        pad = (filter.shape[-1] + 1) % 2
        filter = nn.functional.pad(filter, (pad, 0)).unsqueeze(1)
        padding = cc.get_padding(filter.shape[-1])

        self.upsampler = cc.Conv1d( 1, ratio, filter.shape[-1], stride=1, pdding=padding, bias=False)
        self.ratio = ratio

    def downsample(self, x: torch.Tensor):
        x_down = x.reshape(-1, 1, x.shape[-1])
        x_down = self.downsampler(x_down)

        # TODO: check shapes
        return x_down.reshape(x.shape[0], x.shape[1], -1)

    def upsample(self, x: torch.Tensor):
        x_up = x.reshape(-1, 1, x.shape[-1])
        x_up = self.upsampler(x_up)  # bach, 2, t
        x_up = x_up.permute(0, 2, 1).reshape(x_up.shape[0], -1).unsqueeze(1)
        return x_up.reshape(x.shape[0], x.shape[1], -1)


    def error(self, type: str):
        but = f"but got {self.original_sr} == {self.target_sr}"

        if type == "same_sr":
            return f"Resampler: target & original sr must be different {but}"

        elif type == "multiple":
            return f"target_sr should be a multiple of original_sr {but}"

        else:
            return f"Resampler: ratio must be a power of two {but}"
