import yaml
import traceback
import pathlib
from os import makedirs, path

import librosa as li
import numpy as np
from tqdm import tqdm
import numpy as np
import torch
from effortless_config import Config

from ddsp.core import extract_loudness, extract_pitch

class args(Config):
    CONFIG = "config.yaml"
    DEBUG = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_files(data_location, extension, **kwargs):
    if not path.exists(data_location):
        raise FileNotFoundError(f"Failed to preprocess: {data_location} not found.")

    return list(pathlib.Path(data_location).rglob(f"*.{extension}"))


def preprocess(f, sampling_rate, block_size, signal_length, n_fft, oneshot, **kwargs):
    x, _ = li.load(f, sr=sampling_rate)
    N = (signal_length - len(x) % signal_length) % signal_length
    x = np.pad(x, (0, N))

    if oneshot:
        x = x[..., :signal_length]

    pitch = extract_pitch(x, sampling_rate, block_size, n_fft, str(device))
    loudness = extract_loudness(x, sampling_rate, block_size, n_fft)

    x = x.reshape(-1, signal_length)
    pitch = pitch.reshape(x.shape[0], -1)  # type: ignore
    loudness = loudness.reshape(x.shape[0], -1)

    print("shape of x:", x.shape)

    return x, pitch, loudness

def main():
    args.parse_args()

    with open(args.CONFIG, "r") as config:
        config = yaml.safe_load(config)

    files = get_files(**config["data"])
    pb = tqdm(files)

    signals = []
    pitchs = []
    loudness = []

    for f in pb:
        pb.set_description(str(f))
        x, p, l = preprocess(f, **config["preprocess"])
        signals.append(x)
        pitchs.append(p)
        loudness.append(l)

    signals = np.concatenate(signals, 0).astype(np.float32)
    pitchs = np.concatenate(pitchs, 0).astype(np.float32)
    loudness = np.concatenate(loudness, 0).astype(np.float32)

    out_dir = config["preprocess"]["out_dir"]
    makedirs(out_dir, exist_ok=True)

    np.save(path.join(out_dir, "signals.npy"), signals)
    np.save(path.join(out_dir, "pitchs.npy"), pitchs)
    np.save(path.join(out_dir, "loudness.npy"), loudness)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Something went wrong: {e}")
        if args.DEBUG:
            traceback.print_exc()
