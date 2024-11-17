import torch
import yaml
from effortless_config import Config
from os import path, makedirs, system
from ddsp.model import DDSP
import soundfile as sf
from preprocess import get_files
from ddsp.scripted_models import ScriptDDSP
import traceback

torch.set_grad_enabled(False)

class args(Config):
    RUN = None
    WITH_DATA = False
    OUT_DIR = "export"
    REALTIME = False
    NN_TILDE = False
    DEBUG = False

def load_model(args, config):
    ddsp = DDSP(**config["model"])

    state = ddsp.state_dict()
    pretrained = torch.load(path.join(args.RUN, "state.pth"), map_location="cpu", weights_only=True)
    state.update(pretrained)
    ddsp.load_state_dict(state)
    ddsp.eval()

    return ddsp, path.basename(path.normpath(args.RUN))

def script_model(ddsp, config):
    if args.NN_TILDE:
        raise NotImplementedError("NN-Tilde not supported in export")
    else:
        return ScriptDDSP(ddsp, config, args.REALTIME)


def export_model(ddsp, name, config):
    scripted_model = torch.jit.script(script_model(ddsp, config))
    torch.jit.save( scripted_model, path.join(args.OUT_DIR, f"ddsp_{name}_pretrained.ts"))
    print(f"Exported model to {path.join(args.OUT_DIR, f'ddsp_{name}_pretrained.ts')}")

    impulse = ddsp.reverb.build_impulse().reshape(-1).numpy()
    impulse_path = path.join(args.OUT_DIR, f"ddsp_{name}_impulse.wav")

    sf.write(impulse_path, impulse, config["preprocess"]["sampling_rate"])
    print(f"Exported impulse response to {impulse_path}")


def export_config(name, config):
    config_path = path.join(args.OUT_DIR, f"ddsp_{name}_config.yaml")

    with open(config_path, "w") as config_out:
        yaml.safe_dump(config, config_out)


def export_data(args, config):
    makedirs(path.join(args.OUT_DIR, "data"), exist_ok=True)
    file_list = get_files(**config["data"])
    file_list = [str(f).replace(" ", "\\ ") for f in file_list]
    system(f"cp {' '.join(file_list)} {path.normpath(args.OUT_DIR)}/data/")
    print(f"Copied {len(file_list)} files to {path.normpath(args.OUT_DIR)}/data/")


def main():
    args.parse_args()
    makedirs(args.OUT_DIR, exist_ok=True)

    if args.RUN is None:
        raise ValueError("Please provide a RUN directory")

    with open(path.join(args.RUN, "config.yaml"), "r") as config:
        config = yaml.safe_load(config)

    ddsp, name = load_model(args, config)

    export_model(ddsp, name, config)
    export_config(name, config)

    if args.WITH_DATA:
        export_data(args, config)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Something went wrong: {e}")
        if args.DEBUG:  
             traceback.print_exc()  
