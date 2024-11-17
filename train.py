import torch
import traceback
import yaml
from effortless_config import Config
from os import path
import os

from ddsp.model import DDSP
from ddsp.dataset import Dataset
from ddsp.core import  mean_std_loudness
from ddsp.trainer import Trainer
from ddsp.utils import verify_adjust_stop_lr, print_params

class args(Config):
    CONFIG = "config.yaml"
    NAME = "default"
    ROOT = "runs"
    STEPS = 500000
    BATCH = 16
    LR = 1e-3
    STOP_LR = 1e-4
    DECAY_OVER = 400000
    DEBUG = False

def main():
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
    
    print_params(config, args)
    
    trainer = Trainer(dataloader, model, config, args, device)
    trainer.train()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Something went wrong: {e}")
        if args.DEBUG:  
             traceback.print_exc()  

