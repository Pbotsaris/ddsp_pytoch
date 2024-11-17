def verify_adjust_stop_lr(lr: float, stop_lr: float) -> float:
    if lr < stop_lr:
        stop_lr = max(lr - 1e-1, 1e-7)  # Ensure STOP_LR doesn't go too low
        print(f"Warning: STOP_LR ({stop_lr}) is greater than LR ({lr}). Adjusting STOP_LR to {stop_lr}")
    return stop_lr


def print_params(config, args):
    print("\n\nArguments from 'args':")
    for key, value in vars(args).items():
        print(f"\t{key}: {value}")

    print("\nModel parameters:")
    for key, value in config["model"].items():
        print(f"\t{key}: {value}")
    
    print("\nTraining parameters:")
    if "train" in config:
        for key, value in config["train"].items():
            print(f"\t{key}: {value}")
    else:
        print("No training parameters found in the YAML file.")



