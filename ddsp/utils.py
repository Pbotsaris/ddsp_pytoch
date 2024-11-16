def verify_adjust_stop_lr(lr: float, stop_lr: float) -> float:
    if lr < stop_lr:
        stop_lr = max(lr - 1e-1, 1e-7)  # Ensure STOP_LR doesn't go too low
        print(f"Warning: STOP_LR ({stop_lr}) is greater than LR ({lr}). Adjusting STOP_LR to {stop_lr}")
    return stop_lr
