import os
import yaml
import random
import numpy as np
from pathlib import Path
import torch


# ==============================================================
#                       CONFIG LOADER
# ==============================================================
def load_config(config_path: str) -> dict:
    """Load a YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    print(f"✅ Loaded configuration from {config_path}")
    return config


# ==============================================================
#                          SEED SETUP
# ==============================================================
def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"🌱 Random seed set to: {seed}")


# ==============================================================
#                     CHECKPOINT MANAGEMENT
# ==============================================================

def save_checkpoint(
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    accuracy,
    loss,
    config,
    is_best=False,
    train_losses=None,
    val_losses=None,
    train_accuracies=None,
    val_accuracies=None,
):
    """Save model checkpoint, including training state."""
    train_losses = train_losses or []
    val_losses = val_losses or []
    train_accuracies = train_accuracies or []
    val_accuracies = val_accuracies or []

    # Ensure directory exists
    save_dir = Path(config["Data"]["root"]) / config["About"]["models_folder"]
    save_dir.mkdir(parents=True, exist_ok=True)

    # Define filenames
    checkpoint_name = config["About"]["checkpoint_file"].format(epoch, accuracy, loss)
    best_checkpoint_name = config["About"]["best_checkpoint_file"].format(epoch, accuracy, loss)

    checkpoint_path = save_dir / checkpoint_name

    # Construct checkpoint dictionary
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "accuracy": accuracy,
        "loss": loss,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies,
    }

    # Save current checkpoint
    # torch.save(checkpoint, checkpoint_path)
    # print(f"💾 Checkpoint saved: {checkpoint_path}")

    # Save best model if applicable
    if is_best:
        best_path = save_dir / best_checkpoint_name
        torch.save(checkpoint, best_path)
        print(f"🏆 Best model saved: {best_path}")

import os
import torch
from glob import glob

def load_checkpoint(config, model, optimizer=None, scheduler=None, scaler=None,
                    test=False, path=None):
    """
    Load model and optionally optimizer/scheduler/scaler states.

    Supports:
        - Direct path load
        - Best checkpoint load
        - Continue (latest) checkpoint load

    There is a manual for this function in /utils/manual.txt
    """

    checkpoints_dir = os.path.join(config['Data']['root'], config['About']['models_folder'])
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Step 1: Resolve which checkpoint to load
    ckpt_path = None

    if path:  # Manual path
        ckpt_path = path
    else:
        # Find all checkpoints
        all_ckpts = sorted(glob(os.path.join(checkpoints_dir, "checkpoint_*.pth")))
        best_ckpts = sorted(glob(os.path.join(checkpoints_dir, "best_*.pth")))

        preload_mode = config['About'].get('preload', 'none')

        if test or preload_mode == "best":
            if best_ckpts:
                ckpt_path = best_ckpts[-1]
        elif preload_mode == "cont":
            if all_ckpts:
                ckpt_path = all_ckpts[-1]

    if not ckpt_path:
        print("⚠️ No checkpoint found — starting fresh.")
        return 0, 0.0, [], [], [], []

    # Step 2: Load checkpoint
    print(f"🔄 Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cuda" if torch.cuda.is_available() else "cpu")

    # Step 3: Restore model
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)

    # Step 4: Restore optional training state (if available)
    if optimizer and 'optimizer_state_dict' in checkpoint:
        print("✅ The optimizer has bean loaded")
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in checkpoint:
        print("✅ The scheduler has bean loaded")
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if scaler and 'scaler_state_dict' in checkpoint:
        print("✅ The scaler has bean loaded")
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    # Step 5: Return info
    epoch = checkpoint.get('epoch', 0)
    accuracy = checkpoint.get('accuracy', 0.0)
    train_losses = checkpoint.get('train_losses', [])
    val_losses = checkpoint.get('val_losses', [])
    train_accuracies = checkpoint.get('train_accuracies', [])
    val_accuracies = checkpoint.get('val_accuracies', [])

    print(f"✅ Loaded epoch {epoch} (acc={accuracy:.4f}) from {os.path.basename(ckpt_path)}")

    return epoch, accuracy, train_losses, val_losses, train_accuracies, val_accuracies
