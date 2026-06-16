import os
import yaml
import random
import numpy as np
from pathlib import Path
from glob import glob

import torch


# ==============================================================
#                       CONFIG LOADER
# ==============================================================
def load_config(config_path: str) -> dict:
    """Load a YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    print(f"Loaded config from: {config_path}")
    return config


# ==============================================================
#                          SEED SETUP
# ==============================================================
def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to: {seed}")


# ==============================================================
#                     CHECKPOINT MANAGEMENT
# ==============================================================
def save_checkpoint(
    model,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    val_acc: float,
    val_loss: float,
    config: dict,
    wandb_run_id: str,
    is_best: bool = False,
    train_losses: list = None,
    val_losses: list = None,
    train_accuracies: list = None,
    val_accuracies: list = None,
):
    """
    Save checkpoint every `save_every` epochs as epoch_XXX.pt.
    Always save best.pt when is_best=True.
    Both files store wandb_run_id for run resumption.
    """
    train_losses      = train_losses      or []
    val_losses        = val_losses        or []
    train_accuracies  = train_accuracies  or []
    val_accuracies    = val_accuracies    or []

    save_dir = Path(config["Data"]["root"]) / config["About"]["models_folder"]
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict":    scaler.state_dict(),
        "val_acc":              val_acc,
        "val_loss":             val_loss,
        "train_losses":         train_losses,
        "val_losses":           val_losses,
        "train_accuracies":     train_accuracies,
        "val_accuracies":       val_accuracies,
        "wandb_run_id":         wandb_run_id,
    }

    save_every = config["Modelling"].get("save_every", 5)
    if (epoch + 1) % save_every == 0:
        epoch_path = save_dir / f"epoch_{epoch+1:03d}.pt"
        torch.save(checkpoint, epoch_path)
        print(f"Checkpoint saved: {epoch_path}")

    if is_best:
        best_path = save_dir / "best.pt"
        torch.save(checkpoint, best_path)
        print(f"Best model saved: {best_path}  (acc={val_acc:.4f})")


def load_checkpoint(
    config: dict,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    checkpoint_path: str = None,
    test: bool = False,
):
    """
    Load a checkpoint. Resolution order:
        1. Explicit `checkpoint_path` argument  →  load that file
        2. test=True or preload='best'          →  load best.pt
        3. preload='cont'                       →  load latest epoch_XXX.pt
        4. Nothing found                        →  start fresh

    Returns
    -------
    epoch, val_acc, train_losses, val_losses, train_accuracies, val_accuracies, wandb_run_id
    """
    checkpoints_dir = Path(config["Data"]["root"]) / config["About"]["models_folder"]
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = None

    if checkpoint_path:
        ckpt_path = checkpoint_path
    else:
        preload = config["About"].get("preload", "from_start")
        best_path = checkpoints_dir / "best.pt"
        epoch_ckpts = sorted(glob(str(checkpoints_dir / "epoch_*.pt")))

        if test or preload == "best":
            if best_path.exists():
                ckpt_path = str(best_path)
        elif preload == "cont":
            if epoch_ckpts:
                ckpt_path = epoch_ckpts[-1]

    if not ckpt_path or not Path(ckpt_path).exists():
        print("No checkpoint found — starting fresh.")
        return 0, 0.0, [], [], [], [], None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    if optimizer and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("Optimizer restored.")
    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print("Scheduler restored.")
    if scaler and checkpoint.get("scaler_state_dict"):
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        print("Scaler restored.")

    epoch            = checkpoint.get("epoch", 0)
    val_acc          = checkpoint.get("val_acc", 0.0)
    train_losses     = checkpoint.get("train_losses", [])
    val_losses       = checkpoint.get("val_losses", [])
    train_accuracies = checkpoint.get("train_accuracies", [])
    val_accuracies   = checkpoint.get("val_accuracies", [])
    wandb_run_id     = checkpoint.get("wandb_run_id", None)

    print(f"Resumed from epoch {epoch}  (val_acc={val_acc:.4f})  wandb_run_id={wandb_run_id}")
    return epoch, val_acc, train_losses, val_losses, train_accuracies, val_accuracies, wandb_run_id