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
    print(f"{'='*60}")
    print(f"  Config loaded: {config_path}")
    print(f"{'='*60}")
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
    print(f"  Seed set to: {seed}")


# ==============================================================
#               HUGGINGFACE UPLOAD HELPER
# ==============================================================
def upload_to_huggingface(local_path: str, repo_id: str, repo_filename: str) -> None:
    """
    Upload a file to a HuggingFace repository.

    Parameters
    ----------
    local_path : str
        Local path to the file to upload.
    repo_id : str
        HuggingFace repository ID (e.g. 'username/repo-name').
    repo_filename : str
        Target filename inside the repository.
    """
    try:
        from huggingface_hub import HfApi, get_token
        token = get_token()
        if token is None:
            print(f"  [HuggingFace] WARNING: No token found. Skipping upload of {repo_filename}.")
            return
        api = HfApi()
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_filename,
            repo_id=repo_id,
            token=token,
        )
        print(f"  [HuggingFace] Uploaded: {repo_filename} -> {repo_id}")
    except Exception as e:
        print(f"  [HuggingFace] Upload failed for {repo_filename}: {e}")


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
    global_step: int,
    is_best: bool = False,
    train_losses: list = None,
    val_losses: list = None,
    train_accuracies: list = None,
    val_accuracies: list = None,
):
    """
    Save checkpoint every `save_every` epochs as epoch_XXX.pt.
    Always save best.pt when is_best=True.
    Both files store wandb_run_id and global_step for full run resumption.
    Uploads both files to HuggingFace if repo_id is set in config.
    """
    train_losses      = train_losses      or []
    val_losses        = val_losses        or []
    train_accuracies  = train_accuracies  or []
    val_accuracies    = val_accuracies    or []

    save_dir = Path(config["Data"]["root"]) / config["About"]["models_folder"]
    save_dir.mkdir(parents=True, exist_ok=True)

    repo_id = config["About"].get("repo_id", None)

    checkpoint = {
        "epoch":                epoch,
        "global_step":          global_step,
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
        epoch_filename = f"epoch_{epoch+1:03d}.pt"
        epoch_path = save_dir / epoch_filename
        torch.save(checkpoint, epoch_path)
        print(f"  [Checkpoint] Saved: {epoch_path}")
        if repo_id:
            upload_to_huggingface(str(epoch_path), repo_id, epoch_filename)

    if is_best:
        best_path = save_dir / "best.pt"
        torch.save(checkpoint, best_path)
        print(f"  [Checkpoint] Best model saved: {best_path}  (val_acc={val_acc:.4f}%)")
        if repo_id:
            upload_to_huggingface(str(best_path), repo_id, "best.pt")


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
        1. Explicit `checkpoint_path` argument  ->  load that file
        2. test=True or preload='best'          ->  load best.pt
        3. preload='cont'                       ->  load latest epoch_XXX.pt
        4. Nothing found                        ->  start fresh

    Returns
    -------
    epoch, val_acc, train_losses, val_losses, train_accuracies, val_accuracies,
    wandb_run_id, global_step
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
        print(f"  [Checkpoint] No checkpoint found — starting fresh.")
        return 0, 0.0, [], [], [], [], None, 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  [Checkpoint] Loading: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    if optimizer and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"  [Checkpoint] Optimizer restored.")
    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print(f"  [Checkpoint] Scheduler restored.")
    if scaler and checkpoint.get("scaler_state_dict"):
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        print(f"  [Checkpoint] Scaler restored.")

    epoch            = checkpoint.get("epoch", 0)
    val_acc          = checkpoint.get("val_acc", 0.0)
    train_losses     = checkpoint.get("train_losses", [])
    val_losses       = checkpoint.get("val_losses", [])
    train_accuracies = checkpoint.get("train_accuracies", [])
    val_accuracies   = checkpoint.get("val_accuracies", [])
    wandb_run_id     = checkpoint.get("wandb_run_id", None)
    global_step      = checkpoint.get("global_step", 0)

    print(f"  [Checkpoint] Resumed from epoch {epoch + 1}  |  val_acc={val_acc:.4f}%  |  global_step={global_step}  |  wandb_run_id={wandb_run_id}")
    return epoch + 1, val_acc, train_losses, val_losses, train_accuracies, val_accuracies, wandb_run_id, global_step