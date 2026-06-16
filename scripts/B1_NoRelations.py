"""
scripts/B1_NoRelations.py
=========================
Four entry-point functions for the B1-NoRelations two-stage model.
Each function is fully self-contained: it builds the model, dataloaders,
optimizer, scheduler, scaler, and hands everything off to Trainer / Tester.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.single_frame_models import b1_norelations_stage1, b1_norelations_stage2
from DataSet.GetDataSet import get_dataloader
from DataSet.activities import person_activity_clases, group_activity_clases
from utils import Trainer, Tester, set_seed, load_checkpoint


# ======================================================================
#  Shared helpers
# ======================================================================

def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_loader(dataset, collate_fn, config, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size  = config["Modelling"]["batch_size"],
        shuffle     = shuffle,
        num_workers = config["Modelling"].get("num_workers", 4),
        pin_memory  = True,
        collate_fn  = collate_fn,
    )


def _make_optimizer(model: nn.Module, config: dict):
    return optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = config["Modelling"]["lr"],
        weight_decay = config["Modelling"].get("weight_decay", 0.0),
    )


def _make_scheduler(optimizer, config: dict):
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode    = "min",
        factor  = 0.5,
        patience= config["Modelling"].get("lr_patience", 3),
    )


# ======================================================================
#  Stage 1 — person activity classification
# ======================================================================

def train_stage_one(config: dict, checkpoint_path: str = None):
    """
    Fine-tune ResNet50 for per-player action classification (9 classes).

    Input tensor  : (B, P, 3, 224, 224)
    Output logits : (B*P, 9)
    Loss          : CrossEntropy over all (B*P) positions, ignoring pad=-100
    """
    set_seed(config["Modelling"]["seed"])
    device = _device()

    # ---- Datasets -------------------------------------------------------
    train_dataset, collate_fn = get_dataloader(config, "train")
    val_dataset,   _          = get_dataloader(config, "val")

    train_loader = _make_loader(train_dataset, collate_fn, config, shuffle=True)
    val_loader   = _make_loader(val_dataset,   collate_fn, config, shuffle=False)

    # ---- Model ----------------------------------------------------------
    model = b1_norelations_stage1(
        num_classes=config["Modelling"]["num_classes"]
    ).to(device)

    # ---- Optimiser / scheduler / scaler ---------------------------------
    optimizer = _make_optimizer(model, config)
    scheduler = _make_scheduler(optimizer, config)
    scaler    = torch.amp.GradScaler("cuda")
    # scaler = torch.cuda.amp.GradScaler()

    # ---- Train ----------------------------------------------------------
    Trainer(
        config          = config,
        model           = model,
        optimizer       = optimizer,
        scaler          = scaler,
        dataloaders     = [train_loader, val_loader],
        device          = device,
        scheduler       = scheduler,
        scheduler_type  = "per epoch",
        class_names     = person_activity_clases,
        checkpoint_path = checkpoint_path,
        ignore_index    = -100,
    )


def test_stage_one(config: dict, checkpoint_path: str):
    """
    Evaluate a trained Stage 1 model on the test split.
    Results are logged to the same W&B run stored in the checkpoint.
    """
    set_seed(config["Modelling"]["seed"])
    device = _device()

    test_dataset, collate_fn = get_dataloader(config, "test")
    test_loader = _make_loader(test_dataset, collate_fn, config, shuffle=False)

    model = b1_norelations_stage1(
        num_classes=config["Modelling"]["num_classes"]
    ).to(device)

    Tester(
        config          = config,
        model           = model,
        dataloader      = test_loader,
        device          = device,
        checkpoint_path = checkpoint_path,
        class_names     = person_activity_clases,
        ignore_index    = -100,
    )


# ======================================================================
#  Stage 2 — group activity classification
# ======================================================================

def train_stage_two(
    config: dict,
    stage1_checkpoint: str,
    checkpoint_path: str = None,
):
    """
    Train the group activity head on top of a frozen Stage 1 backbone.

    Architecture recap:
        backbone (frozen) → 2048-d → dense(128) per player
        → team1_mean ⊕ team2_mean → 256-d → FC → 8 classes

    Parameters
    ----------
    stage1_checkpoint : str
        Path to the Stage 1 best.pt file.
    checkpoint_path : str, optional
        Path to a Stage 2 checkpoint to resume from.
    """
    set_seed(config["Modelling"]["seed"])
    device = _device()

    # ---- Datasets -------------------------------------------------------
    train_dataset, collate_fn = get_dataloader(config, "train")
    val_dataset,   _          = get_dataloader(config, "val")

    train_loader = _make_loader(train_dataset, collate_fn, config, shuffle=True)
    val_loader   = _make_loader(val_dataset,   collate_fn, config, shuffle=False)

    # ---- Build Stage 1, load its weights --------------------------------
    stage1_model = b1_norelations_stage1(
        num_classes=config["Modelling"].get("stage1_num_classes", 9)
    )
    # Load stage 1 weights into the stage1 shell (no optimizer needed)
    load_checkpoint(config, stage1_model, checkpoint_path=stage1_checkpoint, test=True)
    stage1_model.to(device)

    # ---- Build Stage 2 --------------------------------------------------
    model = b1_norelations_stage2(
        stage1_model=stage1_model,
        num_classes=config["Modelling"]["num_classes"],
    ).to(device)

    # ---- Freeze backbone if configured ----------------------------------
    if config["Modelling"].get("freeze_backbone", True):
        for param in model.stage1.parameters():
            param.requires_grad = False
        print("Stage 1 backbone frozen.")

    # ---- Optimiser / scheduler / scaler ---------------------------------
    # optimizer only sees parameters that require grad
    optimizer = _make_optimizer(model, config)
    scheduler = _make_scheduler(optimizer, config)
    scaler    = torch.amp.GradScaler("cuda")

    # ---- Train ----------------------------------------------------------
    Trainer(
        config          = config,
        model           = model,
        optimizer       = optimizer,
        scaler          = scaler,
        dataloaders     = [train_loader, val_loader],
        device          = device,
        scheduler       = scheduler,
        scheduler_type  = "per epoch",
        class_names     = group_activity_clases,
        checkpoint_path = checkpoint_path,
        ignore_index    = -100,
    )


def test_stage_two(config: dict, checkpoint_path: str):
    """
    Evaluate a trained Stage 2 model on the test split.
    Results are logged to the same W&B run stored in the checkpoint.

    Note: stage1 weights are loaded from inside the Stage 2 checkpoint —
    no separate --stage1_checkpoint needed for testing.
    """
    set_seed(config["Modelling"]["seed"])
    device = _device()

    test_dataset, collate_fn = get_dataloader(config, "test")
    test_loader = _make_loader(test_dataset, collate_fn, config, shuffle=False)

    # Build a dummy stage1 shell so we can construct stage2
    stage1_shell = b1_norelations_stage1(
        num_classes=config["Modelling"].get("stage1_num_classes", 9)
    )
    model = b1_norelations_stage2(
        stage1_model=stage1_shell,
        num_classes=config["Modelling"]["num_classes"],
    ).to(device)

    Tester(
        config          = config,
        model           = model,
        dataloader      = test_loader,
        device          = device,
        checkpoint_path = checkpoint_path,
        class_names     = group_activity_clases,
        ignore_index    = -100,
    )