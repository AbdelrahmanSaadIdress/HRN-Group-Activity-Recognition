import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.single_frame_models import b1_norelations_stage1, RCRG_3R_421C_conc_stage2
from DataSet.GetDataSet import get_dataloader
from DataSet.activities import person_activity_clases, group_activity_clases
from utils import Trainer, Tester, set_seed, load_checkpoint

from .B1_NoRelations import _device, _make_loader, _make_optimizer, _make_scheduler

# ======================================================================
#  Stage 2 — group activity classification
# ======================================================================

def train_stage_two(
    config: dict,
    stage1_checkpoint: str,
    checkpoint_path: str = None,
):
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
    model = RCRG_3R_421C_conc_stage2(
        stage1_model=stage1_model,
        in_channels = config["Modelling"]["in_channels"],
        hidden_channels=config["Modelling"]["hidden_channels"],
        output_channels=config["Modelling"]["output_channels"],
        num_classes=config["Modelling"]["num_classes"]
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
    model = RCRG_3R_421C_conc_stage2(
        stage1_model=stage1_shell,
        in_channels = config["Modelling"]["in_channels"],
        hidden_channels=config["Modelling"]["hidden_channels"],
        output_channels=config["Modelling"]["output_channels"],
        num_classes=config["Modelling"]["num_classes"]
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