import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import amp
from tqdm import tqdm

import wandb
from sklearn.metrics import f1_score, confusion_matrix, classification_report

from .helper import save_checkpoint, load_checkpoint


class Trainer:
    """
    Trainer with full W&B integration.

    Features
    --------
    - Single global_step axis for ALL W&B logging (no step collision)
    - Resumes the same W&B run when a checkpoint is available
    - global_step is saved/restored from checkpoint for seamless resumption
    - Per-batch and per-epoch metrics logged to W&B
    - Gradient histograms via wandb.watch
    - Weighted cross-entropy (wired from dataset.get_weights())
    - Gradient clipping (grad_clip in config)
    - Periodic checkpointing every save_every epochs
    - Saves best.pt whenever validation accuracy improves
    - Confusion matrix logged as a W&B plot each epoch
    - Clean professional console output
    """

    def __init__(
        self,
        config: dict,
        model: nn.Module,
        optimizer,
        scaler,
        dataloaders: list,               # [train_loader, val_loader]
        device: torch.device,
        scheduler=None,
        scheduler_type: str = "per epoch",   # "per epoch" | "per batch"
        class_names: list = None,
        checkpoint_path: str = None,         # explicit resume path
        ignore_index: int = -100,
    ):
        self.config         = config
        self.model          = model
        self.optimizer      = optimizer
        self.scaler         = scaler
        self.dataloaders    = dataloaders
        self.device         = device
        self.scheduler      = scheduler
        self.scheduler_type = scheduler_type
        self.class_names    = class_names or []
        self.ignore_index   = ignore_index
        self.grad_clip      = config["Modelling"].get("grad_clip", None)

        # ------------------------------------------------------------------
        # Build weighted criterion from train dataset
        # ------------------------------------------------------------------
        self.criterion = self._build_criterion()

        # ------------------------------------------------------------------
        # Load checkpoint (if any) — also recovers wandb_run_id & global_step
        # ------------------------------------------------------------------
        (
            self.start_epoch,
            self.best_acc,
            self.train_losses,
            self.val_losses,
            self.train_accuracies,
            self.val_accuracies,
            wandb_run_id,
            self.global_step,
        ) = load_checkpoint(
            config, model, optimizer, scheduler, scaler,
            checkpoint_path=checkpoint_path,
        )

        # ------------------------------------------------------------------
        # W&B init — resume same run if we have a run_id
        # ------------------------------------------------------------------
        wandb_kwargs = dict(
            project = config["About"]["project_name"],
            name    = config["About"]["name"],
            config  = config,
        )
        if wandb_run_id:
            wandb_kwargs["id"]     = wandb_run_id
            wandb_kwargs["resume"] = "must"

        self.run = wandb.init(**wandb_kwargs)
        self.wandb_run_id = self.run.id

        wandb.watch(self.model, log="all", log_freq=50)

        # ------------------------------------------------------------------
        # Train
        # ------------------------------------------------------------------
        self._print_training_header()
        self._train()
        self.run.finish()

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    def _build_criterion(self):
        """Build CrossEntropyLoss, optionally with class weights from train dataset."""
        use_weights  = self.config["Modelling"].get("weighted_loss", False)
        weight_tensor = None

        if use_weights:
            train_dataset = self.dataloaders[0].dataset
            if hasattr(train_dataset, "get_weights"):
                counts      = train_dataset.get_weights()       # dict {label_int: count}
                num_classes = len(counts)
                counts_arr  = np.array([counts[i] for i in range(num_classes)], dtype=np.float32)

                # Inverse-frequency weights, normalised
                weights       = 1.0 / (counts_arr + 1e-6)
                weights       = weights / weights.sum() * num_classes
                weight_tensor = torch.tensor(weights, dtype=torch.float32).to(self.device)

                # ---- Clear class-weight table ----------------------------
                print(f"\n  {'─'*52}")
                print(f"  {'Class':<20} {'Count':>10} {'Weight':>10}")
                print(f"  {'─'*52}")
                for i in range(num_classes):
                    name = self.class_names[i] if i < len(self.class_names) else str(i)
                    print(f"  {name:<20} {int(counts_arr[i]):>10} {weights[i]:>10.4f}")
                print(f"  {'─'*52}\n")

        return nn.CrossEntropyLoss(
            weight       = weight_tensor,
            ignore_index = self.ignore_index,
        )

    def _print_training_header(self):
        total_epochs = self.config["Modelling"]["epochs"]
        print(f"\n{'='*60}")
        print(f"  Training: {self.config['About']['name']}")
        print(f"  Epochs  : {self.start_epoch + 1} -> {total_epochs}")
        print(f"  Device  : {self.device}")
        print(f"  W&B Run : {self.wandb_run_id}")
        print(f"{'='*60}\n")

    def _print_epoch_summary(self, epoch, total_epochs, train_loss, train_acc,
                              val_loss, val_acc, f1, lr, is_best):
        marker = "  ** NEW BEST **" if is_best else ""
        print(f"\n  {'─'*56}")
        print(f"  Epoch   : {epoch+1:03d} / {total_epochs:03d}{marker}")
        print(f"  {'─'*56}")
        print(f"  {'Metric':<18} {'Train':>12} {'Val':>12}")
        print(f"  {'─'*56}")
        print(f"  {'Loss':<18} {train_loss:>12.4f} {val_loss:>12.4f}")
        print(f"  {'Accuracy (%)':<18} {train_acc:>12.2f} {val_acc:>12.2f}")
        print(f"  {'F1 (weighted)':<18} {'':>12} {f1:>12.4f}")
        print(f"  {'LR':<18} {lr:>12.2e}")
        print(f"  {'─'*56}\n")

    # ------------------------------------------------------------------
    #  Train / Val loops
    # ------------------------------------------------------------------

    def _train_one_epoch(self, epoch: int):
        self.model.train()
        total_loss, correct, total = 0.0, 0.0, 0.0
        torch.cuda.empty_cache()

        train_loader = self.dataloaders[0]
        pbar = tqdm(
            train_loader,
            desc    = f"  [Train] Epoch {epoch+1:03d}",
            ncols   = 80,
            leave   = True,
        )

        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs  = inputs.to(self.device)
            targets = targets.to(self.device)

            with amp.autocast("cuda", dtype=torch.float16):
                outputs = self.model(inputs)
                loss    = self.criterion(
                    outputs.view(-1, outputs.size(-1)),
                    targets.view(-1),
                )

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()

            if self.grad_clip:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler and self.scheduler_type == "per batch":
                self.scheduler.step()

            total_loss += loss.item()
            predicted   = outputs.argmax(-1)

            mask     = targets.view(-1) != self.ignore_index
            correct += predicted.view(-1)[mask].eq(targets.view(-1)[mask]).sum().item()
            total   += mask.sum().item()

            batch_acc = 100.0 * correct / max(total, 1)

            # All W&B logging on one consistent global_step axis
            wandb.log(
                {
                    "train/batch_loss": loss.item(),
                    "train/batch_acc":  batch_acc,
                },
                step=self.global_step,
            )

            self.global_step += 1

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc =f"{batch_acc:.2f}%",
            )

        epoch_loss = total_loss / len(train_loader)
        epoch_acc  = 100.0 * correct / max(total, 1)
        return epoch_loss, epoch_acc

    def _val_one_epoch(self, epoch: int):
        self.model.eval()
        total_loss, correct, total = 0.0, 0.0, 0.0
        y_true, y_pred = [], []
        torch.cuda.empty_cache()

        val_loader = self.dataloaders[1]
        pbar = tqdm(
            val_loader,
            desc  = f"  [Val]   Epoch {epoch+1:03d}",
            ncols = 80,
            leave = True,
        )

        with torch.no_grad():
            for inputs, targets in pbar:
                inputs  = inputs.to(self.device)
                targets = targets.to(self.device)

                with amp.autocast("cuda", dtype=torch.float16):
                    outputs = self.model(inputs)
                    loss    = self.criterion(
                        outputs.view(-1, outputs.size(-1)),
                        targets.view(-1),
                    )

                total_loss += loss.item()
                predicted   = outputs.argmax(-1)

                mask = targets.view(-1) != self.ignore_index
                t    = targets.view(-1)[mask].cpu().numpy()
                p    = predicted.view(-1)[mask].cpu().numpy()

                y_true.extend(t)
                y_pred.extend(p)
                correct += (p == t).sum()
                total   += mask.sum().item()

                pbar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_loss = total_loss / len(val_loader)
        epoch_acc  = 100.0 * correct / max(total, 1)
        f1         = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        return epoch_loss, epoch_acc, f1, y_true, y_pred

    # ------------------------------------------------------------------
    #  Main training loop
    # ------------------------------------------------------------------

    def _train(self):
        total_epochs = self.config["Modelling"]["epochs"]

        for epoch in range(self.start_epoch, total_epochs):

            train_loss, train_acc              = self._train_one_epoch(epoch)
            val_loss, val_acc, f1, y_true, y_pred = self._val_one_epoch(epoch)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)

            if self.scheduler and self.scheduler_type == "per epoch":
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]
            is_best    = val_acc > self.best_acc

            # ---- W&B epoch-level logging (same global_step axis) --------
            wandb.log(
                {
                    "epoch":           epoch + 1,
                    "train/loss":      train_loss,
                    "train/acc":       train_acc,
                    "val/loss":        val_loss,
                    "val/acc":         val_acc,
                    "val/f1_weighted": f1,
                    "lr":              current_lr,
                    "best_val_acc":    max(self.best_acc, val_acc),
                    "val/confusion_matrix": wandb.plot.confusion_matrix(
                        probs      = None,
                        y_true     = y_true,
                        preds      = y_pred,
                        class_names= self.class_names if self.class_names else None,
                    ),
                },
                step=self.global_step,
            )

            # ---- Checkpoint ---------------------------------------------
            save_checkpoint(
                model            = self.model,
                optimizer        = self.optimizer,
                scheduler        = self.scheduler,
                scaler           = self.scaler,
                epoch            = epoch,
                val_acc          = val_acc,
                val_loss         = val_loss,
                config           = self.config,
                wandb_run_id     = self.wandb_run_id,
                global_step      = self.global_step,
                is_best          = is_best,
                train_losses     = self.train_losses,
                val_losses       = self.val_losses,
                train_accuracies = self.train_accuracies,
                val_accuracies   = self.val_accuracies,
            )

            if is_best:
                self.best_acc = val_acc

            self._print_epoch_summary(
                epoch, total_epochs,
                train_loss, train_acc,
                val_loss, val_acc,
                f1, current_lr, is_best,
            )

        print(f"\n{'='*60}")
        print(f"  Training complete.  Best val_acc = {self.best_acc:.2f}%")
        print(f"{'='*60}\n")