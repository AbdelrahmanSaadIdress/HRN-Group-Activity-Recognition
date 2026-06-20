import os
import numpy as np
import torch
import torch.nn as nn
from torch import amp
from tqdm import tqdm

import wandb
from sklearn.metrics import f1_score, confusion_matrix, classification_report

from .helper import load_checkpoint


class Tester:
    """
    Loads a checkpoint, resumes the same W&B run used during training,
    and logs all test metrics there.
    """

    def __init__(
        self,
        config: dict,
        model: nn.Module,
        dataloader,                  # test dataloader
        device: torch.device,
        checkpoint_path: str,        # required — explicit path to the model to test
        class_names: list = None,
        ignore_index: int = -100,
    ):
        self.config          = config
        self.model           = model
        self.dataloader      = dataloader
        self.device          = device
        self.checkpoint_path = checkpoint_path
        self.class_names     = class_names or []
        self.ignore_index    = ignore_index

        # ------------------------------------------------------------------
        # Load checkpoint — recover wandb_run_id and global_step
        # ------------------------------------------------------------------
        (_, _, _, _, _, _, wandb_run_id, global_step) = load_checkpoint(
            config,
            model,
            checkpoint_path=checkpoint_path,
            test=True,
        )
        self.global_step = global_step

        # ------------------------------------------------------------------
        # Resume the same W&B run
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

        # ------------------------------------------------------------------
        # Run test
        # ------------------------------------------------------------------
        print(f"\n{'='*60}")
        print(f"  Testing: {config['About']['name']}")
        print(f"  Device : {self.device}")
        print(f"{'='*60}\n")

        self._test()
        self.run.finish()

    # ------------------------------------------------------------------

    def _test(self):
        self.model.eval()
        total_loss, correct, total = 0.0, 0.0, 0.0
        y_true, y_pred = [], []
        torch.cuda.empty_cache()

        criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        pbar      = tqdm(self.dataloader, desc="  [Test]", ncols=80, leave=True)

        with torch.no_grad():
            for inputs, targets in pbar:
                inputs  = inputs.to(self.device)
                targets = targets.to(self.device)

                with amp.autocast("cuda", dtype=torch.float16):
                    outputs = self.model(inputs)
                    loss    = criterion(
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

        acc      = 100.0 * correct / max(total, 1)
        avg_loss = total_loss / len(self.dataloader)
        f1       = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        report   = classification_report(
            y_true, y_pred,
            target_names = self.class_names if self.class_names else None,
            output_dict  = True,
            zero_division = 0,
        )

        # ---- Console summary --------------------------------------------
        print(f"\n  {'─'*56}")
        print(f"  Test Results")
        print(f"  {'─'*56}")
        print(f"  {'Accuracy (%)':<25} {acc:>10.2f}")
        print(f"  {'Loss':<25} {avg_loss:>10.4f}")
        print(f"  {'F1 (weighted)':<25} {f1:>10.4f}")
        print(f"  {'─'*56}")
        print()
        print(classification_report(
            y_true, y_pred,
            target_names  = self.class_names if self.class_names else None,
            zero_division = 0,
        ))
        print(f"  {'─'*56}\n")

        # ---- W&B logging ------------------------------------------------
        wandb.log(
            {
                "test/accuracy":    acc,
                "test/loss":        avg_loss,
                "test/f1_weighted": f1,
                "test/confusion_matrix": wandb.plot.confusion_matrix(
                    probs      = None,
                    y_true     = list(y_true),
                    preds      = list(y_pred),
                    class_names= self.class_names if self.class_names else None,
                ),
            },
            step=self.global_step,
        )

        # Classification report as a W&B table
        rows = []
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                rows.append([
                    label,
                    round(metrics.get("precision", 0), 4),
                    round(metrics.get("recall",    0), 4),
                    round(metrics.get("f1-score",  0), 4),
                    int(metrics.get("support",     0)),
                ])
        table = wandb.Table(
            columns = ["class", "precision", "recall", "f1", "support"],
            data    = rows,
        )
        wandb.log({"test/classification_report": table}, step=self.global_step)

        return {"accuracy": acc, "loss": avg_loss, "f1": f1}