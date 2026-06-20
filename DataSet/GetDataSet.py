from typing import Literal
import os

import albumentations as A
from albumentations.pytorch import ToTensorV2

from .activities import group_activity_labels, person_activity_labels
from .group_activity_dataSet import Group_Activity_DataSet, group_collate_fn
from .person_activity_dataSet import Person_Activity_DataSet, person_collate_fn


SINGLE_FRAME_GROUP_MODELS = {
    "B1-NoRelations",
    "RCRG-1R-1C", "RCRG-1R-1C-untuned",
    "RCRG-2R-11C-conc", "RCRG-2R-11C",
    "RCRG-2R-21C-conc", "RCRG-2R-21C",
    "RCRG-3R-421C-conc", "RCRG-3R-421C",
}


def get_transform(state: Literal["train", "val", "test"]):
    if state in ("val", "test"):
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    # train
    return A.Compose([
        A.Resize(224, 224),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7)),
            A.ColorJitter(brightness=0.2),
            A.RandomBrightnessContrast(),
            A.GaussNoise(),
        ], p=0.90),
        A.OneOf([
            A.HorizontalFlip(),
            A.VerticalFlip(),
        ], p=0.05),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_dataloader(config: dict, state: Literal["train", "val", "test"]):
    """
    Returns (dataset, collate_fn) for the requested split.

    Weights are always recalculated fresh.
    HuggingFace upload of the weights file is only triggered for the train split.
    """
    is_train   = (state == "train")
    video_path = config["Data"]["frames_annots_path"]
    annot_path = os.path.join(config["Data"]["annotations_path"], "data.pkl")
    split      = config["Modelling"]["data_splits"][state]
    transform  = get_transform(state)
    repo_id    = config["About"].get("repo_id", None)
    model_name = config["About"]["name"]

    def _group_weights_path():
        return os.path.join(
            config["Data"]["weights_path"],
            config["Data"]["group_weights"],
            f"{state}_weights.pkl",
        )

    def _make_group_dataset(seq: bool, sort: bool):
        return Group_Activity_DataSet(
            videos_path         = video_path,
            annot_path          = annot_path,
            seq                 = seq,
            sort                = sort,
            split               = split,
            labels              = group_activity_labels,
            transform           = transform,
            weights_path        = _group_weights_path(),
            huggingface_repo_id = repo_id,
            upload_weights      = is_train,
        )

    # ── Temporal models (seq=True) ────────────────────────────────────────
    if config["About"].get("temporal", False):
        return _make_group_dataset(seq=True, sort=True), group_collate_fn

    # ── B1-NoRelations — person level ─────────────────────────────────────
    if model_name == "B1-NoRelations" and config["About"].get("level") == "person":
        weights_path = os.path.join(
            config["Data"]["weights_path"],
            config["Data"]["person_weights"],
            f"{state}_weights.pkl",
        )
        dataset = Person_Activity_DataSet(
            videos_path         = video_path,
            annot_path          = annot_path,
            seq                 = False,
            split               = split,
            labels              = person_activity_labels,
            transform           = transform,
            weights_path        = weights_path,
            huggingface_repo_id = repo_id,
            upload_weights      = is_train,
        )
        return dataset, person_collate_fn

    # ── B1-NoRelations (group level) + all RCRG single-frame models ───────
    if model_name in SINGLE_FRAME_GROUP_MODELS:
        return _make_group_dataset(seq=False, sort=True), group_collate_fn

    raise ValueError(f"Unknown model name in config: '{model_name}'")