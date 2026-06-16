from typing import Literal
import os

import albumentations as A
from albumentations.pytorch import ToTensorV2

from AnnotationsExtraction import AnnotationPreparer
from .activities import group_activity_labels, person_activity_labels
from .group_activity_dataSet import Group_Activity_DataSet, group_collate_fn
from .person_activity_dataSet import Person_Activity_DataSet, person_collate_fn

def get_transform(state: Literal["train", "val", "test"]):
    if state == "test":
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    if state == "val":
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    if state == "train":
        return A.Compose([
            A.Resize(224, 224),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7)),
                A.ColorJitter(brightness=0.2),
                A.RandomBrightnessContrast(),
                A.GaussNoise()
            ], p=0.90),
            A.OneOf([
                A.HorizontalFlip(),
                A.VerticalFlip(),
            ], p=0.05),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])


def get_dataloader(config: dict, state: Literal["train", "val", "test"]):
    # B1-NoRelations
    if config["About"]["name"] == "B1-NoRelations":
        if config["About"]["level"] == "person":
            video_path = config["Data"]["frames_annots_path"]
            annot_path = os.path.join(config["Data"]["annotations_path"], "data.pkl")
            seq = False
            split = config["Modelling"]["data_splits"][state]
            labels = person_activity_labels
            weights_path = os.path.join(config["Data"]["weights_path"], config["Data"]["person_weights"], f"{state}_weights.pkl")
            transform = get_transform(state)
            repo_id = config['About']['repo_id']

            dataset = Person_Activity_DataSet(
                videos_path=video_path, 
                annot_path=annot_path, 
                seq=seq, 
                split=split, 
                labels=labels, 
                transform=transform,
                weights_path=weights_path,
                huggingface_repo_id=repo_id
            )

            return dataset, person_collate_fn
        
        elif config["About"]["level"] == "group":
            video_path = config["Data"]["frames_annots_path"]
            annot_path = os.path.join(config["Data"]["annotations_path"], "data.pkl")
            seq = False
            sort=True
            split = config["Modelling"]["data_splits"][state]
            labels = group_activity_labels
            weights_path = os.path.join(config["Data"]["weights_path"], config["Data"]["group_weights"], f"{state}_weights.pkl")
            transform = get_transform(state)
            repo_id = config['About']['repo_id']

            dataset = Group_Activity_DataSet(
                videos_path=video_path, 
                annot_path=annot_path, 
                seq=seq, 
                sort=sort,
                split=split, 
                labels=labels, 
                transform=transform,
                weights_path=weights_path,
                huggingface_repo_id=repo_id
            )

            return dataset, group_collate_fn

    if config["About"]["name"] in ["RCRG-1R-1C", "RCRG-1R-1C-untuned", "RCRG-2R-11C_conc", "RCRG-2R-11C"]:
        video_path = config["Data"]["frames_annots_path"]
        annot_path = os.path.join(config["Data"]["annotations_path"], "data.pkl")
        seq = False
        sort=True
        split = config["Modelling"]["data_splits"][state]
        labels = group_activity_labels
        weights_path = os.path.join(config["Data"]["weights_path"], config["Data"]["group_weights"], f"{state}_weights.pkl")
        transform = get_transform(state)
        repo_id = config['About']['repo_id']

        dataset = Group_Activity_DataSet(
            videos_path=video_path, 
            annot_path=annot_path, 
            seq=seq, 
            sort=sort,
            split=split, 
            labels=labels, 
            transform=transform,
            weights_path=weights_path,
            huggingface_repo_id=repo_id
        )

        return dataset, group_collate_fn
