import os
import pickle
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset
from huggingface_hub import HfApi, get_token


class Group_Activity_DataSet(Dataset):
    def __init__(self, 
                videos_path: str, annot_path: str, 
                seq: bool = False, sort: bool=False , split: list = [],
                labels: dict = {}, transform=None,
                weights_path: str = None, huggingface_repo_id:str = None):
        self.videos_path = videos_path
        self.annot_path = annot_path
        self.seq = seq
        self.sort = sort
        self.split = split
        self.labels = labels
        self.transform = transform
        self.weights_path = weights_path
        self.huggingface_repo_id = huggingface_repo_id
        self.weights = True if (weights_path is not None and os.path.exists(weights_path)) else None
        if self.weights is None:
            self.weights_count = {label: 0.0 for label in self.labels.values()}
        self.videos_annot = self.load_data()
        self.samples = self.create_samples()

        self.save_weights()

    def load_data(self):
        with open(self.annot_path, "rb") as f:
            data = pickle.load(f)
        return data

    def create_samples(self):
        samples = []
        for video_id in self.split:
            video_id = str(video_id)
            if video_id not in self.videos_annot:
                raise ValueError(f"Video ID {video_id} not found in annotations.")
            video_data = self.videos_annot[video_id]
            for seq_id, seq_data in video_data.items():
                group_label = seq_data['category']
                if group_label not in self.labels:
                    raise ValueError(f"Group activity label {group_label} not found in labels dictionary.")
                if self.weights is None:
                    self.weights_count[self.labels[group_label]] += 1.0
                if self.seq:
                    frames_pathes, players_boxes = [], []
                for frame_id, frame_data in seq_data['frames_boxes_dct'].items():
                    if self.seq == True:
                        frames_pathes.append(
                            os.path.join(self.videos_path, video_id, seq_id, f"{frame_id}.jpg")
                        )
                        players_boxes.append(frame_data)
                    else:
                        samples.append({
                            "frames_pathes": [os.path.join(self.videos_path, video_id, seq_id, f"{frame_id}.jpg")],
                            "players_boxes": [frame_data],
                            "group_label": self.labels[group_label]
                        })
                if self.seq:
                    samples.append({
                        "frames_pathes":frames_pathes,
                        "players_boxes":players_boxes,
                        "group_label":self.labels[group_label]
                    })
        return samples

    def save_weights(self):
        os.makedirs(os.path.dirname(self.weights_path), exist_ok=True)
    
        if self.weights is None:
            with open(self.weights_path, "wb") as f:
                pickle.dump(self.weights_count, f)
    
        if self.huggingface_repo_id is not None:
            api = HfApi()
            token = get_token()
    
            if token is None:
                raise ValueError(
                    "Huggingface token not found. Please run `huggingface-cli login`."
                )
    
            api.upload_file(
                path_or_fileobj=self.weights_path,
                path_in_repo=os.path.basename(self.weights_path),
                repo_id=self.huggingface_repo_id,
                token=token
            )
    
    def get_weights(self):
        if self.weights is not None:
            with open(self.weights_path, "rb") as f:
                weights = pickle.load(f)
            return weights
        else:
            return self.weights_count

    def get_players_crops(self, img, players_info):
        if self.sort:
            players_info = sorted(
                players_info,
                key=lambda p: (p.xMin + p.xMax) / 2
            )
        players_crops = []
        for player_info in players_info:
            x1, y1, x2, y2 = player_info.xMin, player_info.yMin, player_info.xMax, player_info.yMax
            # add a check to ensure the bounding box is within the image dimensions
            w, h = img.size
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            player_crop = img.crop((x1, y1, x2, y2))
            if self.transform is not None:
                player_crop = self.transform(image=np.array(player_crop))['image']   
            players_crops.append(player_crop)

        return torch.stack(players_crops) if players_crops else torch.empty(0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = torch.tensor(sample["group_label"], dtype=torch.long)

        frames = []

        for frame_path, players_info in zip(
            sample["frames_pathes"],
            sample["players_boxes"]
        ):
            img = Image.open(frame_path).convert("RGB")
            players = self.get_players_crops(img, players_info)
            frames.append(players)

        # if not sequence → wrap in list
        if not self.seq:
            frames = frames[:1]

        return frames, label


def group_collate_fn(batch):
    MAX_PLAYERS = 12
    sequences, labels = zip(*batch)

    B = len(sequences)
    T_max = max(len(seq) for seq in sequences)

    C, H, W = sequences[0][0].shape[1:]

    batch_tensor = torch.zeros(
        B, T_max, MAX_PLAYERS, C, H, W
    )

    for i, seq in enumerate(sequences):
        for t, frame in enumerate(seq):

            n = frame.shape[0]

            # truncate players
            frame = frame[:MAX_PLAYERS]

            batch_tensor[i, t, :frame.shape[0]] = frame

    labels = torch.tensor(labels, dtype=torch.long)

    return batch_tensor.squeeze(1), labels



# If seq=False      -------->      [B, 12, 3, H, W]
# If seq=True       -------->      [B, T_max, 12, 3, H, W]