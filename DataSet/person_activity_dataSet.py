import os
import pickle
from random import sample
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset
from huggingface_hub import HfApi, get_token

class Person_Activity_DataSet(Dataset):
    def __init__(self, 
                videos_path: str, annot_path: str, 
                seq: bool = False, split: list = [],
                labels: dict = {}, transform=None,
                weights_path: str = None, huggingface_repo_id:str = None):
        
        self.videos_path = videos_path
        self.annot_path = annot_path
        self.seq = seq
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
                # if self.weights is None:
                #     self.weights_count[self.labels[group_label]] += 1.0
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
                        })
                if self.seq:
                    samples.append({
                        "frames_pathes":frames_pathes,
                        "players_boxes":players_boxes,
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
        players_crops = []
        players_labels = []
        for player_info in players_info:
            x1, y1, x2, y2 = player_info.xMin, player_info.yMin, player_info.xMax, player_info.yMax
            player_activity_label = self.labels.get(player_info.category)
            if self.weights is None and player_activity_label is not None:
                self.weights_count[player_activity_label] += 1.0
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
            players_labels.append(player_activity_label)

        return torch.stack(players_crops) if players_crops else torch.empty(0), torch.tensor(players_labels, dtype=torch.long)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        frames: torch.List[torch.Tensor] = []   # each: [N, 3, H, W]
        labels: torch.List[torch.Tensor] = []  # each: [N]

        for frame_path, players_info in zip(
            sample["frames_pathes"],
            sample["players_boxes"]
        ):
            img = Image.open(frame_path).convert("RGB")
            players, player_labels = self.get_players_crops(img, players_info)
            frames.append(players)
            labels.append(player_labels)

        # if not sequence → wrap in list
        if not self.seq:
            frames = [frames[-1]]
            labels = [labels[-1]]


        return frames, labels

def person_collate_fn(batch, ignore_index=-100):
    MAX_PLAYERS = 12

    sequences, labels = zip(*batch)

    B = len(sequences)
    T_max = max(len(seq) for seq in sequences)

    C, H, W = sequences[0][0].shape[1:]

    video = torch.zeros(B, T_max, MAX_PLAYERS, C, H, W)
    target = torch.full((B, T_max, MAX_PLAYERS), ignore_index, dtype=torch.long)

    for i, (seq, seq_labels) in enumerate(zip(sequences, labels)):
        for t, (frame, frame_labels) in enumerate(zip(seq, seq_labels)):

            n = min(frame.shape[0], MAX_PLAYERS)

            video[i, t, :n] = frame[:n]
            target[i, t, :n] = frame_labels[:n]

    
    video = video.squeeze(1)
    target = target.squeeze(1)

    if video.shape[1] == 9:
        return video, target[:,-1]
    
    return video, target

# ✅ seq=False
    # video: [B, 12, 3, H, W]
    # labels: [B, 12]
# ✅ seq=True
    # video: [B, T, 12, 3, H, W]
    # labels: [B, 12]   # last frame only