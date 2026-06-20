import os
import pickle
from PIL import Image
from typing import List
import torch
import numpy as np
from torch.utils.data import Dataset

from AnnotationsExtraction.BoxInfo import BoxInfo


class Person_Activity_DataSet(Dataset):
    def __init__(self,
                videos_path: str,
                annot_path: str,
                seq: bool = False,
                split: list = [],
                labels: dict = {},
                transform=None,
                weights_path: str = None,
                huggingface_repo_id: str = None,
                upload_weights: bool = False):
        """
        Parameters
        ----------
        upload_weights : bool
            Whether to upload the weights file to HuggingFace.
            Should be True only for the train split.
        """
        self.videos_path         = videos_path
        self.annot_path          = annot_path
        self.seq                 = seq
        self.split               = split
        self.labels              = labels
        self.transform           = transform
        self.weights_path        = weights_path
        self.huggingface_repo_id = huggingface_repo_id
        self.upload_weights      = upload_weights

        # Always start fresh — weights are always recalculated
        self.weights_count = {label: 0.0 for label in self.labels.values()}

        self.videos_annot = self._load_data()
        self.samples      = self._create_samples()
        self._save_weights()

    # ------------------------------------------------------------------

    def _load_data(self):
        from AnnotationsExtraction.BoxInfo import BoxInfo

        class FixedUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if name == 'BoxInfo':
                    return BoxInfo
                return super().find_class(module, name)

        with open(self.annot_path, "rb") as f:
            return FixedUnpickler(f).load()

    def _create_samples(self):
        samples = []
        for video_id in self.split:
            video_id = str(video_id)
            if video_id not in self.videos_annot:
                raise ValueError(f"Video ID {video_id} not found in annotations.")
            video_data = self.videos_annot[video_id]

            for seq_id, seq_data in video_data.items():
                if self.seq:
                    frames_pathes, players_boxes = [], []

                for frame_id, frame_data in seq_data['frames_boxes_dct'].items():
                    # Always count weights from scratch
                    for player_info in frame_data:
                        player_activity_label = self.labels.get(player_info.category)
                        if player_activity_label is not None:
                            self.weights_count[player_activity_label] += 1.0

                    if self.seq:
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
                        "frames_pathes": frames_pathes,
                        "players_boxes": players_boxes,
                    })
        return samples

    def _save_weights(self):
        """Always save freshly recalculated weights. Upload to HF only if flagged."""
        if self.weights_path is None:
            return

        os.makedirs(os.path.dirname(self.weights_path), exist_ok=True)

        with open(self.weights_path, "wb") as f:
            pickle.dump(self.weights_count, f)

        if self.upload_weights and self.huggingface_repo_id is not None:
            try:
                from huggingface_hub import HfApi, get_token
                token = get_token()
                if token is None:
                    print("  [HuggingFace] WARNING: No token found. Skipping weights upload.")
                    return
                api = HfApi()
                api.upload_file(
                    path_or_fileobj = self.weights_path,
                    path_in_repo    = f"persons_{os.path.basename(self.weights_path)}",
                    repo_id         = self.huggingface_repo_id,
                    token           = token,
                )
                print(f"  [HuggingFace] Weights uploaded: persons_{os.path.basename(self.weights_path)}")
            except Exception as e:
                print(f"  [HuggingFace] Weights upload failed: {e}")

    def get_weights(self) -> dict:
        """Return class counts for loss weighting. Always freshly calculated."""
        return self.weights_count

    def get_players_crops(self, img, players_info):
        players_crops  = []
        players_labels = []

        for player_info in players_info:
            x1, y1, x2, y2       = player_info.xMin, player_info.yMin, player_info.xMax, player_info.yMax
            player_activity_label = self.labels.get(player_info.category)

            w, h = img.size
            x1 = max(0, x1);  y1 = max(0, y1)
            x2 = min(w, x2);  y2 = min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            player_crop = img.crop((x1, y1, x2, y2))
            if self.transform is not None:
                player_crop = self.transform(image=np.array(player_crop))['image']
            players_crops.append(player_crop)
            players_labels.append(player_activity_label)

        if not players_crops:
            return torch.empty(0, 3, 224, 224), torch.empty(0, dtype=torch.long)

        return torch.stack(players_crops), torch.tensor(players_labels, dtype=torch.long)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        frames: List[torch.Tensor] = []
        labels: List[torch.Tensor] = []

        for frame_path, players_info in zip(sample["frames_pathes"], sample["players_boxes"]):
            img                    = Image.open(frame_path).convert("RGB")
            players, player_labels = self.get_players_crops(img, players_info)
            frames.append(players)
            labels.append(player_labels)

        if not self.seq:
            frames = [frames[-1]]
            labels = [labels[-1]]

        return frames, labels


# ==============================================================
#  Collate function
# ==============================================================

def person_collate_fn(batch, ignore_index=-100):
    MAX_PLAYERS = 12
    sequences, labels = zip(*batch)

    B     = len(sequences)
    T_max = max(len(seq) for seq in sequences)
    C, H, W = sequences[0][0].shape[1:]

    video  = torch.zeros(B, T_max, MAX_PLAYERS, C, H, W)
    target = torch.full((B, T_max, MAX_PLAYERS), ignore_index, dtype=torch.long)

    for i, (seq, seq_labels) in enumerate(zip(sequences, labels)):
        for t, (frame, frame_labels) in enumerate(zip(seq, seq_labels)):
            n = min(frame.shape[0], MAX_PLAYERS)
            video[i, t, :n]  = frame[:n]
            target[i, t, :n] = frame_labels[:n]

    video  = video.squeeze(1)
    target = target.squeeze(1)

    # For seq=True, return only the last frame's labels
    if T_max > 1:
        return video, target[:, -1]

    return video, target

# Shape reference:
# seq=False -> video: [B, 12, 3, H, W]  |  labels: [B, 12]
# seq=True  -> video: [B, T, 12, 3, H, W]  |  labels: [B, 12]  (last frame)