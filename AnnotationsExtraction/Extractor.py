from typing import Dict, List
from pathlib import Path
from .BoxInfo import BoxInfo


class Extractor:
    """
    Provides class methods to extract bounding box and frame-level annotations
    from annotation text files.

    Usage
    -----
    Extractor.extract_crops_annot(txt_path, tgt_frame)
    Extractor.extract_frame_annot(txt_path)
    """

    @classmethod
    def extract_crops_annot(cls, txt_path: str | Path, tgt_frame: int) -> Dict[str, List['BoxInfo']]:
        """
        Extracts bounding boxes for frames near the target frame.

        Parameters
        ----------
        txt_path : str | Path
            Path to the annotation text file.
        tgt_frame : int
            Target frame number around which to collect annotations.

        Returns
        -------
        Dict[str, List[BoxInfo]]
            A dictionary where each key is a frameId and each value is a list of BoxInfo objects.
        """
        txt_path = Path(txt_path)
        if not txt_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {txt_path}")

        frame_boxes: Dict[str, List[BoxInfo]] = {}

        with txt_path.open('r', encoding='utf-8') as f:
            for line in f:
                box = BoxInfo(line)
                frame_id_int = int(box.frameId)
                player_id_int = int(box.playerId)

                if (tgt_frame - 5) <= frame_id_int <= (tgt_frame + 3) and player_id_int < 12:
                    frame_boxes.setdefault(box.frameId, []).append(box)

        return frame_boxes

    @classmethod
    def extract_frame_annot(cls, txt_path: str | Path) -> Dict[str, str]:
        """
        Extracts frame-level annotations from the file.

        Parameters
        ----------
        txt_path : str | Path
            Path to the annotation text file.

        Returns
        -------
        Dict[str, str]
            A dictionary mapping frame_id → category, sorted by frame number.
        """
        txt_path = Path(txt_path)
        if not txt_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {txt_path}")

        frames_annot_dict: Dict[str, str] = {}

        with txt_path.open('r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue  # skip malformed lines
                frame_id = parts[0].strip()
                if frame_id.endswith(".jpg"):
                    frame_id = frame_id[:-4]  # remove .jpg if present
                if not frame_id.isdigit():
                    continue  # skip empty or invalid frame IDs
                category = parts[1]
                frames_annot_dict[frame_id] = category

        # Sort only valid numeric frame IDs
        return dict(sorted(frames_annot_dict.items(), key=lambda p: int(p[0])))
