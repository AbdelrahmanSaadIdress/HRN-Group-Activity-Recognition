import os
import pickle
from tqdm import tqdm
from typing import Dict, Any
from .Extractor import Extractor  # assuming same package
from .BoxInfo import BoxInfo


class AnnotationPreparer:
    """
    Handles the preparation, saving, and loading of volleyball dataset annotations.

    This class organizes annotations from raw crop-level and frame-level annotation
    text files into a hierarchical dictionary structure and serializes it for reuse.
    """

    @staticmethod
    def prepare_annotations(
        crops_annots_path: str,
        frames_annots_path: str,
        save_path: str = '/kaggle/working/',
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Prepares nested annotation data from raw text files and saves it as a pickle.

        Structure:
        {
            match_id: {
                clip_id: {
                    'category': str,
                    'frames_boxes_dct': Dict[frame_id, List[BoxInfo]]
                },
                ...
            },
            ...
        }

        Parameters
        ----------
        crops_annots_path : str
            Root path containing crop-level annotation folders.
        frames_annots_path : str
            Root path containing frame-level annotations.
        save_path : str, optional
            Directory to save the serialized `data.pkl` file (default: `/kaggle/working/`).
        verbose : bool, optional
            Whether to print progress and completion messages.

        Returns
        -------
        Dict[str, Any]
            Nested dictionary of all matches and their annotations.
        """
        matches_id = sorted(
            [match_id for match_id in os.listdir(crops_annots_path) if match_id.isdigit()],
            key=int
        )
        matches_dict: Dict[str, Any] = {}

        for match_id in tqdm(matches_id, desc="Processing matches", disable=not verbose):
            match_path = os.path.join(crops_annots_path, match_id)
            clips_id = sorted(
                [clip_id for clip_id in os.listdir(match_path) if clip_id.isdigit()],
                key=int
            )
            clips_dict: Dict[str, Any] = {}

            frames_file = os.path.join(frames_annots_path, match_id, 'annotations.txt')
            if not os.path.exists(frames_file):
                if verbose:
                    print(f"⚠️ Missing frame annotation file for match {match_id}: {frames_file}")
                continue

            frames_annots = Extractor.extract_frame_annot(frames_file)

            for clip_id in clips_id:
                clip_id_str = str(clip_id)
                if clip_id_str not in frames_annots:
                    if verbose:
                        print(f"⚠️ Clip ID {clip_id_str} missing in frames_annots for match {match_id}")
                    continue

                clip_category = frames_annots[clip_id_str]
                crops_file = os.path.join(match_path, clip_id_str, f"{clip_id_str}.txt")

                try:
                    crops_annots = Extractor.extract_crops_annot(crops_file, int(clip_id))
                except Exception as e:
                    print(f"❌ Failed to extract crops for match {match_id}, clip {clip_id_str}: {e}")
                    continue

                clips_dict[clip_id_str] = {
                    "category": clip_category,
                    "frames_boxes_dct": crops_annots,
                }

            matches_dict[match_id] = clips_dict

        # Save the full dictionary
        save_file = os.path.join(save_path, "data.pkl")
        os.makedirs(save_path, exist_ok=True)

        with open(save_file, "wb") as f:
            pickle.dump(matches_dict, f)

        if verbose:
            print(f"\n✅ All annotations processed successfully!")
            print(f"💾 Saved to: {save_file}")

        return matches_dict

    @staticmethod
    def load_annotations(save_path: str = '/kaggle/working/') -> Dict[str, Any]:
        """
        Loads prepared annotations from the pickle file.

        Parameters
        ----------
        save_path : str
            Directory containing `data.pkl`.

        Returns
        -------
        Dict[str, Any]
            Nested annotations dictionary.
        """
        file_path = os.path.join(save_path, "data.pkl")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Saved annotations file not found: {file_path}")

        with open(file_path, "rb") as f:
            return pickle.load(f)
