import argparse
import os
from pathlib import Path
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from num2words import num2words
from torchvision.datasets import DatasetFolder
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torchvision.transforms.functional import convert_image_dtype
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms.functional import to_tensor
from torchvision.utils import draw_bounding_boxes

from hardest.detectors import DetectionType
from hardest.hardness_strategies import DempsterHardnessCalculation
from hardest.hardness_strategies import EntropyCalculation
from hardest.hardness_strategies import MaxUncertaintyCalculation
from hardest.hardness_strategies import ScoreSamplingHardnessCalculation
from hardest.pycocotools import hardness_definition_factory
from hardest.utils import filter_pytorch_detections_by_score
from hardest.utils import load_detections


class UnsupervisedDataset(ImageFolder):
    def __init__(self, root: str, transform: Optional[Callable] = to_tensor):
        super().__init__(
            root,
            transform,
        )

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        return list(), dict()

    @staticmethod
    def make_dataset(
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        instances = []
        for root, _, fnames in sorted(os.walk(directory, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                instances.append((path, 0))
        return instances

    def __getitem__(self, index: int) -> torch.Tensor:
        image, target = super().__getitem__(index)
        return image


def main():
    parser = argparse.ArgumentParser(description="Reproduce experiments from the paper")
    parser.add_argument("--root", type=Path, help="root directory for images", required=True)
    parser.add_argument("--save-dir", type=Path, help="directory to save results", required=True)
    parser.add_argument("--detection-path", type=Path, help="path to detections", required=True)

    args = parser.parse_args()

    root = args.root
    save_dir = args.save_dir

    images = UnsupervisedDataset(root)

    detections = load_detections(args.detection_path)

    assert len(images) == len(detections)

    hardness = {}
    indices = {}
    for hardness_name, hardness_definition in hardness_definition_factory.items():
        hardness[hardness_name] = ScoreSamplingHardnessCalculation(hardness_definition, n_samples=10).eval_dataset(
            detections, images
        )
        indices[hardness_name] = np.argsort(hardness[hardness_name])

    hardness_entropy = EntropyCalculation(score_threshold=0).eval_dataset(detections, images)
    hardness_ds = DempsterHardnessCalculation(score_threshold=0).eval_dataset(detections, images)
    hardness_max = MaxUncertaintyCalculation(score_threshold=0).eval_dataset(detections, images)
    indices_entropy = np.argsort(hardness_entropy)
    indices_ds = np.argsort(hardness_ds)
    indices_max = np.argsort(hardness_max)

    for hardness_name in hardness_definition_factory:
        fig = plot_hardest_images_grid(
            images,
            detections,
            [indices[hardness_name], indices_entropy, indices_ds, indices_max],
            ["Score Sampling", "Score Entropy", "Evidential", "Max Uncertainty"],
        )
        fig.savefig(save_dir / f"{hardness_name}.png")


def plot_image(image: torch.Tensor, detections: DetectionType) -> torch.Tensor:
    detections = filter_pytorch_detections_by_score(detections, 0.5)
    image = convert_image_dtype(image, torch.uint8)

    plotted_box_image = image

    if len(detections["boxes"]) > 0:
        plotted_box_image = draw_bounding_boxes(plotted_box_image, detections["boxes"], None, "green", width=3)

    return plotted_box_image


def plot_hardest_images_grid(
    dataset: UnsupervisedDataset,
    detections: List[DetectionType],
    hardness_rank_indices: List[np.ndarray],
    hardness_rank_names: List[str],
    n_hardest=3,
):
    """Creates a grid of images with their detector performance bounding boxes over it."""
    assert len(hardness_rank_indices) == len(hardness_rank_names)
    n_techniques = len(hardness_rank_names)
    fig, ax_array = plt.subplots(n_techniques, n_hardest, figsize=(15, 15))
    for i in range(n_hardest):
        for hardness_rank_idx, (name, indices) in enumerate(zip(hardness_rank_names, hardness_rank_indices)):
            image = dataset[np.flip(indices)[i]]
            detection = detections[np.flip(indices)[i]]
            plotted_box_image = plot_image(image, detection)
            image_with_boxes = to_pil_image(plotted_box_image)
            ax_array[hardness_rank_idx, i].imshow(image_with_boxes)
            ordinal = num2words(i + 1, to="ordinal_num")
            ax_array[hardness_rank_idx, i].set(xlabel=f"{ordinal} hardest image", ylabel=name)
            ax_array[hardness_rank_idx, i].label_outer()
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    main()
