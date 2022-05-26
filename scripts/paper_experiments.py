import argparse
import glob
import pickle
import typing
from pathlib import Path
from typing import Dict
from typing import Iterable
from typing import List
from typing import Set
from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics
import torch
import torch.utils.data as data
from num2words import num2words
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score
from torchvision.datasets.coco import CocoDetection
from torchvision.transforms.functional import convert_image_dtype
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms.functional import to_tensor
from torchvision.utils import draw_bounding_boxes
from tqdm import tqdm
from tueplots import bundles

from hardest.detectors import DetectionType
from hardest.detectors import KClassDetectionType
from hardest.hardness_strategies import BinaryDempsterHardnessCalculation
from hardest.hardness_strategies import BinaryEntropyCalculation
from hardest.hardness_strategies import DempsterHardnessCalculation
from hardest.hardness_strategies import EntropyCalculation
from hardest.hardness_strategies import MaxUncertaintyCalculation
from hardest.hardness_strategies import ScoreSamplingHardnessCalculation
from hardest.hardness_strategies import SupervisedHardnessCalculation
from hardest.pycocotools import FN
from hardest.pycocotools import FP
from hardest.pycocotools import CocoHardness
from hardest.pycocotools import HardnessDefinition
from hardest.pycocotools import TotalFalse
from hardest.pycocotools import overlap
from hardest.pycocotools import pixel
from hardest.utils import AnnotationTypeTorch
from hardest.utils import coco_api_to_torchvision
from hardest.utils import filter_pytorch_detections_by_score
from hardest.utils import load_detections
from hardest.utils import nuimages_coco_mmdet_to_av_simple

plt.rcParams.update(bundles.icml2022())


matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


METRICS_NAME = {
    "fp": "Total(fp)",
    "fn": "Total(fn)",
    "false": "Total(false)",
    "fp_pixel": "PixelAdj(fp)",
    "fn_pixel": "PixelAdj(fn)",
    "false_pixel": "PixelAdj(false)",
    "fp_overlap": "OccAware(fp)",
    "fn_overlap": "OccAware(fn)",
    "false_overlap": "OccAware(false)",
}


class ImageDatasetWrapper(data.Dataset):
    """
    Wraps a dataset to provide an image dataset without caching all images in memory
    """

    def __init__(self, dataset: CocoDetection):
        self.dataset = dataset

    def __getitem__(self, index: int) -> torch.Tensor:
        image, _ = self.dataset.__getitem__(index)
        return image

    def __len__(self):
        return self.dataset.__len__()


class OnlyImageSizeDatasetWrapper(data.Dataset):
    """
    If your hardness definition only requires the size of each image this wrapper can be used to avoid loading each
    image multiple times by caching the image sizes
    """

    def __init__(self, dataset: CocoDetection, constant_sizes=False):
        self.dataset = dataset
        if constant_sizes:
            image, _ = dataset[0]
            self.image_shapes = [image.shape for _ in range(len(dataset))]
        else:
            self.image_shapes = [image.shape for image, _ in tqdm(dataset, desc="Reading image sizes")]

    def __getitem__(self, index: int):
        return torch.empty(self.image_shapes[index])

    def __len__(self):
        return self.dataset.__len__()


def get_targets(dataset: CocoDetection) -> List[AnnotationTypeTorch]:
    """Get all targets quickly from a CocoDetection dataset without loading the images."""
    return list(dataset.target_transform(dataset._load_target(idx)) for idx in dataset.ids)


def use_latex_fonts() -> None:
    """Enable using latex fonts while plotting."""
    plt.rcParams.update({"text.usetex": True, "font.family": "sans-serif", "font.sans-serif": ["Helvetica"]})
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Palatino"],
        }
    )
    METRICS_NAME["fp"] = r"$\texttt{Total}(\texttt{fp})$"
    METRICS_NAME["fn"] = r"$\texttt{Total}(\texttt{fn})$"
    METRICS_NAME["false"] = r"$\texttt{Total}(\texttt{false})$"
    METRICS_NAME["fp_pixel"] = r"$\texttt{PixelAdj}(\texttt{fp})$"
    METRICS_NAME["fn_pixel"] = r"$\texttt{PixelAdj}(\texttt{fn})$"
    METRICS_NAME["false_pixel"] = r"$\texttt{PixelAdj}(\texttt{false})$"
    METRICS_NAME["fp_overlap"] = r"$\texttt{OccAware}(\texttt{fp})$"
    METRICS_NAME["fn_overlap"] = r"$\texttt{OccAware}(\texttt{fn})$"
    METRICS_NAME["false_overlap"] = r"$\texttt{OccAware}(\texttt{false})$"


def main():
    parser = argparse.ArgumentParser(description="Reproduce experiments from the paper")
    parser.add_argument("--coco-root", type=Path, help="root directory for coco")
    parser.add_argument("--nuimages-root", type=Path, help="root directory for nuimages")
    parser.add_argument("--save-dir", type=Path, help="directory to save results")
    parser.add_argument(
        "--detection-path-nuimages",
        type=Path,
        help="path to folder containing detections for nuimages",
        default="./detections/nuimages/",
    )
    parser.add_argument(
        "--detection-path-coco",
        type=Path,
        help="path to folder containing detections for coco",
        default="./detections/coco/",
    )
    parser.add_argument("--latex", action="store_true", help="Enable using latex for plots.")
    parser.add_argument("--no-remap", action="store_true", help="Disable nuimages remap")

    args = parser.parse_args()

    save_dir = args.save_dir

    if args.latex:
        use_latex_fonts()

    # Name: (Dataset, path to detections, hardness definitions)
    detector_dataset_pairs: Dict[Tuple[CocoDetection, str]] = {}

    if args.coco_root is not None:
        dataset = CocoDetection(
            root=args.coco_root / "val2017/",
            annFile=args.coco_root / "annotations/instances_val2017.json",
            transform=to_tensor,
            target_transform=coco_api_to_torchvision,
        )
        cached_detection_files = glob.glob(str(args.detection_path_coco / "*.json"))
        images = OnlyImageSizeDatasetWrapper(dataset, constant_sizes=False)
        for file in cached_detection_files:
            humane_name = " ".join(Path(file).stem.split("_"))
            detector_dataset_pairs[f"Coco {humane_name}"] = (dataset, Path(file), images)

    if args.nuimages_root is not None:
        dataset = CocoDetection(
            root=args.nuimages_root,
            annFile=args.nuimages_root / "nuimages_v1.0-val.json",
            transform=to_tensor,
            target_transform=coco_api_to_torchvision
            if args.no_remap
            else lambda x: nuimages_coco_mmdet_to_av_simple(coco_api_to_torchvision(x)),
        )
        # This requires converting to the coco schema, see:
        # https://github.com/open-mmlab/mmdetection3d/blob/master/configs/nuimages/README.md
        cached_detection_files = glob.glob(str(args.detection_path_nuimages / "*.json"))
        images = OnlyImageSizeDatasetWrapper(dataset, constant_sizes=True)
        for file in cached_detection_files:
            humane_name = " ".join(Path(file).stem.split("_"))
            detector_dataset_pairs[f"Nuimages {humane_name}"] = (dataset, Path(file), images)

    if len(detector_dataset_pairs) == 0:
        raise ValueError("No detector dataset pairs found")

    results = {}
    for name, (
        dataset,
        file,
        images,
    ) in detector_dataset_pairs.items():
        print(f"Processing dataset/detector pair: {name}...")
        cat_ids = {cat["id"] for cat in dataset.coco.cats.values()}
        hardness_definitions: Dict[str, HardnessDefinition] = {
            "fp": FP(cat_ids=cat_ids),
            "fn": FN(cat_ids=cat_ids),
            "false": TotalFalse(cat_ids=cat_ids),
            "fp_pixel": FP(weighting_fn=pixel, cat_ids=cat_ids),
            "fn_pixel": FN(weighting_fn=pixel, cat_ids=cat_ids),
            "false_pixel": TotalFalse(weighting_fn=pixel, cat_ids=cat_ids),
            "fp_overlap": FP(weighting_fn=overlap, cat_ids=cat_ids),
            "fn_overlap": FN(weighting_fn=overlap, cat_ids=cat_ids),
            "false_overlap": TotalFalse(weighting_fn=overlap, cat_ids=cat_ids),
        }

        targets = get_targets(dataset)

        detections = load_detections(file)
        if "Nuimages" in name and not args.no_remap:
            detections = [nuimages_coco_mmdet_to_av_simple(det) for det in detections]

        assert len(dataset) == len(detections), "Detections do not match dataset"

        accuracies, bins = detector_calibration(targets, detections, images, cat_ids=cat_ids)
        fig = plot_calibration(accuracies, bins)
        fig.savefig(
            save_dir / "_".join([name, "calibration.pdf"]),
        )

        hardness = {}
        hardness_gt = {}
        indices = {}
        indices_gt = {}
        for hardness_name, hardness_definition in hardness_definitions.items():
            hardness[hardness_name] = ScoreSamplingHardnessCalculation(hardness_definition, n_samples=10).eval_dataset(
                detections, images
            )
            indices[hardness_name] = np.argsort(hardness[hardness_name])
            hardness_gt[hardness_name] = SupervisedHardnessCalculation(hardness_definition).eval_dataset(
                detections, targets, images
            )
            indices_gt[hardness_name] = np.argsort(hardness_gt[hardness_name])

        if "retinanet" not in name:
            detections = typing.cast(KClassDetectionType, detections)
            hardness_entropy = EntropyCalculation(score_threshold=0).eval_dataset(detections, images)
            hardness_ds = DempsterHardnessCalculation(score_threshold=0).eval_dataset(detections, images)
        else:
            assert "logits" not in detections[0]
            hardness_entropy = BinaryEntropyCalculation(score_threshold=0).eval_dataset(detections, images)
            hardness_ds = BinaryDempsterHardnessCalculation(score_threshold=0).eval_dataset(detections, images)

        hardness_max = MaxUncertaintyCalculation(score_threshold=0).eval_dataset(detections, images)
        indices_entropy = np.argsort(hardness_entropy)
        indices_ds = np.argsort(hardness_ds)
        indices_max = np.argsort(hardness_max)

        for hardness_name, hardness_definition in hardness_definitions.items():
            regret_plot(
                hardness_name,
                hardness_gt[hardness_name],
                {
                    "Score Sampling": indices[hardness_name],
                    "Entropy": indices_entropy,
                    "Dempster Shafer": indices_ds,
                },
            )
            plt.savefig(save_dir / f"regret_{name}_{hardness_name}.pdf")

        score_sampling_roc = []
        entropy_roc = []
        ds_roc = []
        max_roc = []
        for percentile_threshold in [5, 10, 25, 50]:
            hardness_threshold = {
                name: np.percentile(values, 100 - percentile_threshold) for name, values in hardness_gt.items()
            }
            hard_mask = {name: hardness_gt[name] > hardness_threshold[name] for name, values in hardness_gt.items()}
            score_sampling_roc.append(
                {f"{name}": sklearn.metrics.roc_auc_score(mask, hardness[name]) for name, mask in hard_mask.items()}
            )
            entropy_roc.append(
                {name: sklearn.metrics.roc_auc_score(mask, hardness_entropy) for name, mask in hard_mask.items()}
            )
            ds_roc.append({name: sklearn.metrics.roc_auc_score(mask, hardness_ds) for name, mask in hard_mask.items()})
            max_roc.append(
                {name: sklearn.metrics.roc_auc_score(mask, hardness_max) for name, mask in hard_mask.items()}
            )

        roc_df = pd.DataFrame(
            data={
                "hardness_def": hardness_threshold.keys(),
                "threshold": hardness_threshold.values(),
                "score_entropy": np.mean([list(item.values()) for item in entropy_roc], axis=0),
                "evidential": np.mean([list(item.values()) for item in ds_roc], axis=0),
                "score_sampling": np.mean([list(item.values()) for item in score_sampling_roc], axis=0),
                "max_uncertainty": np.mean([list(item.values()) for item in max_roc], axis=0),
            }
        )

        print(roc_df)

        score_sampling_mqc = {
            name: median_query_count(indices[name], np.array(true_hardness))
            for name, true_hardness in hardness_gt.items()
        }
        entropy_mqc = {
            name: median_query_count(indices_entropy, np.array(true_hardness))
            for name, true_hardness in hardness_gt.items()
        }
        ds_mqc = {
            name: median_query_count(indices_ds, np.array(true_hardness)) for name, true_hardness in hardness_gt.items()
        }
        max_mqc = {
            name: median_query_count(indices_max, np.array(true_hardness))
            for name, true_hardness in hardness_gt.items()
        }

        mqc_df = pd.DataFrame(
            data={
                "hardness_def": score_sampling_mqc.keys(),
                "score_entropy": entropy_mqc.values(),
                "evidential": ds_mqc.values(),
                "score_sampling": score_sampling_mqc.values(),
                "max_uncertainty": max_mqc.values(),
            }
        )

        print(mqc_df)

        hardness_names = list(hardness_definitions.keys())
        score_sampling_correlation = [
            spearmanr(hardness[hardness_name], hardness_gt[hardness_name]).correlation
            for hardness_name in hardness_definitions
        ]
        entropy_correlation = [
            spearmanr(hardness_entropy, hardness_gt[hardness_name]).correlation
            for hardness_name in hardness_definitions
        ]
        ds_correlation = [
            spearmanr(hardness_ds, hardness_gt[hardness_name]).correlation for hardness_name in hardness_definitions
        ]
        max_correlation = [
            spearmanr(hardness_max, hardness_gt[hardness_name]).correlation for hardness_name in hardness_definitions
        ]

        spe_df = pd.DataFrame(
            data={
                "hardness_def": hardness_names,
                "score_entropy": entropy_correlation,
                "evidential": ds_correlation,
                "score_sampling": score_sampling_correlation,
                "max_uncertainty": max_correlation,
            }
        )

        print(spe_df)

        score_sampling_ndcg = [
            ndcg(
                hardness_gt[hardness_name],
                hardness[hardness_name],
            )
            for hardness_name in hardness_definitions
        ]
        entropy_ndcg = [
            ndcg(
                hardness_gt[hardness_name],
                hardness_entropy,
            )
            for hardness_name in hardness_definitions
        ]
        ds_ndcg = [
            ndcg(
                hardness_gt[hardness_name],
                hardness_ds,
            )
            for hardness_name in hardness_definitions
        ]
        max_ndcg = [
            ndcg(
                hardness_gt[hardness_name],
                hardness_max,
            )
            for hardness_name in hardness_definitions
        ]

        ndcg_df = pd.DataFrame(
            data={
                "hardness_def": hardness_names,
                "score_entropy": entropy_ndcg,
                "evidential": ds_ndcg,
                "score_sampling": score_sampling_ndcg,
                "max_uncertainty": max_ndcg,
            }
        )

        print(ndcg_df)

        results[name] = {
            "spe_df": spe_df,
            "mqc_df": mqc_df,
            "roc_df": roc_df,
            "ndcg_df": ndcg_df,
            "hardness_gt": hardness_gt,
            "hardness": hardness,
            "hardness_entropy": hardness_entropy,
            "hardness_ds": hardness_ds,
            "hardness_max": hardness_max,
        }

    with open(save_dir / "hardness_dump.pkl", "wb") as output_file:
        pickle.dump(results, output_file)

    dpi = 80
    fig = plt.figure(constrained_layout=True, figsize=(650 / dpi, 500 / dpi), dpi=dpi)
    ax_array = fig.subplots(2, 2, squeeze=False)
    for detector_idx, (
        name,
        _,
    ) in enumerate(detector_dataset_pairs.items()):
        im = definitions_of_hardness_are_different(
            ax_array[detector_idx % 2, detector_idx // 2],
            true_hardness=results[name]["hardness_gt"],
            detector_name=name,
        )
    fig.tight_layout()
    fig.colorbar(im, ax=ax_array.ravel().tolist(), label="Spearman correlation coefficient")

    fig.savefig(
        f"{save_dir}/hardness_defs_are_different.pdf",
    )


def ndcg(true: List, ranking: List) -> float:
    return ndcg_score(
        np.array(
            [
                true,
            ]
        ),
        np.array(
            [
                ranking,
            ]
        ),
    )


def median_query_count(sort_indices: np.ndarray, true_hardness: np.ndarray) -> float:
    total_hardness = sum(true_hardness)
    cumulative_hardness_estimated = np.cumsum(true_hardness[np.flip(sort_indices)])
    return 1 - np.searchsorted(cumulative_hardness_estimated, total_hardness * 0.5, "right") / len(true_hardness)


def plot_hardest_images_grid(
    dataset: CocoDetection,
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
            image, target = dataset[np.flip(indices)[i]]
            detection = detections[np.flip(indices)[i]]
            plotted_box_image = plot_image(image, target, detection)
            image_with_boxes = to_pil_image(plotted_box_image)
            ax_array[hardness_rank_idx, i].imshow(image_with_boxes)
            ordinal = num2words(i + 1, to="ordinal_num")
            ax_array[hardness_rank_idx, i].set(xlabel=f"{ordinal} hardest image", ylabel=name)
            ax_array[hardness_rank_idx, i].label_outer()
            ax_array[hardness_rank_idx, i].tick_params(
                axis="both",
                which="both",
                bottom=False,
                left=False,
                top=False,
                right=False,
                labelbottom=False,
                labelleft=False,
            )
    fig.tight_layout()
    return fig


def plot_image(image: torch.Tensor, target: AnnotationTypeTorch, detections: DetectionType) -> torch.Tensor:
    detections = filter_pytorch_detections_by_score(detections, 0.5)
    image = convert_image_dtype(image, torch.uint8)
    fp_mask, fn_mask = CocoHardness().get_masks(detections, target, image)
    plotted_box_image = image
    if len(target["boxes"]) > 0:
        tp = target["boxes"][~torch.Tensor(np.array(fn_mask)).bool()]
        fn = target["boxes"][torch.Tensor(np.array(fn_mask)).bool()]
        if len(fn) > 0:
            plotted_box_image = draw_bounding_boxes(plotted_box_image, fn, colors="blue", width=3)
        if len(tp) > 0:
            plotted_box_image = draw_bounding_boxes(plotted_box_image, tp, colors="green", width=3)

    if len(detections["boxes"]) > 0:
        fp = detections["boxes"][torch.Tensor(np.array(fp_mask)).bool()]
        if len(fp) > 0:
            plotted_box_image = draw_bounding_boxes(plotted_box_image, fp, colors="red", width=3)

    return plotted_box_image


def detector_calibration(
    targets: List[AnnotationTypeTorch],
    detections: List[DetectionType],
    images: Iterable[torch.Tensor],
    cat_ids: Set[int],
    bins: np.ndarray = np.linspace(-1e-10, 1 + 1e-10, 11),
) -> Tuple[np.ndarray, np.ndarray]:
    all_fps = []
    for target, detection, image in zip(targets, detections, images):
        fp_mask, _ = CocoHardness(cat_ids).get_masks(detection, target, image)
        all_fps.append(fp_mask)
    bin_mask = np.digitize(torch.cat([detection["scores"] for detection in detections]), bins=bins)
    all_fp_mask = np.concatenate(all_fps)
    bin_accuracies = np.array([np.mean(~all_fp_mask[bin_mask == bin_idx]) for bin_idx in range(1, len(bins))])
    return bin_accuracies, bins


def plot_calibration(accuracies: np.ndarray, bin_edges: np.ndarray):
    fig, ax = plt.subplots()
    ax.plot([0.0, 1.0], [0.0, 1.0], label="x = y", linestyle="dashed")
    x = np.stack([bin_edges[:-1], bin_edges[1:]]).flatten("F")
    y = np.stack([accuracies, accuracies]).flatten("F")
    ax.plot(x, y)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Precision")
    ax.legend()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    return fig


def definitions_of_hardness_are_different(
    ax,
    true_hardness: Dict[str, np.ndarray],
    detector_name: str,
):
    """This function makes the plot showing that different definitions of hardness are different."""
    ks = [METRICS_NAME[name] for name in list(true_hardness.keys())]
    z = np.array([[spearmanr(x, y)[0] for x in true_hardness.values()] for y in true_hardness.values()])
    im = ax.imshow(
        z,
        vmin=0.4,
        vmax=1,
    )
    ax.set_xticks(np.arange(len(ks)), labels=ks)
    ax.set_yticks(np.arange(len(ks)), labels=ks)
    ax.set_xlabel("Hardness metric")
    ax.set_ylabel("Hardness metric")
    ax.set_title(f"{detector_name}")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for j in range(len(ks)):
        for k in range(len(ks)):
            ax.text(j, k, f"{z[j, k]:.2f}", ha="center", va="center", color="w")
    return im


def regret_plot(
    hardness_name: str,
    gt_hardness: np.ndarray,
    indices_rankings: Dict[str, np.ndarray],
):
    plt.clf()
    plt.plot(np.cumsum(np.sort(gt_hardness)[::-1]), label="Perfect")
    plt.plot([0, len(gt_hardness)], [0, sum(gt_hardness)], label="Random")
    for ranking_name, indices in indices_rankings.items():
        plt.plot(np.cumsum(np.array(gt_hardness)[indices][::-1]), label=ranking_name)
    plt.ylabel(f"Cumulative {METRICS_NAME[hardness_name]}")
    plt.xlabel("Query budget")
    plt.xlim([0, len(gt_hardness)])
    plt.ylim([0, sum(gt_hardness)])
    plt.legend()
    # return fig


def convert_to_coco(label_array_index):
    return [coco_by_id[int(label)] for label in label_array_index]


COCO = [
    {"supercategory": "person", "id": 1, "name": "person"},
    {"supercategory": "vehicle", "id": 2, "name": "bicycle"},
    {"supercategory": "vehicle", "id": 3, "name": "car"},
    {"supercategory": "vehicle", "id": 4, "name": "motorcycle"},
    {"supercategory": "vehicle", "id": 5, "name": "airplane"},
    {"supercategory": "vehicle", "id": 6, "name": "bus"},
    {"supercategory": "vehicle", "id": 7, "name": "train"},
    {"supercategory": "vehicle", "id": 8, "name": "truck"},
    {"supercategory": "vehicle", "id": 9, "name": "boat"},
    {"supercategory": "outdoor", "id": 10, "name": "traffic light"},
    {"supercategory": "outdoor", "id": 11, "name": "fire hydrant"},
    {"supercategory": "outdoor", "id": 13, "name": "stop sign"},
    {"supercategory": "outdoor", "id": 14, "name": "parking meter"},
    {"supercategory": "outdoor", "id": 15, "name": "bench"},
    {"supercategory": "animal", "id": 16, "name": "bird"},
    {"supercategory": "animal", "id": 17, "name": "cat"},
    {"supercategory": "animal", "id": 18, "name": "dog"},
    {"supercategory": "animal", "id": 19, "name": "horse"},
    {"supercategory": "animal", "id": 20, "name": "sheep"},
    {"supercategory": "animal", "id": 21, "name": "cow"},
    {"supercategory": "animal", "id": 22, "name": "elephant"},
    {"supercategory": "animal", "id": 23, "name": "bear"},
    {"supercategory": "animal", "id": 24, "name": "zebra"},
    {"supercategory": "animal", "id": 25, "name": "giraffe"},
    {"supercategory": "accessory", "id": 27, "name": "backpack"},
    {"supercategory": "accessory", "id": 28, "name": "umbrella"},
    {"supercategory": "accessory", "id": 31, "name": "handbag"},
    {"supercategory": "accessory", "id": 32, "name": "tie"},
    {"supercategory": "accessory", "id": 33, "name": "suitcase"},
    {"supercategory": "sports", "id": 34, "name": "frisbee"},
    {"supercategory": "sports", "id": 35, "name": "skis"},
    {"supercategory": "sports", "id": 36, "name": "snowboard"},
    {"supercategory": "sports", "id": 37, "name": "sports ball"},
    {"supercategory": "sports", "id": 38, "name": "kite"},
    {"supercategory": "sports", "id": 39, "name": "baseball bat"},
    {"supercategory": "sports", "id": 40, "name": "baseball glove"},
    {"supercategory": "sports", "id": 41, "name": "skateboard"},
    {"supercategory": "sports", "id": 42, "name": "surfboard"},
    {"supercategory": "sports", "id": 43, "name": "tennis racket"},
    {"supercategory": "kitchen", "id": 44, "name": "bottle"},
    {"supercategory": "kitchen", "id": 46, "name": "wine glass"},
    {"supercategory": "kitchen", "id": 47, "name": "cup"},
    {"supercategory": "kitchen", "id": 48, "name": "fork"},
    {"supercategory": "kitchen", "id": 49, "name": "knife"},
    {"supercategory": "kitchen", "id": 50, "name": "spoon"},
    {"supercategory": "kitchen", "id": 51, "name": "bowl"},
    {"supercategory": "food", "id": 52, "name": "banana"},
    {"supercategory": "food", "id": 53, "name": "apple"},
    {"supercategory": "food", "id": 54, "name": "sandwich"},
    {"supercategory": "food", "id": 55, "name": "orange"},
    {"supercategory": "food", "id": 56, "name": "broccoli"},
    {"supercategory": "food", "id": 57, "name": "carrot"},
    {"supercategory": "food", "id": 58, "name": "hot dog"},
    {"supercategory": "food", "id": 59, "name": "pizza"},
    {"supercategory": "food", "id": 60, "name": "donut"},
    {"supercategory": "food", "id": 61, "name": "cake"},
    {"supercategory": "furniture", "id": 62, "name": "chair"},
    {"supercategory": "furniture", "id": 63, "name": "couch"},
    {"supercategory": "furniture", "id": 64, "name": "potted plant"},
    {"supercategory": "furniture", "id": 65, "name": "bed"},
    {"supercategory": "furniture", "id": 67, "name": "dining table"},
    {"supercategory": "furniture", "id": 70, "name": "toilet"},
    {"supercategory": "electronic", "id": 72, "name": "tv"},
    {"supercategory": "electronic", "id": 73, "name": "laptop"},
    {"supercategory": "electronic", "id": 74, "name": "mouse"},
    {"supercategory": "electronic", "id": 75, "name": "remote"},
    {"supercategory": "electronic", "id": 76, "name": "keyboard"},
    {"supercategory": "electronic", "id": 77, "name": "cell phone"},
    {"supercategory": "appliance", "id": 78, "name": "microwave"},
    {"supercategory": "appliance", "id": 79, "name": "oven"},
    {"supercategory": "appliance", "id": 80, "name": "toaster"},
    {"supercategory": "appliance", "id": 81, "name": "sink"},
    {"supercategory": "appliance", "id": 82, "name": "refrigerator"},
    {"supercategory": "indoor", "id": 84, "name": "book"},
    {"supercategory": "indoor", "id": 85, "name": "clock"},
    {"supercategory": "indoor", "id": 86, "name": "vase"},
    {"supercategory": "indoor", "id": 87, "name": "scissors"},
    {"supercategory": "indoor", "id": 88, "name": "teddy bear"},
    {"supercategory": "indoor", "id": 89, "name": "hair drier"},
    {"supercategory": "indoor", "id": 90, "name": "toothbrush"},
]

coco_by_id = {item["id"]: item["name"] for item in COCO}

if __name__ == "__main__":
    main()
