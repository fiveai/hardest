import json
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import TypedDict
from typing import Union
from typing import overload

import torch
from torchvision.ops.boxes import box_convert

from hardest.detectors import DetectionType
from hardest.detectors import KClassDetectionType


class AnnotationType(TypedDict):
    """Annotation as a pycocotools target.

    See https://github.com/cocodataset/cocoapi for more information

    """

    bbox: List[float]
    """2d bounding box in (x1, y1, x2, y2) format."""
    iscrowd: bool
    """Indicates whether the box represents a single object or a crowd."""
    category_id: int
    """Labels for the bounding boxes"""


class AnnotationTypeTorch(TypedDict):
    boxes: torch.Tensor
    """shape (n, 4); dtype float32.
    Bounding boxes in (x1, y1, x2, y2) format.,
    0 <= x1 < x2 and 0 <= y1 < y2.
    """
    labels: torch.Tensor
    """shape (n,); dtype int64.
    Labels for the detected objects.
    """
    iscrowd: torch.Tensor
    """shape (n,); dtype bool.
    Class probabilities for the detected objects.
    """


def coco_api_to_torchvision(coco_ann: List[AnnotationType]) -> AnnotationTypeTorch:
    """
    Convert python coco api targets to torchvision targets
    Args:
        coco_ann: Annotations in pycocotools format

    Returns:
        Annotations in torchvision format
    """
    boxes_coco = torch.Tensor([ann["bbox"] for ann in coco_ann])  # x, y, w, h
    boxes_pytorch = box_convert(boxes_coco, "xywh", "xyxy") if len(boxes_coco) > 0 else boxes_coco
    iscrowd = torch.Tensor([ann["iscrowd"] for ann in coco_ann])
    return {
        "boxes": boxes_pytorch,
        "labels": torch.Tensor([ann["category_id"] for ann in coco_ann]),
        "iscrowd": iscrowd,
    }


@overload
def filter_pytorch_detections_by_score(detections: KClassDetectionType, score_threshold: float) -> KClassDetectionType:
    ...


@overload
def filter_pytorch_detections_by_score(detections: DetectionType, score_threshold: float) -> DetectionType:
    ...


def filter_pytorch_detections_by_score(
    detections: Union[DetectionType, KClassDetectionType], score_threshold: float
) -> Union[DetectionType, KClassDetectionType]:
    """
    Applies a score threshold to torchvision detections
    Args:
        detections: Detections to be filtered in torchvision format
        score_threshold: Only detections with scores above the score threshold will be kept

    Returns:
        filtered detections
    """
    keep_mask = detections["scores"] >= score_threshold
    return {key: value[keep_mask] for key, value in detections.items()}  # type: ignore
    # ignore required until https://github.com/python/mypy/issues/7981 is fixed


def serialise_detections(save_path: Path, detections: List[DetectionType]):
    """
    Saves torchvision detections into a json
    Args:
        save_path: path to json where detections should be saved
        detections: detections to be saved

    """
    exportable_detections = [{key: value.cpu().numpy().tolist() for key, value in det.items()} for det in detections]  # type: ignore
    # ignore required until https://github.com/python/mypy/issues/7981 is fixed
    with open(save_path, "w") as output_file:
        json.dump(exportable_detections, output_file)


def load_detections(load_path: Path) -> List[DetectionType]:
    """
    Loads torchvision detections from a json
    Args:
        load_path: path to json where detections should be loaded from

    Returns:
        loaded detections

    """
    with open(load_path, "r") as load_file:
        exportable_detections = json.load(load_file)
    return [{key: torch.Tensor(value) for key, value in det.items()} for det in exportable_detections]  # type: ignore
    # ignore required until https://github.com/python/mypy/issues/7981 is fixed


NUSCENES_CATEGORIES_MMDET = (
    "car",
    "truck",
    "trailer",
    "bus",
    "construction_vehicle",
    "bicycle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "barrier",
)

TO_AV_SIMPLE = {
    # mmdet
    "car": "Vehicle",
    "truck": "Vehicle",
    "trailer": "Vehicle",
    "bus": "Vehicle",
    "construction_vehicle": "Vehicle",
    "bicycle": "Vehicle",
    "motorcycle": "Vehicle",
    "pedestrian": "Pedestrian",
    "traffic_cone": None,
    "barrier": None,
}

CONTIGUOUS_ID_OF_AVSIMPLE_NAME = {
    "Vehicle": 0,
    "Pedestrian": 1,
}


def nuimages_coco_mmdet_to_av_simple(annotations: Optional[AnnotationTypeTorch]) -> Optional[AnnotationTypeTorch]:
    """
    Converts labels for annotations in mmdet schema to a simplified schema.
    Args:
        annotations: Annotations in the mmdet schema

    Returns:
        Annotations in a simplified schema

    """
    if annotations is None:
        return None
    if len(annotations["labels"]) == 0:
        return annotations
    label = annotations["labels"]
    names = [NUSCENES_CATEGORIES_MMDET[int(instance.item())] for instance in label]
    av_simple_names = [TO_AV_SIMPLE[n] for n in names]
    mask = torch.tensor([n is not None for n in av_simple_names])

    new_labels = torch.tensor([-1 if n is None else CONTIGUOUS_ID_OF_AVSIMPLE_NAME[n] for n in av_simple_names])
    annotations["labels"] = new_labels
    for key, value in annotations.items():
        annotations[key] = value[mask]  # type: ignore
    # ignore required until https://github.com/python/mypy/issues/7981 is fixed
    return annotations
