import contextlib
import copy
from collections import defaultdict
from contextlib import redirect_stdout
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import TypedDict
from typing import Union

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.ops import box_area
from torchvision.ops import box_convert
from torchvision.ops.boxes import _box_inter_union

from hardest.detectors import BoxesType
from hardest.detectors import MasksType
from hardest.hardness_definitions import AnnotationTypeTorch
from hardest.hardness_definitions import DetectionType
from hardest.hardness_definitions import HardnessDefinition

COCO_CAT_IDS = set(range(91))
AV_SIMPLE_CAT_IDS = set(range(2))

# Note that this file contains lots of copied boilerplate code from
# https://github.com/pytorch/vision/tree/main/references/detection which has never been properly tested or integrated
# into torchvision, in addition to copied code from pycocotools


def no_weighting(instances: BoxesType, masks: MasksType, image: torch.Tensor) -> torch.Tensor:
    """
    Equal weight for each detection
    Args:
        instances: Box instance for a single image.
        masks: TP/FP/FN masks for the box instances.
        image: Image for which hardness is being computed in torchvision format (C, H, W)

    Returns:
        weighting
    """
    return torch.ones_like(instances["labels"])


def pixel(instances: BoxesType, masks: MasksType, image: torch.Tensor) -> torch.Tensor:
    """
    Weight for each detection based on area relative to whole image area
    Args:
        instances: Box instance for a single image.
        masks: TP/FP/FN masks for the box instances.
        image: Image for which hardness is being computed in torchvision format (C, H, W)

    Returns:
        weighting
    """
    if len(instances["labels"]) > 0:
        image_area = image.shape[-1] * image.shape[-2]
        weights_det = box_area(instances["boxes"]) / image_area
    else:
        weights_det = torch.ones_like(instances["labels"])

    return weights_det


def overlap(instances: BoxesType, masks: MasksType, image: torch.Tensor) -> torch.Tensor:
    """
    Weight for each detection equal to number of overlapping TP boxes (can be fractional)
    Args:
        instances: Box instance for a single image.
        masks: TP/FP/FN masks for the box instances.
        image: Image for which hardness is being computed in torchvision format (C, H, W)

    Returns:
        weighting
    """
    if len(instances["labels"]) > 0:
        assert instances["boxes"].shape[0] == masks["tp"].shape[0]
        intersection_all_det, _ = _box_inter_union(
            instances["boxes"], instances["boxes"][torch.Tensor(masks["tp"]).bool()]
        )
        weights_det = (
            intersection_all_det.sum(axis=1) / box_area(instances["boxes"]) - masks["tp"]
        )  # subtract one to avoid counting intersection with itself when computing weights for TPs
    else:
        weights_det = torch.ones_like(instances["labels"])

    return weights_det


weightings: Dict[Optional[str], Callable[[BoxesType, MasksType, torch.Tensor], torch.Tensor]] = {
    None: no_weighting,
    "overlap": overlap,
    "pixel": pixel,
}


class CocoHardness(HardnessDefinition):
    """
    Parent class for hardness definitions based on the coco evaluation

    Args:
        cat_ids: Category ids for labels. Will default to coco ids if this is not set, which could cause errors.
        area_rng: Range of bounding box areas to be considered for evaluation (default is maximum range in coco).
        max_dets: Number of detections to be used in evaluation (extra detections will be trimmed).
        iou_threshold: Only boxes with an intersection over union larger than iou_threshold can be associated.
        weighting_fn: Weighting function to be used for boxes in evaluation when computing hardness.
            See hardest.pycocotools.weightings

    Attributes:
        cat_ids: Category ids for labels. Will default to coco ids if this is not set, which could cause errors.
        area_rng: Range of bounding box areas to be considered for evaluation (default is maximum range in coco).
        max_dets: Number of detections to be used in evaluation (extra detections will be trimmed).
        iou_threshold: Only boxes with an intersection over union larger than iou_threshold can be associated.
        weighting_fn: Weighting function to be used for boxes in evaluation when computing hardness.
            See hardest.pycocotools.weightings
    """

    def __init__(
        self,
        cat_ids: Set[int] = COCO_CAT_IDS,
        area_rng: Tuple[float, float] = (0.0, 10000000000.0),
        max_dets: int = 100,
        iou_threshold: float = 0.5,
        weighting_fn: Callable[[BoxesType, MasksType, torch.Tensor], torch.Tensor] = no_weighting,
    ):
        self.cat_ids: Set[int] = cat_ids
        self.area_rng: List[float] = list(area_rng)
        self.max_dets: int = max_dets
        self.iou_threshold: float = iou_threshold
        self.weighting_fn: Callable[[BoxesType, MasksType, torch.Tensor], torch.Tensor] = weighting_fn

    def __call__(self, detections: DetectionType, annotations: AnnotationTypeTorch, image: torch.Tensor) -> float:
        """
        Abstract method to compute hardness for image
        Args:
            detections: Detections for a single image in torchvision format
            annotations: Annotations for a single image in torchvision format
            image: Image for which hardness is being computed in torchvision format (C, H, W)

        Returns:
            Hardness for image
        """
        raise NotImplementedError

    def get_masks(
        self, detections: DetectionType, annotations: AnnotationTypeTorch, image: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        False positive and false negative mask for single image

        Largely copied from:
        https://github.com/pytorch/vision/blob/de31e4b8bf9b4a7e0668d19059a5ac4760dceee1/references/detection/coco_eval.py#L28

        Args:
            detections: Detections for a single image in torchvision format
            annotations: Annotations for a single image in torchvision format
            image: Image for which hardness is being computed in torchvision format (C, H, W)

        Returns:
            False positive binary mask
            False negative binary mask
        """
        with redirect_stdout(None):
            coco_gt = convert_to_coco_api(copy.deepcopy(annotations), image, categories=self.cat_ids)

            coco_eval = COCOeval(coco_gt, iouType="bbox")
            coco_eval.params.maxDets = [self.max_dets]
            coco_eval.params.iouThrs = [self.iou_threshold]

            results = prepare_for_coco_detection(detections)
            coco_dt = loadRes(coco_gt, results) if results else COCO()

            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = [0]
            coco_eval.params.areaRng = [
                self.area_rng,
            ]
            coco_eval.evaluate()
            eval_imgs = coco_eval.evalImgs
            fp_mask, fn_mask = accumulate(
                eval_imgs, list(self.cat_ids), n_gt=len(annotations["labels"]), n_det=len(detections["labels"])
            )
            return fp_mask, fn_mask

    def get_weights(
        self, detections: DetectionType, annotations: AnnotationTypeTorch, image: torch.Tensor
    ) -> Tuple[MasksType, torch.Tensor]:
        """
        Compute TP/FP/FN masks and associated weights for all box instances in an image
        Args:
            detections: Detections for a single image in torchvision format
            annotations: Annotations for a single image in torchvision format
            image: Image for which hardness is being computed in torchvision format (C, H, W)

        Returns:
            TP/FP/FN masks.
            Associated weights.
        """
        fp_mask, fn_mask = self.get_masks(detections, annotations, image)

        nfn = np.sum(fn_mask)

        instances: BoxesType = {
            "boxes": torch.cat((detections["boxes"], annotations["boxes"][torch.Tensor(fn_mask).bool()])),
            "labels": torch.cat((detections["labels"], annotations["labels"][torch.Tensor(fn_mask).bool()])),
        }

        masks: MasksType = {
            "tp": np.concatenate((~fp_mask, np.zeros(nfn, dtype=bool))),
            "fp": np.concatenate((fp_mask, np.zeros(nfn, dtype=bool))),
            "fn": np.concatenate((np.zeros_like(fp_mask), np.ones(nfn, dtype=bool))),
        }

        weights = self.weighting_fn(instances, masks, image)

        assert (weights >= 0).all()

        return masks, weights


class FP(CocoHardness):
    """
    Pycocotools false positives (weighted)
    """

    def __call__(self, detections: DetectionType, annotations: AnnotationTypeTorch, image: torch.Tensor) -> float:
        """
        Computes weighted false positive score

        Args:
            detections: Detections for a single image in torchvision format
            annotations: Annotations for a single image in torchvision format
            image: Image for which hardness is being computed in torchvision format (C, H, W)

        Returns:
            Hardness for image
        """
        masks, weights = self.get_weights(detections, annotations, image)
        return np.sum(masks["fp"] * weights.numpy()).item()


class FN(CocoHardness):
    """
    Pycocotools coco false negatives (weighted)
    """

    def __call__(self, detections: DetectionType, annotations: AnnotationTypeTorch, image: torch.Tensor) -> float:
        """
        Computes weighted false negative score

        Args:
            detections: Detections for a single image in torchvision format
            annotations: Annotations for a single image in torchvision format
            image: Image for which hardness is being computed in torchvision format (C, H, W)

        Returns:
            Hardness for image
        """
        masks, weights = self.get_weights(detections, annotations, image)
        return np.sum(masks["fn"] * weights.numpy()).item()


class TotalFalse(CocoHardness):
    """
    Pycocotools coco total false (weighted)
    """

    def __call__(self, detections: DetectionType, annotations: AnnotationTypeTorch, image: torch.Tensor) -> float:
        """
        Computes weighted total false score

        Args:
            detections: Detections for a single image in torchvision format
            annotations: Annotations for a single image in torchvision format
            image: Image for which hardness is being computed in torchvision format (C, H, W)

        Returns:
            Hardness for image
        """
        masks, weights = self.get_weights(detections, annotations, image)
        return np.sum(np.logical_or(masks["fn"], masks["fp"]) * weights.numpy()).item()


# The below functions are copied with minor modification from pycocotools and torchvision tutorials


def accumulate(
    eval_images: List[Dict[str, Any]], cat_ids: List[int], n_gt: int, n_det: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert eval images for every category into a mask for detections and annotations

    Copied from https://github.com/cocodataset/cocoapi/blob/6c3b394c07aed33fd83784a8bf8798059a1e9ae4/PythonAPI/pycocotools/cocoeval.py#L315
    Modified to remove code unrelated to producing mask and to produce fn

    Args:
        eval_images: Evaluation results from pycocotools
        cat_ids: List of category (label) ids
        n_gt: Number of annotations
        n_det: Number of detections

    Returns:
        False positive binary mask
        False negative binary mask

    """
    fp_mask = np.zeros((n_det,), dtype=bool)
    fn_mask = np.zeros((n_gt,), dtype=bool)

    for k, _ in enumerate(cat_ids):
        if eval_images[k] is None:
            continue

        dtm = eval_images[k]["dtMatches"]
        gtm = eval_images[k]["gtMatches"]
        dtIg = eval_images[k]["dtIgnore"]
        gtIg = eval_images[k]["gtIgnore"]
        if len(eval_images[k]["dtIds"]) > 0:
            fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg))
            fp_mask[np.array(eval_images[k]["dtIds"]) - 1] = fps
        if len(eval_images[k]["gtIds"]) > 0:
            fns = np.logical_and(np.logical_not(gtm), np.logical_not(gtIg))
            fn_mask[np.array(eval_images[k]["gtIds"]) - 1] = fns
    return fp_mask, fn_mask


class CocoDetection(TypedDict):
    image_id: int
    category_id: float
    bbox: List[float]
    score: float
    area: Optional[float]
    id: Optional[int]
    iscrowd: Optional[int]


def prepare_for_coco_detection(predictions: DetectionType) -> List[CocoDetection]:
    """
    Convert torchvision detections into pycocotools evaluation format

    Bounding boxes are converted from xyxy to xywh. Other attributes are converted to list from tensor and renamed.
    Copied from: https://github.com/pytorch/vision/blob/de31e4b8bf9b4a7e0668d19059a5ac4760dceee1/references/detection/coco_eval.py#L67

    Args:
        predictions: Detections for a single image in torchvision format

    Returns:
        pycocotools format detections
    """
    labels = predictions["labels"].tolist()
    n_det = len(labels)
    boxes = predictions["boxes"]
    if n_det > 0:
        boxes = box_convert(boxes, "xyxy", "xywh").tolist()
    scores = predictions["scores"].tolist()

    return [
        {
            "image_id": 0,
            "category_id": labels[k],
            "bbox": box,
            "score": scores[k],
            "iscrowd": None,
            "id": None,
            "area": None,
        }
        for k, box in enumerate(boxes)
    ]


def loadRes(coco_gt: COCO, anns: List[CocoDetection]) -> COCO:
    """
    Load result detections and return a result api object.

    Modified from:
    https://github.com/cocodataset/cocoapi/blob/6c3b394c07aed33fd83784a8bf8798059a1e9ae4/PythonAPI/pycocotools/coco.py#L297
    to avoid loading res from hard disk
    Args:
        coco_gt: annotation api object
        anns: the list of prediction results

    Returns:
        result api object
    """
    res = COCO()
    res.dataset["images"] = [img for img in coco_gt.dataset["images"]]

    assert type(anns) == list, "results in not an array of objects"
    annsImgIds = [ann["image_id"] for ann in anns]
    assert set(annsImgIds) == (
        set(annsImgIds) & set(coco_gt.getImgIds())
    ), "Results do not correspond to current coco set"

    if "bbox" in anns[0] and not anns[0]["bbox"] == []:
        res.dataset["categories"] = copy.deepcopy(coco_gt.dataset["categories"])
        for id, ann in enumerate(anns):
            bb = ann["bbox"]
            ann["area"] = bb[2] * bb[3]
            ann["id"] = id + 1
            ann["iscrowd"] = 0
    else:
        raise ValueError("bbox must be present in annotations")

    res.dataset["annotations"] = anns
    res.createIndex()
    return res


def convert_to_coco_api(target: AnnotationTypeTorch, img: torch.Tensor, categories: Set[int]) -> COCO:
    """
    Convert annotatinos to coco api

    Copied from: https://github.com/pytorch/vision/blob/de31e4b8bf9b4a7e0668d19059a5ac4760dceee1/references/detection/coco_utils.py#L146
    with modification to remove unused segmentation and keypoints and to work on annotations
    Args:
        target: Annotations for a single image in torchvision format
        img: Image for which hardness is being computed in torchvision format (C, H, W)
        categories: List of category (label) ids

    Returns:
        result api object

    """
    coco_ds = COCO()
    # annotation IDs need to start at 1, not 0, see torchvision issue #1530
    ann_id = 1
    dataset: Dict[str, List[Dict[str, Any]]] = {"images": [], "categories": [], "annotations": []}
    image_id = 0
    img_dict = {}
    img_dict["id"] = image_id
    img_dict["height"] = img.shape[-2]
    img_dict["width"] = img.shape[-1]
    dataset["images"].append(img_dict)
    num_objs = len(target["boxes"])
    if num_objs > 0:
        bboxes = box_convert(target["boxes"], "xyxy", "xywh").tolist()
        labels = target["labels"].tolist()
        areas = box_area(target["boxes"]).tolist()
        iscrowd = target["iscrowd"].tolist()
    for i in range(num_objs):
        assert labels[i] in categories
        ann = {
            "image_id": image_id,
            "bbox": bboxes[i],
            "category_id": labels[i],
            "area": areas[i],
            "iscrowd": iscrowd[i],
            "id": ann_id,
        }
        dataset["annotations"].append(ann)
        ann_id += 1
    dataset["categories"] = [{"id": i} for i in sorted(categories)]
    coco_ds.dataset = dataset
    with contextlib.redirect_stdout(None):
        coco_ds.createIndex()
    return coco_ds


hardness_definition_factory: Dict[str, HardnessDefinition] = {
    "fp": FP(),
    "fn": FN(),
    "false": TotalFalse(),
    "fp_pixel": FP(weighting_fn=pixel),
    "fn_pixel": FN(weighting_fn=pixel),
    "false_pixel": TotalFalse(weighting_fn=pixel),
    "fp_overlap": FP(weighting_fn=overlap),
    "fn_overlap": FN(weighting_fn=overlap),
    "false_overlap": TotalFalse(weighting_fn=overlap),
}

hardness_definition_factory_av_simple: Dict[str, HardnessDefinition] = {
    "fp": FP(cat_ids=AV_SIMPLE_CAT_IDS),
    "fn": FN(cat_ids=AV_SIMPLE_CAT_IDS),
    "false": TotalFalse(cat_ids=AV_SIMPLE_CAT_IDS),
    "fp_pixel": FP(weighting_fn=pixel, cat_ids=AV_SIMPLE_CAT_IDS),
    "fn_pixel": FN(weighting_fn=pixel, cat_ids=AV_SIMPLE_CAT_IDS),
    "false_pixel": TotalFalse(weighting_fn=pixel, cat_ids=AV_SIMPLE_CAT_IDS),
    "fp_overlap": FP(weighting_fn=overlap, cat_ids=AV_SIMPLE_CAT_IDS),
    "fn_overlap": FN(weighting_fn=overlap, cat_ids=AV_SIMPLE_CAT_IDS),
    "false_overlap": TotalFalse(weighting_fn=overlap, cat_ids=AV_SIMPLE_CAT_IDS),
}
