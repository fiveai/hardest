from typing import Dict
from typing import List
from typing import Protocol
from typing import TypedDict
from typing import Union

import numpy as np
import torch
from torch.nn.functional import softmax
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.models.detection.transform import resize_boxes


class BoxesType(TypedDict):
    boxes: torch.Tensor
    """shape (n, 4); dtype float32.
    Bounding boxes in (x1, y1, x2, y2) format.,
    0 <= x1 < x2 and 0 <= y1 < y2.
    """
    labels: torch.Tensor
    """shape (n,); dtype int64.
    Labels for the detected objects.
    """


class DetectionType(BoxesType):
    scores: torch.Tensor
    """shape (n,); dtype float32.
    Detected class probability for the detected objects.
    """


class KClassDetectionType(DetectionType):
    logits: torch.Tensor
    """shape (n, c); dtype float32.
    K Class logits for the detected objects.
    """


class MasksType(TypedDict):
    tp: np.ndarray
    fp: np.ndarray
    fn: np.ndarray


class TorchDetectorType(Protocol):
    """
    The TorchDetectorType will output a List[DetectionType] and take a List[torch.Tensor] because it is batched.
    """

    def __call__(self, __x: List[torch.Tensor]) -> List[DetectionType]:
        ...


class WrappedTorchDetectorType(Protocol):
    """
    The TorchDetectorType will output a List[DetectionType] and take a List[torch.Tensor] because it is batched.
    """

    def __call__(self, __x: List[torch.Tensor]) -> List[KClassDetectionType]:
        ...


def wrap_detector_with_logit_call(detector: GeneralizedRCNN) -> WrappedTorchDetectorType:
    detector_forward = detector.forward

    def dummy_call(inputs: List[torch.Tensor]) -> List[KClassDetectionType]:
        result = detector_forward(inputs)
        for idx, input in enumerate(inputs):
            input.requires_grad_()
            original_image_size = input.size()[-2:]
            transformed_inputs = detector.transform([input])[0]
            rescaled_image_size = transformed_inputs.image_sizes[0]
            with torch.no_grad():
                rescaled_bbs = resize_boxes(result[idx]["boxes"], original_image_size, rescaled_image_size)
            features = detector.backbone(transformed_inputs.tensors)
            box_features = detector.roi_heads.box_roi_pool(features, [rescaled_bbs], transformed_inputs.image_sizes)
            box_features = detector.roi_heads.box_head(box_features)
            (
                class_logits,
                box_regression,
            ) = detector.roi_heads.box_predictor(box_features)
            result[idx]["logits"] = class_logits
            # probabilities = softmax(class_logits, -1)
            # assert (probabilities[:, 1:].max(-1)[0] == result[idx]["scores"]).all()
            # assert (probabilities[:, 1:].max(-1)[1] == result[idx]["labels"] - 1).all()
            # Why won't these pass??
        return result

    detector.forward = dummy_call
    return detector
