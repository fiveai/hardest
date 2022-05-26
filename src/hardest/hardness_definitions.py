from abc import ABC
from abc import abstractmethod

import torch

from hardest.detectors import DetectionType
from hardest.utils import AnnotationTypeTorch


class HardnessDefinition(ABC):
    """
    Abstract class for hardness definitions
    """

    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, detections: DetectionType, annotations: AnnotationTypeTorch, image: torch.Tensor) -> float:
        """
        Calculate hardness of image
        Args:
            detections: Detections for a single image in torchvision format
            annotations: Annotations for a single image in torchvision format
            image: Image for which hardness is being computed in torchvision format (C, H, W)

        Returns:
            Hardness

        """
        raise NotImplementedError
