from abc import ABC
from abc import abstractmethod
from typing import List
from typing import Tuple

import numpy as np
import torch
from torch.distributions import Bernoulli

from hardest.hardness_definitions import DetectionType
from hardest.utils import AnnotationTypeTorch
from hardest.utils import filter_pytorch_detections_by_score


class GroundTruthSampler(ABC):
    """
    Abstract class for sampling based pseudo ground truth generation.
    """

    @abstractmethod
    def sample(self, detections: DetectionType, n_samples: int) -> List[AnnotationTypeTorch]:
        """
        Sample the pseudo ground truth.
        Args:
            detections: detections for a single image
            n_samples: number of samples to generate

        Returns:
            Sampled pseudo ground truth in torchvision format
        """
        pass


class ScoreSampling(GroundTruthSampler):
    """
    Score sampling pseudo ground truth generation.

    Produces samples of pseudo ground truth using detections which are included based on a Bernouilli sampled mask from
    the detection scores.
    """

    def sample(self, detections: DetectionType, n_samples: int) -> List[AnnotationTypeTorch]:
        """
        Sample the pseudo ground truth.
        Args:
            detections: detections for a single image
            n_samples: number of samples to generate

        Returns:
            Sampled pseudo ground truth in torchvision format
        """
        keep_distribution = Bernoulli(probs=detections["scores"])
        keep_mask = keep_distribution.sample((n_samples,)).bool()
        psuedo_ground_truth: List[AnnotationTypeTorch] = [
            {
                "boxes": detections["boxes"][keep_sample],
                "labels": detections["labels"][keep_sample],
                "iscrowd": torch.zeros_like(detections["labels"][keep_sample], dtype=torch.bool),
            }
            for keep_sample in keep_mask
        ]
        return psuedo_ground_truth


class DiscreteCommittee(ABC):
    """
    A discrete distribution for pseudo ground truth using detections from several detectors
    """

    @abstractmethod
    def probability_mass(self, detection_list: List[DetectionType]) -> Tuple[List[AnnotationTypeTorch], List[float]]:
        pass


class UniformCommittee(DiscreteCommittee):
    """
    A uniform distribution for pseudo ground truth using detections from several detectors

    Args:
        score_threshold: The detector score threshold (proposed detections with a score greater than this are
            are considered confirmed).
    """

    def __init__(self, score_threshold: float = 0.5):
        self.score_threshold: float = score_threshold

    def probability_mass(self, detection_list: List[DetectionType]) -> Tuple[List[AnnotationTypeTorch], List[float]]:
        """
        Probability mass of detections in discrete distribution.
        Args:
            detection_list: List of detections for a single image, list over committee members.

        Returns:
            Pseudo ground truth detections in torchvision format
            List of probability mass for each detection
        """
        detection_list = [filter_pytorch_detections_by_score(det, self.score_threshold) for det in detection_list]
        psuedo_ground_truth: List[AnnotationTypeTorch] = [
            {
                "boxes": detections["boxes"],
                "labels": detections["labels"],
                "iscrowd": torch.zeros_like(detections["labels"], dtype=torch.bool),
            }
            for detections in detection_list
        ]
        return psuedo_ground_truth, len(detection_list) * [1 / len(detection_list)]


class ScoreSamplingUniformCommittee(ABC):
    """
    Uniform committee with score sampling for the committee members.
    """

    def sample(self, detection_list: List[DetectionType], n_samples: int) -> List[AnnotationTypeTorch]:
        """
        Sample the pseudo ground truth.
        Args:
            detection_list: List of detections for a single image, list over committee members.
            n_samples: number of samples to generate

        Returns:
            Sampled pseudo ground truth in torchvision format
        """
        detections, probability_mass = self.underlying_committee_model(detection_list)
        sample_mask = np.random.choice(len(detections), size=n_samples, p=probability_mass)
        sampled_detections = [detections[idx] for idx in sample_mask]
        return [ScoreSampling().sample(det, n_samples=n_samples)[0] for det in sampled_detections]

    def underlying_committee_model(
        self, detection_list: List[DetectionType]
    ) -> Tuple[List[DetectionType], List[float]]:
        """
        Probability mass of detections in discrete distribution for underlying committe.

        Note that this is NOT the distribution for this pseudo ground truth model, since a continious sampling step is
        applied to the results of this method.
        Args:
            detection_list: List of detections for a single image, list over committee members.

        Returns:
            Pseudo ground truth detections in torchvision format
            List of probability mass for each detection
        """
        return detection_list, (np.ones(len(detection_list)) / len(detection_list)).tolist()
