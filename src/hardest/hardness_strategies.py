import itertools
from abc import ABC
from abc import abstractmethod
from typing import Callable
from typing import Iterable
from typing import List
from typing import Literal
from typing import Union

import numpy as np
import torch
from scipy.special import entr
from scipy.stats import entropy
from torch.distributions import Bernoulli
from torch.nn.functional import softmax
from tqdm import tqdm

from hardest.detectors import BoxesType
from hardest.detectors import DetectionType
from hardest.detectors import KClassDetectionType
from hardest.detectors import MasksType
from hardest.hardness_definitions import HardnessDefinition
from hardest.pseudo_ground_truth_models import DiscreteCommittee
from hardest.pseudo_ground_truth_models import GroundTruthSampler
from hardest.pseudo_ground_truth_models import ScoreSampling
from hardest.pseudo_ground_truth_models import ScoreSamplingUniformCommittee
from hardest.pycocotools import no_weighting
from hardest.utils import AnnotationTypeTorch
from hardest.utils import filter_pytorch_detections_by_score


class SupervisedHardnessCalculation:
    """
    Computes hardness according to a specific definition for a 2d object detection dataset based on annotations.

    Args:
        hardness_definition: Definition of hardness to be computed (function of detections, targets and image)
        score_threshold: The detector score threshold (proposed detections with a score greater than this are
            are considered confirmed).

    Attributes:
        hardness_definition: Definition of hardness to be computed (function of detections, targets and image)
        score_threshold: The detector score threshold (proposed detections with a score greater than this are
            are considered confirmed).
    """

    def __init__(
        self,
        hardness_definition: HardnessDefinition,
        score_threshold: float = 0.5,
    ):
        self.score_threshold: float = score_threshold
        self.hardness_definition: HardnessDefinition = hardness_definition

    def calculate_hardness(self, detection: DetectionType, target: AnnotationTypeTorch, image: torch.Tensor) -> float:
        """
        Calculate the hardness for a single image
        Args:
            detection: Detections for a single image in torchvision format
            target: Annotations for a single image in torchvision format
            image: Image for which hardness is being computed in torchvision format (C, H, W)

        Returns:
            Hardness as a float
        """
        detection = filter_pytorch_detections_by_score(detection, self.score_threshold)
        return self.hardness_definition(detection, target, image)

    def eval_dataset(
        self,
        detections: Iterable[DetectionType],
        targets: Iterable[AnnotationTypeTorch],
        images: Iterable[torch.Tensor],
    ) -> List[float]:
        """
        Scores an entire dataset by hardness.
        Args:
            detections: List of detections in torchvision format
            targets: List of targets in torchvision format
            images: List of images in torchvision format (C, H, W) (can be a generator if dataset is large)

        Returns:
            Hardness of dataset
        """
        with tqdm(detections, desc=f"Calculating hardness with {self}, fraction of dataset processed") as bar:
            return list(itertools.starmap(self.calculate_hardness, zip(bar, targets, images)))


class UnsupervisedHardnessCalculation(ABC):
    """
    Abstract class for hardness estimation strategies when targets/annotations are not available
    """

    @abstractmethod
    def calculate_hardness(self, detection: DetectionType, image: torch.Tensor) -> float:
        """
        Calculate the hardness for a single image
        Args:
            detection: Detections for a single image in torchvision format
            image: Image for which hardness is being computed in torchvision format (C, H, W)

        Returns:
            Estimated hardness
        """
        raise NotImplementedError

    def eval_dataset(self, detections: Iterable[DetectionType], images: Iterable[torch.Tensor]) -> List[float]:
        """
        Scores an entire dataset by hardness.
        Args:
            detections: List of detections in torchvision format (should be obtained by running the detector with a
                low score threshold as low score detections are required for the algorithm to work correctly).
            images: List of images in torchvision format (C, H, W) (can be a generator if dataset is large)

        Returns:
            Estimated hardness of dataset
        """
        with tqdm(detections, desc=f"Calculating hardness with {self}, fraction of dataset processed") as bar:
            return list(itertools.starmap(self.calculate_hardness, zip(bar, images)))


class UncertaintyMeasure(ABC):
    """
    Abstract class for uncertainty functions from detected boxes in an image.

    Args:
        reduction: A function to calculate the hardness for the image based on the array of hardnesses for
            individual detected bounding boxes.

    Attributes:
        reduction: A function to calculate the hardness for the image based on the array of hardnesses for
            individual detected bounding boxes.
    """

    def __init__(
        self,
        reduction: Callable[[np.ndarray], np.ndarray] = np.sum,
    ):
        self.reduction: Callable[[np.ndarray], np.ndarray] = reduction

    @abstractmethod
    def __call__(self, detection: DetectionType) -> float:
        pass


class KClassUncertaintyMeasure(ABC):
    """
    Abstract class for uncertainty functions from detected boxes in an image.

    Args:
        reduction: A function to calculate the hardness for the image based on the array of hardnesses for
            individual detected bounding boxes.

    Attributes:
        reduction: A function to calculate the hardness for the image based on the array of hardnesses for
            individual detected bounding boxes.
    """

    def __init__(
        self,
        reduction: Callable[[np.ndarray], np.ndarray] = np.sum,
    ):
        self.reduction: Callable[[np.ndarray], np.ndarray] = reduction

    @abstractmethod
    def __call__(self, detection: KClassDetectionType) -> float:
        pass


class UncertaintyMeasureHardnessCalculation(UnsupervisedHardnessCalculation):
    """
    Unsupervised hardness calculation with a uncertainty measure which is a function of detections.

    Parent class for unsupervised hardness calculations where the calculated hardness does not aim to estimate a
    specific definition of hardness which could be calculated exactly if the targets were available. In this sense they
    are "generic" to any hardness definition.

    Args:
        uncertainty_measure: A function from detections to hardness
        score_threshold: The detector score threshold (proposed detections with a score greater than this are
            are considered confirmed).

    Attributes:
        uncertainty_measure: A function from detections to hardness
        score_threshold: The detector score threshold (proposed detections with a score greater than this are
            are considered confirmed).
    """

    def __init__(
        self,
        uncertainty_measure: UncertaintyMeasure,
        score_threshold: float = 0.5,
    ):
        super().__init__()
        self.uncertainty_measure: UncertaintyMeasure = uncertainty_measure
        self.score_threshold: float = score_threshold

    def calculate_hardness(self, detection: DetectionType, image: torch.Tensor) -> float:
        """
        Calculate the hardness for a single image
        Args:
            detections: List of detections in torchvision format (should be obtained by running the detector with a
                low score threshold as low score detections are required for the algorithm to work correctly).
            image: Image for which hardness is being computed in torchvision format (C, H, W)

        Returns:
            Estimated hardness
        """
        detection = filter_pytorch_detections_by_score(detection, self.score_threshold)
        return self.uncertainty_measure(detection)


class KClassUncertaintyMeasureHardnessCalculation(ABC):
    """
    Unsupervised hardness calculation with a uncertainty measure which is a function of detections.

    Parent class for unsupervised hardness calculations where the calculated hardness does not aim to estimate a
    specific definition of hardness which could be calculated exactly if the targets were available. In this sense they
    are "generic" to any hardness definition.

    Args:
        uncertainty_measure: A function from detections to hardness
        score_threshold: The detector score threshold (proposed detections with a score greater than this are
            are considered confirmed).

    Attributes:
        uncertainty_measure: A function from detections to hardness
        score_threshold: The detector score threshold (proposed detections with a score greater than this are
            are considered confirmed).
    """

    def __init__(
        self,
        uncertainty_measure: KClassUncertaintyMeasure,
        score_threshold: float = 0.5,
    ):
        super().__init__()
        self.uncertainty_measure: KClassUncertaintyMeasure = uncertainty_measure
        self.score_threshold: float = score_threshold

    def calculate_hardness(self, detection: KClassDetectionType, image: torch.Tensor) -> float:
        """
        Calculate the hardness for a single image
        Args:
            detections: List of detections in torchvision format (should be obtained by running the detector with a
                low score threshold as low score detections are required for the algorithm to work correctly).
            image: Image for which hardness is being computed in torchvision format (C, H, W)

        Returns:
            Estimated hardness
        """
        detection = filter_pytorch_detections_by_score(detection, self.score_threshold)
        return self.uncertainty_measure(detection)

    def eval_dataset(self, detections: Iterable[KClassDetectionType], images: Iterable[torch.Tensor]) -> List[float]:
        """
        Scores an entire dataset by hardness.
        Args:
            detections: List of detections in torchvision format (should be obtained by running the detector with a
                low score threshold as low score detections are required for the algorithm to work correctly).
            images: List of images in torchvision format (C, H, W) (can be a generator if dataset is large)

        Returns:
            Estimated hardness of dataset
        """
        with tqdm(detections, desc=f"Calculating hardness with {self}, fraction of dataset processed:") as bar:
            return list(itertools.starmap(self.calculate_hardness, zip(bar, images)))


class BinaryDempster(UncertaintyMeasure):
    r"""
    Dempster Shafer (evidential deep learning) uncertainty measure.

    Calculates hardness using evidential deep learning based on a modified version of the method provided in:
    https://arxiv.org/abs/1806.01768

    We modify the method to only consider two classes (detected class vs all other classes).
    Hence unassigned belief is calculated as:
    .. math::
        u = \sum_{b \in B} \frac{|K|}{\sum_{k \in K} 1 + \exp s_{b,k}}
    with  |K| = 2 , where s_{b,k} are the
    class scores for box $b$.
    """

    def __call__(self, detection: DetectionType) -> float:
        r"""
        Calculates uncertainty, i.e. unassigned belief, in the dempster shafer framework.
        .. math::
            u = \sum_{b \in B} \frac{|K|}{\sum_{k \in K} 1 + \exp s_{b,k}}
        with  |K| = 2 , where s_{b,k} are the
        class scores for box $b$.
        Args:
            detection: Detections for a single image in torchvision format

        Returns:
            Dempster score

        """
        p = detection["scores"]
        Z = np.sum([p / (1 - p), (1 - p) / p], axis=0)
        DS = 2 / (2 + Z)
        return self.reduction(DS.numpy()).item()


class Dempster(KClassUncertaintyMeasure):
    r"""
    Dempster Shafer (evidential deep learning) uncertainty measure.

    Calculates hardness using evidential deep learning based on a modified version of the method provided in:
    https://arxiv.org/abs/1806.01768

    We modify the method to only consider two classes (detected class vs all other classes).
    Hence unassigned belief is calculated as:
    .. math::
        u = \sum_{b \in B} \frac{|K|}{\sum_{k \in K} 1 + \exp s_{b,k}}
    with  |K| = 2 , where s_{b,k} are the
    class scores for box $b$.
    """

    def __call__(self, detection: KClassDetectionType) -> float:
        r"""
        Calculates uncertainty, i.e. unassigned belief, in the dempster shafer framework.
        .. math::
            u = \sum_{b \in B} \frac{|K|}{\sum_{k \in K} 1 + \exp s_{b,k}}
        with  |K| = 2 , where s_{b,k} are the
        class scores for box $b$.
        Args:
            detection: Detections for a single image in torchvision format

        Returns:
            Dempster score

        """
        logits = detection["logits"].numpy()
        K = logits.shape[-1]
        Z = np.sum(np.exp(logits), axis=-1)
        DS = K / (K + Z)
        return self.reduction(DS).item() if len(logits) > 0 else float(0.0)


def BinaryDempsterHardnessCalculation(
    reduction: Callable[[np.ndarray], np.ndarray] = np.sum, score_threshold: float = 0.5
) -> UncertaintyMeasureHardnessCalculation:
    """
    Convenience function for hardness calculation with Dempster uncertainty measure

    Args:
        reduction: A function to calculate the hardness for the image based on the array of hardnesses for
            individual detected bounding boxes.
        score_threshold: The detector score threshold (proposed detections with a score greater than this are
            are considered confirmed).

    Returns:
        Uncertainty measure hardness calculation

    """
    return UncertaintyMeasureHardnessCalculation(BinaryDempster(reduction), score_threshold)


def DempsterHardnessCalculation(
    reduction: Callable[[np.ndarray], np.ndarray] = np.sum, score_threshold: float = 0.5
) -> KClassUncertaintyMeasureHardnessCalculation:
    """
    Convenience function for hardness calculation with Dempster uncertainty measure

    Args:
        reduction: A function to calculate the hardness for the image based on the array of hardnesses for
            individual detected bounding boxes.
        score_threshold: The detector score threshold (proposed detections with a score greater than this are
            are considered confirmed).

    Returns:
        Uncertainty measure hardness calculation

    """
    return KClassUncertaintyMeasureHardnessCalculation(Dempster(reduction), score_threshold)


class MaxUncertainty(UncertaintyMeasure):
    """
    Hardness defined as 1-score of the most uncertain bounding box in the image

    Defined in from https://arxiv.org/abs/1801.05124 sec 3.1
    """

    def __call__(self, detection: DetectionType) -> float:
        """
        1-score of the most uncertain bounding box
        Args:
            detection: Detections for a single image in torchvision format

        Returns:
            Most uncertain bounding box uncertainty

        """
        p = detection["scores"]
        uncertainty = 1 - p
        return max(uncertainty).item() if len(p) > 0 else float(0.0)


def MaxUncertaintyCalculation(
    reduction: Callable[[np.ndarray], np.ndarray] = np.sum, score_threshold: float = 0.5
) -> UncertaintyMeasureHardnessCalculation:
    """
    Convenience function for hardness calculation with MaxUncertainty uncertainty measure

    Args:
        reduction: A function to calculate the hardness for the image based on the array of hardnesses for
            individual detected bounding boxes.
        score_threshold: The detector score threshold (proposed detections with a score greater than this are
            are considered confirmed).

    Returns:
        Uncertainty measure hardness calculation

    """
    return UncertaintyMeasureHardnessCalculation(MaxUncertainty(reduction), score_threshold)


class BinaryEntropy(UncertaintyMeasure):
    """
    Hardness defined by the binary entropy for the score for each bounding box
    """

    def __call__(self, detection: DetectionType) -> float:
        r"""
        .. math::
            \sum_{b \in B} \sum_{k \in K} - p_{b,k} \log p_{b,k}
        Args:
            detection: Detections for a single image in torchvision format

        Returns:
            Binary entropy
        """
        p = detection["scores"].numpy()
        return float(self.reduction(entr(p) + entr(1 - p)))


class Entropy(KClassUncertaintyMeasure):
    """
    Hardness defined by the entropy for the k class probabilities for each bounding box
    """

    def __call__(self, detection: KClassDetectionType) -> float:
        r"""
        .. math::
            \sum_{b \in B} \sum_{k \in K} - p_{b,k} \log p_{b,k}
        Args:
            detection: Detections for a single image in torchvision format

        Returns:
            Binary entropy
        """
        p = softmax(detection["logits"], -1).numpy()
        return float(self.reduction(entropy(p, axis=-1))) if len(p) > 0 else float(0.0)


def EntropyCalculation(
    reduction: Callable[[np.ndarray], np.ndarray] = np.sum, score_threshold: float = 0.5
) -> KClassUncertaintyMeasureHardnessCalculation:
    """
    Convenience function for hardness calculation with Entropy uncertainty measure

    Args:
        reduction: A function to calculate the hardness for the image based on the array of hardnesses for
            individual detected bounding boxes.
        score_threshold: The detector score threshold (proposed detections with a score greater than this are
            are considered confirmed).

    Returns:
        Uncertainty measure hardness calculation

    """
    return KClassUncertaintyMeasureHardnessCalculation(Entropy(reduction), score_threshold)


def BinaryEntropyCalculation(
    reduction: Callable[[np.ndarray], np.ndarray] = np.sum, score_threshold: float = 0.5
) -> UncertaintyMeasureHardnessCalculation:
    """
    Convenience function for hardness calculation with Entropy uncertainty measure

    Args:
        reduction: A function to calculate the hardness for the image based on the array of hardnesses for
            individual detected bounding boxes.
        score_threshold: The detector score threshold (proposed detections with a score greater than this are
            are considered confirmed).

    Returns:
        Uncertainty measure hardness calculation

    """
    return UncertaintyMeasureHardnessCalculation(BinaryEntropy(reduction), score_threshold)


class MonteCarloHardnessCalculation(UnsupervisedHardnessCalculation):
    """
    Unsupervised hardness calculation using Monte Carlo integration with stochastic ground truth model.

    A targeted hardness calculation where Monte Carlo simulation is used with a stochastic model for ground truth to
    approximate the hardness. The stochastic model is required because in this setting the ground truth is _uncertain_.

    Args:
        hardness_definition: Definition of hardness to be computed (function of detections, targets and image)
        pseudo_ground_truth_model: Produces samples of pseudo ground truth based on detections.
        n_samples: Number of samples to make from pseudo ground truth model, which are used to compute hardness
            samples
        score_threshold: The detector score threshold (proposed detections with a score greater than this are
            are considered confirmed).
        reduction: Reduction function to be applied to samples of hardness, i.e. np.mean will yield expectation.
            A combination of np.mean and np.std can be used to obtain a lower bound

    Attributes:
        hardness_definition: Definition of hardness to be computed (function of detections, targets and image)
        pseudo_ground_truth_model: Produces samples of pseudo ground truth based on detections.
        n_samples: Number of samples to make from pseudo ground truth model, which are used to compute hardness
            samples
        score_threshold: The detector score threshold (proposed detections with a score greater than this are
            are considered confirmed).
        reduction: Reduction function to be applied to samples of hardness, i.e. np.mean will yield expectation.
            A combination of np.mean and np.std can be used to obtain a lower bound
    """

    def __init__(
        self,
        hardness_definition: HardnessDefinition,
        pseudo_ground_truth_model: GroundTruthSampler,
        n_samples: int,
        score_threshold: float = 0.5,
        reduction: Callable[[np.ndarray], float] = np.mean,
    ):
        self.n_samples: int = n_samples
        self.pseudo_ground_truth_model: GroundTruthSampler = pseudo_ground_truth_model
        self.reduction: Callable[[np.ndarray], float] = reduction
        self.score_threshold: float = score_threshold
        self.hardness_definition: HardnessDefinition = hardness_definition

    def calculate_hardness(self, detection: DetectionType, image: torch.Tensor) -> float:
        """
        Estimate hardness for a single image

        Args:
            detection: Detections for a single image in torchvision format
            image: Image for which hardness is being computed in torchvision format (C, H, W)

        Returns:
            Hardness
        """
        pseudo_ground_truth = self.pseudo_ground_truth_model.sample(detection, self.n_samples)

        filtered_detection = filter_pytorch_detections_by_score(detection, self.score_threshold)

        hardness_samples = [self.hardness_definition(filtered_detection, gt, image) for gt in pseudo_ground_truth]

        return self.reduction(np.array(hardness_samples))


class ScoreSamplingHardnessCalculation(MonteCarloHardnessCalculation):
    """
    Uses the score sampling pseudo ground truth model with Monte Carlo sampling to estimate hardness.

    Score sampling uses a Bernoulli random variable parameterised by bounding box scores to randomly discard boxes from
    the sampled pseudo ground truth. Runtime is usually approximately n_samples x hardness_definition time

    Args:
        hardness_definition: Definition of hardness to be computed (function of detections, targets and image)
        n_samples: Number of samples to make from pseudo ground truth model, which are used to compute hardness
            samples
        score_threshold: The detector score threshold (proposed detections with a score greater than this are
            are considered confirmed).
        reduction: Reduction function to be applied to samples of hardness, i.e. np.mean will yield expectation.
            A combination of np.mean and np.std can be used to obtain a lower bound

    Attributes:
        hardness_definition: Definition of hardness to be computed (function of detections, targets and image)
        n_samples: Number of samples to make from pseudo ground truth model, which are used to compute hardness
            samples
        score_threshold: The detector score threshold (proposed detections with a score greater than this are
            are considered confirmed).
        reduction: Reduction function to be applied to samples of hardness, i.e. np.mean will yield expectation.
            A combination of np.mean and np.std can be used to obtain a lower bound
        pseudo_ground_truth_model: Produces samples of pseudo ground truth based on detections.
    """

    def __init__(
        self,
        hardness_definition: HardnessDefinition,
        n_samples: int,
        score_threshold: float = 0.5,
        reduction: Callable[[np.ndarray], float] = np.mean,
    ):
        pseudo_ground_truth_model = ScoreSampling()
        super().__init__(hardness_definition, pseudo_ground_truth_model, n_samples, score_threshold, reduction)


class FastScoreSamplingHardnessCalculation(UnsupervisedHardnessCalculation):
    """
    A highly efficient alternative to ScoreSamplingHardness which avoids association of ground truth and detections.

    Can be targeted to some specific hardness definitions without actually calling the hardness definition function.

    False negatives and false positives are computed by assuming that the sampled pseudo ground truth will always
    associate to the detection from which it was generated, and hence the expensive association algorithm can be
    avoided.

    Args:
        n_samples: Number of samples to make from pseudo ground truth model, which are used to compute hardness
            samples
        hardness_name: The name of the hardness definition to be targeted. Must be "fp" (false positives),
            "fn" (false negatives) or "false".
        weighting_fn: Weighting function to be used for boxes in evaluation when computing hardness.
            See hardest.pycocotools.weightings for more details.
        score_threshold: The detector score threshold (proposed detections with a score greater than this are
            are considered confirmed).
        reduction: Reduction function to be applied to samples of hardness, i.e. np.mean will yield expectation.
            A combination of np.mean and np.std can be used to obtain a lower bound

    Attributes:
        n_samples: Number of samples to make from pseudo ground truth model, which are used to compute hardness
            samples
        hardness_name: The name of the hardness definition to be targeted. Must be "fp" (false positives),
            "fn" (false negatives) or "false".
        weighting_fn: Weighting function to be used for boxes in evaluation when computing hardness.
            See hardest.pycocotools.weightings for more details.
        score_threshold: The detector score threshold (proposed detections with a score greater than this are
            are considered confirmed).
        reduction: Reduction function to be applied to samples of hardness, i.e. np.mean will yield expectation.
            A combination of np.mean and np.std can be used to obtain a lower bound
    """

    def __init__(
        self,
        n_samples: int,
        hardness_name: Union[Literal["fp"], Literal["fn"], Literal["false"]],
        weighting_fn: Callable[[BoxesType, MasksType, torch.Tensor], torch.Tensor] = no_weighting,
        score_threshold: float = 0.5,
        reduction: Callable[[np.ndarray], float] = np.mean,
    ):
        super().__init__()
        self.n_samples: int = n_samples
        self.reduction: Callable[[np.ndarray], float] = reduction
        self.score_threshold: float = score_threshold
        self.weighting_fn: Callable[[BoxesType, MasksType, torch.Tensor], torch.Tensor] = weighting_fn
        self.hardness_name: Union[Literal["fp"], Literal["fn"], Literal["false"]] = hardness_name

    def calculate_hardness(self, detection: DetectionType, image: torch.Tensor) -> float:
        """
        Estimate hardness for a single image
        Args:
            detection: Detections for a single image in torchvision format
            image: Image for which hardness is being computed in torchvision format (C, H, W)

        Returns:
            Hardness

        """
        keep_distribution = Bernoulli(probs=detection["scores"])
        true = keep_distribution.sample((self.n_samples,)).bool()

        instances: BoxesType = detection

        detection_mask = detection["scores"] >= self.score_threshold

        masks: MasksType = {
            "tp": torch.logical_and(detection_mask, true).numpy(),
            "fp": torch.logical_and(detection_mask, ~true).numpy(),
            "fn": torch.logical_and(true, ~detection_mask).numpy(),
        }
        weight = torch.stack(
            [
                self.weighting_fn(
                    instances,
                    MasksType(
                        tp=masks["tp"][idx, ...],
                        fp=masks["fp"][idx, ...],
                        fn=masks["fn"][idx, ...],
                    ),
                    image,
                )
                for idx in range(self.n_samples)
            ],
            0,
        )

        assert (weight >= 0).all()

        fp = (weight * (detection["scores"] >= self.score_threshold) * (~true)).sum(axis=-1)
        fn = (weight * (detection["scores"] < self.score_threshold) * true).sum(axis=-1)

        if self.hardness_name == "fp":
            hardness_samples = fp
        elif self.hardness_name == "fn":
            hardness_samples = fn
        elif self.hardness_name == "false":
            hardness_samples = fp + fn
        else:
            raise NotImplementedError(f"Hardness name {self.hardness_name} is unknown, must be fp, fn or false.")

        assert len(hardness_samples) == self.n_samples

        return self.reduction(np.array(hardness_samples))


class IntegratedCommitteeHardnessCalculation(ABC):
    """
    Hardness calculation integrating over a committee (i.e. a set) of different detectors.

    The commitee could be identical detectors as in an ensemble, or otherwise. The committee will be treated as
    providing a discrete distribution over detections. This can be used to obtain a discrete distribution over hardness
    which can then be integrated to provide an expectated hardness.

    Args:
        hardness_definition: Definition of hardness to be computed (function of detections, targets and image)
        pseudo_ground_truth_model: Produces samples of pseudo ground truth based on detections.
        score_threshold: The detector score threshold (proposed detections with a score greater than this are
            are considered confirmed).

    Attributes:
        hardness_definition: Definition of hardness to be computed (function of detections, targets and image)
        pseudo_ground_truth_model: Produces samples of pseudo ground truth based on detections.
        score_threshold: The detector score threshold (proposed detections with a score greater than this are
            are considered confirmed).
    """

    def __init__(
        self,
        hardness_definition: HardnessDefinition,
        pseudo_ground_truth_model: DiscreteCommittee,
        score_threshold: float = 0.5,
    ):
        super().__init__()
        self.hardness_definition: HardnessDefinition = hardness_definition
        self.pseudo_ground_truth_model: DiscreteCommittee = pseudo_ground_truth_model
        self.score_threshold: float = score_threshold

    def calculate_hardness(
        self, detection_list_committee: List[DetectionType], detection: DetectionType, image: torch.Tensor
    ) -> float:
        """
        Estimate hardness for a single image

        Args:
            detection_list_committee: List of detections for each member of the committee for the image
            detection: Detections for a single image for detector for which hardness is being evaluated in torchvision
                format
            image: Image for which hardness is being computed in torchvision format (C, H, W)

        Returns:
            Hardness

        """
        pseudo_ground_truth, probability_mass = self.pseudo_ground_truth_model.probability_mass(
            detection_list_committee
        )

        detection = filter_pytorch_detections_by_score(detection, self.score_threshold)

        hardness_samples = [
            self.hardness_definition(detections=detection, annotations=gt, image=image) for gt in pseudo_ground_truth
        ]

        return sum(np.array(hardness_samples) * np.array(probability_mass))

    def eval_dataset(
        self,
        detection_list_committee: List[Iterable[DetectionType]],
        detections: Iterable[DetectionType],
        images: Iterable[torch.Tensor],
    ) -> List[float]:
        """
        Scores a dataset by hardness.

        Args:
            detection_list_committee: List of list of detections for committee, outer list over images in dataset, inner
                list over committee members.
            detections: List of detections in torchvision format for detector for which hardness is being evaluated
                (should be obtained by running the detector with a low score threshold as low score detections are
                required for the algorithm to work correctly).
            images: List of images in torchvision format (C, H, W) (can be a generator if dataset is large)

        Returns:
            Estimated hardness of dataset
        """
        detection_list_committee_per_image = list(zip(*detection_list_committee))

        with tqdm(detections, desc=f"Calculating hardness with {self}, fraction of dataset processed") as bar:
            return list(
                itertools.starmap(self.calculate_hardness, zip(detection_list_committee_per_image, bar, images))
            )


class SampledCommitteeHardnessCalculation(ABC):
    """
    Hardness calculation by score sampling algorithm on detections from a committee (i.e. a set) of different detectors

    A combination of IntegratedCommitteeHardness and ScoreSamplingHardness.

    Args:
        hardness_definition: Definition of hardness to be computed (function of detections, targets and image)
        n_samples: Number of samples to make from pseudo ground truth model, which are used to compute hardness
            samples
        pseudo_ground_truth_model: Produces samples of pseudo ground truth based on detections.
        score_threshold: The detector score threshold (proposed detections with a score greater than this are
            are considered confirmed).

    Attributes:
        hardness_definition: Definition of hardness to be computed (function of detections, targets and image)
        n_samples: Number of samples to make from pseudo ground truth model, which are used to compute hardness
            samples
        pseudo_ground_truth_model: Produces samples of pseudo ground truth based on detections.
        score_threshold: The detector score threshold (proposed detections with a score greater than this are
            are considered confirmed).
    """

    def __init__(
        self,
        hardness_definition: HardnessDefinition,
        n_samples: int,
        pseudo_ground_truth_model: ScoreSamplingUniformCommittee,  # TODO: eventually generalise this to non uniform
        score_threshold: float = 0.5,
    ):
        super().__init__()
        self.n_samples: int = n_samples
        self.hardness_definition: HardnessDefinition = hardness_definition
        self.pseudo_ground_truth_model: ScoreSamplingUniformCommittee = pseudo_ground_truth_model
        self.score_threshold: float = score_threshold

    def calculate_hardness(
        self, detection_list_committee: List[DetectionType], detection: DetectionType, image: torch.Tensor
    ) -> float:
        """
        Estimate hardness for a single image

        Args:
            detection_list_committee: List of detections for each member of the committee for the image
            detection: Detections for a single image for detector for which hardness is being evaluated in torchvision
                format
            image: Image for which hardness is being computed in torchvision format (C, H, W)

        Returns:
            Hardness

        """
        detection = filter_pytorch_detections_by_score(detection, self.score_threshold)

        pseudo_ground_truth = self.pseudo_ground_truth_model.sample(detection_list_committee, n_samples=self.n_samples)

        hardness_samples = [
            self.hardness_definition(detections=detection, annotations=gt, image=image) for gt in pseudo_ground_truth
        ]

        return np.mean(hardness_samples)

    def eval_dataset(
        self,
        detection_list_committee: List[Iterable[DetectionType]],
        detections: Iterable[DetectionType],
        images: Iterable[torch.Tensor],
    ) -> List[float]:
        """
        Scores a dataset by hardness.
        Args:
            detection_list_committee: List of list of detections for committee, outer list over images in dataset, inner
                list over committee members.
            detections: List of detections for detector for which hardness is being evaluated in torchvision format
                (should be obtained by running the detector with a low score threshold as low score detections are
                required for the algorithm to work correctly).
            images: List of images in torchvision format (C, H, W) (can be a generator if dataset is large)

        Returns:
            Estimated hardness of dataset
        """
        detection_list_committee_per_image = list(zip(*detection_list_committee))
        with tqdm(detections, desc=f"Calculating hardness with {self}, fraction of dataset processed") as bar:
            return list(
                itertools.starmap(self.calculate_hardness, zip(detection_list_committee_per_image, bar, images))
            )
