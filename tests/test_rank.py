import pathlib
import zipfile
from io import BytesIO

import numpy as np
import pytest
import requests
import torch
import torchvision
from torchvision.datasets.coco import CocoDetection
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

from hardest.detectors import wrap_detector_with_logit_call
from hardest.hardness_strategies import BinaryDempsterHardnessCalculation
from hardest.hardness_strategies import BinaryEntropyCalculation
from hardest.hardness_strategies import DempsterHardnessCalculation
from hardest.hardness_strategies import EntropyCalculation
from hardest.hardness_strategies import FastScoreSamplingHardnessCalculation
from hardest.hardness_strategies import IntegratedCommitteeHardnessCalculation
from hardest.hardness_strategies import MaxUncertaintyCalculation
from hardest.hardness_strategies import SampledCommitteeHardnessCalculation
from hardest.hardness_strategies import ScoreSamplingHardnessCalculation
from hardest.hardness_strategies import SupervisedHardnessCalculation
from hardest.pseudo_ground_truth_models import ScoreSamplingUniformCommittee
from hardest.pseudo_ground_truth_models import UniformCommittee
from hardest.pycocotools import hardness_definition_factory
from hardest.pycocotools import weightings
from hardest.utils import coco_api_to_torchvision


@pytest.fixture(scope="session")
def minicoco(tmpdir_factory):
    tmpdir = pathlib.Path(tmpdir_factory.mktemp("data"))
    url = "https://github.com/zhiqwang/yolov5-rt-stack/releases/download/v0.3.0/coco128.zip"
    req = requests.get(url)
    file = zipfile.ZipFile(BytesIO(req.content))
    file.extractall(tmpdir)
    dataset = CocoDetection(
        root=tmpdir / "coco128/images/train2017/",
        annFile=tmpdir / "coco128/annotations/instances_train2017.json",
        transform=to_tensor,
        target_transform=coco_api_to_torchvision,
    )
    return [thing for thing in dataset][:2]  # Speed up evaluation by only evaluating on 2 images


@pytest.fixture(
    params=[torchvision.models.detection.retinanet_resnet50_fpn, torchvision.models.detection.fasterrcnn_resnet50_fpn]
)
def model(request):
    model = request.param(pretrained=True)
    model.eval()
    if isinstance(model, GeneralizedRCNN):
        wrap_detector_with_logit_call(model)

    return model


@pytest.mark.parametrize("hardness_definition", list(hardness_definition_factory.values()))
def test_smoke_rank(minicoco, hardness_definition, model):
    with torch.no_grad():
        detections = [model([image])[0] for image, target in tqdm(minicoco, desc="Running detector on dataset")]

    images, targets = zip(*minicoco)

    rank = ScoreSamplingHardnessCalculation(hardness_definition, n_samples=10).eval_dataset(detections, images)
    assert np.isfinite(rank).all()

    rank = SupervisedHardnessCalculation(hardness_definition).eval_dataset(detections, targets, images)
    assert np.isfinite(rank).all()

    if isinstance(model, GeneralizedRCNN):
        rank = DempsterHardnessCalculation().eval_dataset(detections, images)
        assert np.isfinite(rank).all()

        rank = EntropyCalculation().eval_dataset(detections, images)
        assert np.isfinite(rank).all()
    else:
        rank = BinaryDempsterHardnessCalculation().eval_dataset(detections, images)
        assert np.isfinite(rank).all()

        rank = BinaryEntropyCalculation().eval_dataset(detections, images)
        assert np.isfinite(rank).all()

    rank = MaxUncertaintyCalculation().eval_dataset(detections, images)
    assert np.isfinite(rank).all()


@pytest.mark.parametrize("hardness_definition", ["fp", "fn", "false"])
@pytest.mark.parametrize("weighting", weightings.values())
def test_fast_score_sampling(minicoco, model, hardness_definition, weighting):
    with torch.no_grad():
        detections = [model([image])[0] for image, target in tqdm(minicoco, desc="Running detector on dataset")]

    images, targets = zip(*minicoco)

    rank = FastScoreSamplingHardnessCalculation(
        hardness_name=hardness_definition, weighting_fn=weighting, n_samples=10
    ).eval_dataset(detections, images)
    assert np.isfinite(rank).all()


@pytest.mark.parametrize("hardness_definition", list(hardness_definition_factory.values()))
def test_committee(minicoco, hardness_definition):
    committee = [
        torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True),
        torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True),
    ]
    detections = []
    with torch.no_grad():
        for model_idx, model in enumerate(committee):
            model.eval()
            detections.append(
                [model([image])[0] for image, target in tqdm(minicoco, desc="Running detector on dataset")]
            )
    images, targets = zip(*minicoco)
    rank = IntegratedCommitteeHardnessCalculation(
        hardness_definition, pseudo_ground_truth_model=UniformCommittee()
    ).eval_dataset(detection_list_committee=detections, detections=detections[0], images=images)
    assert np.isfinite(rank).all()

    rank = SampledCommitteeHardnessCalculation(
        hardness_definition, pseudo_ground_truth_model=ScoreSamplingUniformCommittee(), n_samples=10
    ).eval_dataset(detection_list_committee=detections, detections=detections[0], images=images)
    assert np.isfinite(rank).all()


@pytest.mark.parametrize("hardness_definition", list(hardness_definition_factory.values()))
def test_score_sampling_constant_hardness(minicoco, hardness_definition, model):
    images, targets = zip(*minicoco)
    with torch.no_grad():
        detections = [model([image])[0] for image, target in tqdm(minicoco, desc="Running detector on dataset")]
    n_repeat = 10
    repeated_detections = [detections[0] for _ in range(n_repeat)]
    repeated_images = [images[0] for _ in range(n_repeat)]
    repeated_targets = [targets[0] for _ in range(n_repeat)]
    rank = SupervisedHardnessCalculation(hardness_definition).eval_dataset(
        repeated_detections, repeated_targets, repeated_images
    )
    np.testing.assert_array_equal(rank, rank[0])

    if isinstance(model, GeneralizedRCNN):
        rank = DempsterHardnessCalculation().eval_dataset(repeated_detections, repeated_images)
        np.testing.assert_array_equal(rank, rank[0])

        rank = EntropyCalculation().eval_dataset(repeated_detections, repeated_images)
        np.testing.assert_array_equal(rank, rank[0])
    else:
        rank = BinaryDempsterHardnessCalculation().eval_dataset(repeated_detections, repeated_images)
        np.testing.assert_array_equal(rank, rank[0])

        rank = BinaryEntropyCalculation().eval_dataset(repeated_detections, repeated_images)
        np.testing.assert_array_equal(rank, rank[0])

    rank = MaxUncertaintyCalculation().eval_dataset(repeated_detections, repeated_images)
    np.testing.assert_array_equal(rank, rank[0])
