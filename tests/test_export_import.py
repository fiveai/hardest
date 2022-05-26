from pathlib import Path
from typing import List

import torch

from hardest.detectors import DetectionType
from hardest.utils import load_detections
from hardest.utils import serialise_detections


def test_export_import(tmpdir: Path):
    detections: List[DetectionType] = [
        {
            "boxes": torch.Tensor(
                [
                    [0, 0, 2, 1],
                    [1, 0, 2, 2],
                ]
            ),
            "labels": torch.Tensor([1, 1]),
            "scores": torch.Tensor([0.5, 0.7]),
        }
        for _ in range(4)
    ]
    serialise_detections(tmpdir / "test.json", detections)
    loaded_detections = load_detections(tmpdir / "test.json")
    for det_a, det_b in zip(detections, loaded_detections):
        for key in det_a:
            assert (det_a[key] == det_b[key]).all()  # type: ignore
    # ignore required until https://github.com/python/mypy/issues/7981 is fixed
