import unittest

import torch

from hardest.pycocotools import FN
from hardest.pycocotools import FP


class TestPycocotools(unittest.TestCase):
    def setUp(self):
        self.fp = FP()
        self.fn = FN()
        height = 500
        width = 500
        self.image = torch.zeros((3, height, width))

    def test_overlapping(self):
        detections = {
            "boxes": torch.Tensor(
                [
                    [0, 0, 1, 1],
                ]
            ),
            "labels": torch.Tensor([1]),
            "scores": torch.Tensor([0.5]),
        }
        annotations = {
            "boxes": torch.Tensor(
                [
                    [0, 0, 1, 1],
                ]
            ),
            "labels": torch.Tensor([1]),
            "iscrowd": torch.Tensor([0]),
        }
        fps = self.fp(detections, annotations, self.image)
        fns = self.fn(detections, annotations, self.image)
        assert fps == 0
        assert fns == 0

    def test_no_annotations(self):
        detections = {
            "boxes": torch.Tensor(
                [
                    [0, 0, 1, 1],
                ]
            ),
            "labels": torch.Tensor([1]),
            "scores": torch.Tensor([0.5]),
        }
        annotations = {"boxes": torch.Tensor([]), "labels": torch.Tensor([]), "iscrowd": torch.Tensor([])}
        fps = self.fp(detections, annotations, self.image)
        fns = self.fn(detections, annotations, self.image)
        assert fps == 1
        assert fns == 0

    def test_no_detections(self):
        annotations = {
            "boxes": torch.Tensor(
                [
                    [0, 0, 1, 1],
                ]
            ),
            "labels": torch.Tensor([1]),
            "iscrowd": torch.Tensor([0]),
        }
        detections = {
            "boxes": torch.Tensor([]),
            "labels": torch.Tensor([]),
            "scores": torch.Tensor([]),
        }
        fps = self.fp(detections, annotations, self.image)
        fns = self.fn(detections, annotations, self.image)
        assert fps == 0
        assert fns == 1

    def test_no_overlap(self):
        detections = {
            "boxes": torch.Tensor(
                [
                    [0, 0, 1, 1],
                ]
            ),
            "labels": torch.Tensor([1]),
            "scores": torch.Tensor([0.5]),
        }
        annotations = {
            "boxes": torch.Tensor(
                [
                    [1, 1, 2, 2],
                ]
            ),
            "labels": torch.Tensor([1]),
            "iscrowd": torch.Tensor([0]),
        }
        fps = self.fp(detections, annotations, self.image)
        fns = self.fn(detections, annotations, self.image)
        assert fps == 1
        assert fns == 1

    def test_different_class_overlap(self):
        detections = {
            "boxes": torch.Tensor(
                [
                    [0, 0, 1, 1],
                ]
            ),
            "labels": torch.Tensor([1]),
            "scores": torch.Tensor([0.5]),
        }
        annotations = {
            "boxes": torch.Tensor(
                [
                    [1, 1, 2, 2],
                ]
            ),
            "labels": torch.Tensor([2]),
            "iscrowd": torch.Tensor([0]),
        }
        fps = self.fp(detections, annotations, self.image)
        fns = self.fn(detections, annotations, self.image)
        assert fps == 1
        assert fns == 1

    def test_partial_overlap(self):
        detections = {
            "boxes": torch.Tensor(
                [
                    [0, 0, 2, 1],
                    [1, 0, 2, 2],
                ]
            ),
            "labels": torch.Tensor([1, 1]),
            "scores": torch.Tensor([0.5, 0.7]),
        }
        annotations = {
            "boxes": torch.Tensor(
                [
                    [0, 0, 2, 2],
                ]
            ),
            "labels": torch.Tensor([1]),
            "iscrowd": torch.Tensor([0]),
        }
        fps = self.fp(detections, annotations, self.image)
        fns = self.fn(detections, annotations, self.image)
        assert fps == 1
        assert fns == 0

    def test_is_crowd_is_ignored(self):
        detections = {
            "boxes": torch.Tensor([]),
            "labels": torch.Tensor([]),
            "scores": torch.Tensor([]),
        }
        annotations = {
            "boxes": torch.Tensor(
                [
                    [0, 0, 2, 2],
                ]
            ),
            "labels": torch.Tensor([1]),
            "iscrowd": torch.Tensor([1]),
        }
        fps = self.fp(detections, annotations, self.image)
        fns = self.fn(detections, annotations, self.image)
        # is crowd boxes are ignored for evaluation so the fn is discarded
        assert fps == 0
        assert fns == 0
