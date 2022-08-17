# Hardest

[![Run tests](https://github.com/fiveai/hardest/actions/workflows/python-package.yml/badge.svg)](https://github.com/fiveai/hardest/actions/workflows/python-package.yml)
[![Upload Python Package](https://github.com/fiveai/hardest/actions/workflows/deploy.yml/badge.svg)](https://github.com/fiveai/hardest/actions/workflows/deploy.yml)
[![pre-commit](https://github.com/fiveai/hardest/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/fiveai/hardest/actions/workflows/pre-commit.yml)

The HARDness ESTimation package: A library for ranking images from a dataset by hardness with respect to a specific detector.
Currently, we provide examples in the library for torchvision datasets and detectors, but other datasets and detectors
can be used by converting the data to the torchvision format.

## Getting started
To install:
```bash
pip install hardest
```

Here we provide instructions for computing the hardness of an entire dataset.
Firstly obtain a pytorch dataset:
```python
from torchvision.datasets.coco import CocoDetection
import itertools
dataset = CocoDetection(...)

# Run on a subset of the dataset
n_examples = 50
images, targets = zip(*itertools.islice(dataset, n_examples))

```
Obtain detections:
```python
import torch, torchvision

model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
model.eval()
with torch.no_grad():
    detections = [model([image])[0] for image in images]
```

Choose a definition of hardness:
```python
from hardest.pycocotools import TotalFalse
hardness_definition = TotalFalse()
```

Estimate hardness using annotations:
```python
from hardest.hardness_strategies import SupervisedHardnessCalculation
rank = SupervisedHardnessCalculation(hardness_definition).eval_dataset(
    detections,
    targets,
    images,
)
```

If annotations are not available, you can estimate hardness without annotations:
```python
from hardest.hardness_strategies import ScoreSamplingHardnessCalculation
rank = ScoreSamplingHardnessCalculation(hardness_definition, n_samples=10).eval_dataset(detections, images)
```

## Reproducing published results
To repeat the experiments from our paper (details here), first install the [COCO](https://cocodataset.org) as well as the [nuImages](https://www.nuscenes.org/nuimages) datasets and convert it to a COCO compatible format [using these instructions](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/nuimages/README.md).
Then export the detections for your desired detector to torchvision json format.

Finally, run the script:
```bash
python scripts/paper_experiments.py --coco-root datasets/coco --nuimages-root datasets/nuimages-coco --save-dir ./results --detection-path-coco detections/coco --detection-path-nuimages detections/nuimages
```
This will require that you have stored detections in json files in the appropriate paths and downloaded the datasets to
the specified paths.

To reproduce the nuimages results you will need to convert the nuimages dataset to the coco schema:
https://github.com/open-mmlab/mmdetection3d/blob/master/configs/nuimages/README.md

If you use the package in your research please consider citing our paper:
```
details here
```

## Contributors

### Authors

- Jonathan Sadeghi
- Edward Ayres

### Internal Review

- Anuj Sharma
- Blaine Rogers
- Romain Mueller
- Zygmunt Lenyk
