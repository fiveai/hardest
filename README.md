# Hardest

[![Run tests](https://github.com/fiveai/hardest/actions/workflows/python-package.yml/badge.svg)](https://github.com/fiveai/hardest/actions/workflows/python-package.yml)
[![Upload Python Package](https://github.com/fiveai/hardest/actions/workflows/deploy.yml/badge.svg)](https://github.com/fiveai/hardest/actions/workflows/deploy.yml)
[![pre-commit](https://github.com/fiveai/hardest/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/fiveai/hardest/actions/workflows/pre-commit.yml)

[![arXiv](https://img.shields.io/badge/arXiv-2209.11559-b31b1b.svg)](https://arxiv.org/abs/2209.11559)

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
from os.path import join
import torchvision.transforms as T
from hardest.utils import coco_api_to_torchvision

coco_dir = "<path/to/coco>"

transform = T.Compose([
    T.PILToTensor(),
    T.ConvertImageDtype(torch.float),
])

val_dataset = CocoDetection(
    root = join(coco_dir, "val2017"),
    annFile = join(coco_dir, "annotations", "instances_val2017.json"),
    transform = transform,
    target_transform = coco_api_to_torchvision
)

# Run on a subset of the dataset
n_examples = 50
images, targets = zip(*itertools.islice(val_dataset, n_examples))

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
To repeat the experiments from our paper (details here), first download the [COCO Dataset](https://cocodataset.org/#download) (you only need _2017 Val images_ and _2017 Train/Val annotations_) as well as the [nuImages](https://www.nuscenes.org/nuimages) dataset and convert it to a COCO compatible format [using these instructions](https://github.com/open-mmlab/mmdetection3d/blob/master/configs/nuimages/README.md).
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
Ayers, E., Sadeghi, J., Redford, J., Mueller, R., & Dokania, P. K. (2022). Query-based Hard-Image Retrieval for Object Detection at Test Time. Thirty-Seventh AAAI Conference on Artificial Intelligence. doi:10.48550/ARXIV.2209.11559
```

## Contributors

### Authors

- Jonathan Sadeghi
- Edward Ayers

### Internal Review

- Anuj Sharma
- Blaine Rogers
- Romain Mueller
- Zygmunt Lenyk
