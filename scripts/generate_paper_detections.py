import argparse
import pickle
from pathlib import Path

import torch
import torchvision
from torchvision.datasets.coco import CocoDetection
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

from hardest.detectors import KClassDetectionType
from hardest.detectors import wrap_detector_with_logit_call
from hardest.utils import coco_api_to_torchvision
from hardest.utils import serialise_detections


def main():
    parser = argparse.ArgumentParser(description="Dump detections")
    parser.add_argument("--coco-root", type=Path, help="root directory for coco", required=True)
    parser.add_argument("--save-dir", type=Path, help="directory to save results", default=Path("."))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, choices=["rcnn", "retina"], required=True)
    args = parser.parse_args()

    root = args.coco_root
    save_dir = args.save_dir
    device = torch.device(args.device)

    dataset = CocoDetection(
        root=root / "val2017/",
        annFile=root / "annotations/instances_val2017.json",
        transform=to_tensor,
        target_transform=coco_api_to_torchvision,
    )
    if args.model == "rcnn":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        out_path = save_dir / "fasterrcnn_torchvision.json" if save_dir is not None else None
    elif args.model == "retina":
        model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
        out_path = save_dir / "retinanet_torchvision.json" if save_dir is not None else None
    else:
        NotImplementedError(f"{args.model} is an unknown model")

    model.eval()
    model.to(device=device)

    if args.model == "rcnn":
        model = wrap_detector_with_logit_call(model)

    images, targets = zip(*dataset)

    with torch.no_grad():
        detections: KClassDetectionType = list(
            map(lambda image: model([image.to(device)])[0], tqdm(images, desc="Running detector on dataset"))
        )

    if save_dir is not None:
        serialise_detections(out_path, detections)


if __name__ == "__main__":
    main()
