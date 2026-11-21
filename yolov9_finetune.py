"""Fine-tune YOLOv9 from a SparK-pretrained backbone.

Example:
    python yolov9_finetune.py --data /path/to/dataset.yaml \
        --weights /path/to/pretrain_net_v9.pth

Requires Ultralytics 8.3.29 and a trusted local SparK checkpoint whose
stage1...stage5 keys match the YOLOv9 backbone defined below.
The pretrained backbone and detection layers remain trainable.
"""

from __future__ import annotations

import argparse
import importlib
import os
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Mapping

import torch
from torch import nn
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer


MODEL_YAML = "yolov9m.yaml"
STAGE_GROUPS = ((0,), (1, 2), (3, 4), (5,), (6, 7, 8, 9))
WEIGHTS_ENV = "ELEFLAI_SPARK_V9_WEIGHTS"


class PretrainNet(nn.Module):
    """Recreate the five stage names and layer groups used by the checkpoint."""

    def __init__(self, layers: nn.Sequential) -> None:
        super().__init__()
        for number, indices in enumerate(STAGE_GROUPS, start=1):
            stage = nn.Sequential(*(layers[index] for index in indices))
            setattr(self, f"stage{number}", deepcopy(stage))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for number in range(1, len(STAGE_GROUPS) + 1):
            x = getattr(self, f"stage{number}")(x)
        return x


def extract_state_dict(checkpoint: object) -> OrderedDict[str, torch.Tensor]:
    """Unwrap a trusted SparK checkpoint and remove a DDP module prefix."""
    state = checkpoint
    while isinstance(state, Mapping) and (
        "model" in state or "state_dict" in state
    ):
        state = state["model"] if "model" in state else state["state_dict"]
    if isinstance(state, nn.Module):
        state = state.state_dict()
    if not isinstance(state, Mapping):
        raise TypeError("The SparK checkpoint does not contain a state dict.")
    if not all(isinstance(key, str) for key in state):
        raise TypeError("The SparK checkpoint has non-string parameter keys.")
    return OrderedDict(
        (key.removeprefix("module."), value) for key, value in state.items()
    )


class SparKDetectionTrainer(DetectionTrainer):
    """Initialize the actual trainer model with SparK backbone weights."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        weights_name = os.environ.get(WEIGHTS_ENV)
        if not weights_name:
            raise RuntimeError(f"Set {WEIGHTS_ENV} to the SparK checkpoint path.")
        weights_path = Path(weights_name)
        if not weights_path.is_file():
            raise FileNotFoundError(f"SparK checkpoint not found: {weights_path}")

        # Load only a checkpoint from a trusted source; torch.load may unpickle.
        checkpoint = torch.load(weights_path, map_location="cpu")
        pretrained = PretrainNet(model.model)
        pretrained.load_state_dict(extract_state_dict(checkpoint), strict=True)

        # These Sequential wrappers reference the trainer model's real modules.
        with torch.no_grad():
            for number, indices in enumerate(STAGE_GROUPS, start=1):
                target = nn.Sequential(*(model.model[index] for index in indices))
                source = getattr(pretrained, f"stage{number}")
                target.load_state_dict(source.state_dict(), strict=True)
                if not all(
                    torch.equal(actual, expected)
                    for actual, expected in zip(
                        target.state_dict().values(), source.state_dict().values()
                    )
                ):
                    raise RuntimeError(f"SparK stage{number} transfer failed.")

        print(
            f"Loaded SparK backbone into the YOLOv9 trainer model "
            f"from {weights_path}."
        )
        return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="YOLO dataset YAML")
    parser.add_argument(
        "--weights", required=True, type=Path, help="SparK pretraining checkpoint"
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--device", default=None, help='Ultralytics device, e.g. "0" or "0,1,2,3"'
    )
    args = parser.parse_args()

    data_path = args.data.expanduser().resolve()
    weights_path = args.weights.expanduser().resolve()
    if not data_path.is_file():
        parser.error(f"Dataset YAML not found: {data_path}")
    if not weights_path.is_file():
        parser.error(f"SparK checkpoint not found: {weights_path}")
    if args.epochs <= 0 or args.batch <= 0 or args.imgsz <= 0:
        parser.error("--epochs, --batch, and --imgsz must be positive")

    os.environ[WEIGHTS_ENV] = str(weights_path)

    # Import the trainer under its module name so Ultralytics DDP can import it.
    script_dir = str(Path(__file__).resolve().parent)
    python_path = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = (
        script_dir if not python_path else script_dir + os.pathsep + python_path
    )
    trainer_class = importlib.import_module(Path(__file__).stem).SparKDetectionTrainer

    train_args = dict(
        trainer=trainer_class,
        data=str(data_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        # No freeze override: Supplementary Note 5 leaves it at the framework default.
        optimizer="auto",
        # Manuscript-reported lr0; Ultralytics 8.3.29 ignores it with Auto.
        lr0=0.01,
        amp=True,
    )
    if args.device is not None:
        train_args["device"] = args.device

    YOLO(MODEL_YAML).train(**train_args)


if __name__ == "__main__":
    main()

