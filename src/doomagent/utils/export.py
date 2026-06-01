import json
from pathlib import Path

import onnx
import torch

from ..models.base import BaseModel


def export_onnx(model: BaseModel, path: Path, obs_shape: tuple) -> None:
    """
    Export model to ONNX and attach the config metadata the eval server expects.

    obs_shape: (C, H, W) — no batch dim. Use env.observation_space.shape.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    device = next(model.parameters()).device
    dummy = torch.zeros(1, *obs_shape, device=device)

    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=["obs"],
        output_names=["logits"],
        opset_version=17,
    )

    # Attach the config JSON the grading server reads from metadata_props
    onnx_model = onnx.load(str(path))
    entry = onnx_model.metadata_props.add()
    entry.key = "config"
    entry.value = json.dumps(model.onnx_config)
    onnx.save(onnx_model, str(path))

    size_mb = path.stat().st_size / (1024 ** 2)
    print(f"ONNX export → {path}  ({size_mb:.1f} MB)")
    if size_mb > 50:
        print("WARNING: submission exceeds the 50 MB size limit!")
