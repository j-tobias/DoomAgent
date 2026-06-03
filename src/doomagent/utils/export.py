import json
import warnings
from pathlib import Path

import onnx
import torch
import torch.nn as nn

from ..models.base import BaseModel


class _LogitsOnly(nn.Module):
    """
    Thin wrapper that strips the value head output before export.

    PPOActorCritic.forward() returns (logits, value). Exporting a tuple
    causes onnx2pytorch to receive a Python list instead of a tensor,
    crashing at inference time. Wrapping to return only logits fixes this
    while remaining compatible with the eval server, which already handles
    both single-tensor and tuple outputs:
        if isinstance(logits, tuple): logits, _ = logits
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        out = self.model(obs)
        return out[0] if isinstance(out, tuple) else out


def export_onnx(model: BaseModel, path: Path, obs_shape: tuple) -> None:
    """
    Export model to ONNX (legacy TorchScript exporter, opset 12) and attach
    the config metadata the grading server reads from metadata_props.

    obs_shape: (C, H, W) — no batch dim.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    device = next(model.parameters()).device
    dummy = torch.zeros(1, *obs_shape, device=device)

    # Legacy exporter (dynamo=False) + opset 12 is required for onnx2pytorch
    # compatibility. The new dynamo-based exporter (PyTorch ≥ 2.9 default)
    # produces Conv attribute layouts that onnx2pytorch cannot parse.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        torch.onnx.export(
            _LogitsOnly(model),
            dummy,
            str(path),
            input_names=["obs"],
            output_names=["logits"],
            opset_version=12,
            dynamo=False,
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
