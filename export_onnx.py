"""Export the trained style transfer model to a fixed-shape ONNX model."""

from pathlib import Path

import torch

from model import StyleTransferNet


PROJECT_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = PROJECT_DIR / "outputs" / "model_epoch_010.pt"
ONNX_PATH = PROJECT_DIR / "starry_night.onnx"
INPUT_SHAPE = (1, 3, 256, 256)

model = StyleTransferNet()
state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
model.load_state_dict(state_dict)
model.eval()

dummy_input = torch.randn(INPUT_SHAPE)

torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    input_names=["input"],
    output_names=["output"],
    dynamo=True,
)

print(f"Exported: {ONNX_PATH}")
print(f"Input shape: {INPUT_SHAPE}")
