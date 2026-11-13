"""Validate ONNX and compare its output with the PyTorch model."""

from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from PIL import Image
from torchvision import transforms

from model import StyleTransferNet


CHECKPOINT_PATH = Path("outputs/model_epoch_010.pt")
ONNX_PATH = Path("starry_night.onnx")
TEST_IMAGE_PATH = Path("images/000000000802.jpg")
RTOL = 1e-4 #relative tolerance
ATOL = 1e-4 #absolute tolerance

transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ]
)


onnx_model = onnx.load(ONNX_PATH)
onnx.checker.check_model(onnx_model)
print("ONNX validation: PASS")

operators = Counter(node.op_type for node in onnx_model.graph.node)
print("ONNX operators:")
for operator, count in sorted(operators.items()):
    print(f"  {operator}: {count}")

model = StyleTransferNet()
state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
model.load_state_dict(state_dict)
model.eval()

with Image.open(TEST_IMAGE_PATH) as image:
    input_tensor = transform(image.convert("RGB")).unsqueeze(0)

with torch.inference_mode():
    pytorch_output = model(input_tensor).numpy()

session = ort.InferenceSession(
    ONNX_PATH,
    providers=["CPUExecutionProvider"],
)
input_info = session.get_inputs()[0]
output_info = session.get_outputs()[0]

print(f"Input name: {input_info.name}")
print(f"Input shape: {input_info.shape}")
print(f"Output name: {output_info.name}")
print(f"Output shape: {output_info.shape}")

input_array = input_tensor.numpy().astype(np.float32)
onnx_output = session.run(
    [output_info.name],
    {input_info.name: input_array},
)[0]

shape_match = pytorch_output.shape == onnx_output.shape
absolute_difference = np.abs(pytorch_output - onnx_output)
maximum_difference = float(absolute_difference.max())
mean_difference = float(absolute_difference.mean())
allclose = np.allclose(
    pytorch_output,
    onnx_output,
    rtol=RTOL,
    atol=ATOL,
)
passed = shape_match and allclose

print()
print(f"Shape match: {shape_match}")
print(f"Maximum absolute difference: {maximum_difference:.10f}")
print(f"Mean absolute difference: {mean_difference:.10f}")
print(f"Allclose: {allclose}")
print(f"PyTorch vs ONNX Runtime: {'PASS' if passed else 'FAIL'}")
