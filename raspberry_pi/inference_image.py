"""Run the style transfer HEF model on one image."""

from pathlib import Path

import numpy as np
from PIL import Image
from hailo_platform import (
    HEF,
    ConfigureParams,
    FormatType,
    HailoStreamInterface,
    InferVStreams,
    InputVStreamParams,
    OutputVStreamParams,
    VDevice,
)


PROJECT_DIR = Path(__file__).resolve().parent.parent
HEF_PATH = PROJECT_DIR / "starry_night.hef"
INPUT_IMAGE = PROJECT_DIR / "images" / "000000000802.jpg"
OUTPUT_IMAGE = PROJECT_DIR / "hef_output.jpg"

hef = HEF(str(HEF_PATH))
input_info = hef.get_input_vstream_infos()[0]
output_info = hef.get_output_vstream_infos()[0]
input_height, input_width, _ = input_info.shape

with Image.open(INPUT_IMAGE) as image:
    image = image.convert("RGB")
    image = image.resize((input_width, input_height))
    input_tensor = np.asarray(image, dtype=np.float32)

# Match the [-1, 1] input range used during PyTorch training.
input_tensor = input_tensor / 127.5 - 1.0
input_tensor = np.expand_dims(input_tensor, axis=0)

with VDevice() as target:
    configure_params = ConfigureParams.create_from_hef(
        hef,
        interface=HailoStreamInterface.PCIe,
    )
    network_group = target.configure(hef, configure_params)[0]
    network_group_params = network_group.create_params()

    input_params = InputVStreamParams.make(
        network_group,
        quantized=False,
        format_type=FormatType.FLOAT32,
    )
    output_params = OutputVStreamParams.make(
        network_group,
        quantized=False,
        format_type=FormatType.FLOAT32,
    )

    with InferVStreams(network_group, input_params, output_params) as pipeline:
        with network_group.activate(network_group_params):
            results = pipeline.infer({input_info.name: input_tensor})

output_tensor = results[output_info.name][0]
output_image = ((output_tensor + 1.0) / 2.0).clip(0.0, 1.0)
output_image = (output_image * 255.0).astype(np.uint8)

Image.fromarray(output_image).save(OUTPUT_IMAGE)

print(f"Input shape: {input_info.shape}")
print(f"Output shape: {output_info.shape}")
print(f"Saved: {OUTPUT_IMAGE}")
