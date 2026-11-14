"""Run real-time style transfer with a Raspberry Pi Camera and Hailo-8L."""

import time
from pathlib import Path

import cv2
import numpy as np
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
from picamera2 import Picamera2


PROJECT_DIR = Path(__file__).resolve().parent.parent
HEF_PATH = PROJECT_DIR / "starry_night.hef"
CAMERA_SIZE = (640, 480)

hef = HEF(str(HEF_PATH))
input_info = hef.get_input_vstream_infos()[0]
output_info = hef.get_output_vstream_infos()[0]
input_height, input_width, _ = input_info.shape

camera = Picamera2()
camera_config = camera.create_preview_configuration(
    main={"size": CAMERA_SIZE, "format": "RGB888"}
)
camera.configure(camera_config)
camera.start()

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
            previous_time = time.perf_counter()

            try:
                while True:
                    frame = camera.capture_array()
                    input_frame = cv2.resize(frame, (input_width, input_height))

                    # Match the [-1, 1] input range used during training.
                    input_tensor = input_frame.astype(np.float32) / 127.5 - 1.0
                    input_tensor = np.expand_dims(input_tensor, axis=0)

                    results = pipeline.infer({input_info.name: input_tensor})
                    output_tensor = results[output_info.name][0]

                    output_frame = ((output_tensor + 1.0) / 2.0).clip(0.0, 1.0)
                    output_frame = (output_frame * 255.0).astype(np.uint8)
                    output_frame = cv2.resize(output_frame, CAMERA_SIZE)

                    current_time = time.perf_counter()
                    fps = 1.0 / (current_time - previous_time)
                    previous_time = current_time

                    display_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
                    cv2.putText(
                        display_frame,
                        f"FPS: {fps:.1f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )
                    cv2.imshow("Starry Night", display_frame)

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            finally:
                camera.stop()
                cv2.destroyAllWindows()
