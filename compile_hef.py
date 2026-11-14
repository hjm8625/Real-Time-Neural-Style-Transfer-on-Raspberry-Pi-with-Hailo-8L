"""Compile the style transfer ONNX model for Hailo-8L."""

from pathlib import Path
import sys
import traceback

import numpy as np
from hailo_sdk_client import ClientRunner


MODEL_NAME = "starry_night_style_transfer"
HW_ARCH = "hailo8l"

WORK_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = WORK_DIR / "hailo_output"

ONNX_PATH = WORK_DIR / "starry_night.onnx"
CALIBRATION_PATH = (
    WORK_DIR
    / "calibration_output"
    / "starry_night_calibration_100.npy"
)

PARSED_HAR_PATH = OUTPUT_DIR / "starry_night_parsed.har"
QUANTIZED_HAR_PATH = OUTPUT_DIR / "starry_night_quantized.har"
COMPILED_HAR_PATH = OUTPUT_DIR / "starry_night_compiled.har"
HEF_PATH = WORK_DIR / "starry_night.hef"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_calibration_data():
    calibration_data = np.load(CALIBRATION_PATH).astype(np.float32)
    expected_shape = (100, 256, 256, 3)

    if calibration_data.shape != expected_shape:
        raise ValueError(
            f"Expected calibration shape {expected_shape}, "
            f"but got {calibration_data.shape}"
        )

    if calibration_data.min() < -1.0 or calibration_data.max() > 1.0:
        raise ValueError("Calibration values must be in the [-1, 1] range")

    print("Calibration dataset")
    print(f"  Shape: {calibration_data.shape}")
    print(f"  dtype: {calibration_data.dtype}")
    print(
        f"  Range: {calibration_data.min():.4f} "
        f"to {calibration_data.max():.4f}"
    )

    return calibration_data


def parse_onnx():
    print("\n[1/3] Parsing ONNX")

    runner = ClientRunner(hw_arch=HW_ARCH)
    runner.translate_onnx_model(
        str(ONNX_PATH),
        MODEL_NAME,
    )
    runner.save_har(str(PARSED_HAR_PATH))

    print(f"Saved: {PARSED_HAR_PATH}")


def optimize_model(calibration_data):
    print("\n[2/3] Optimizing and quantizing")

    runner = ClientRunner(har=str(PARSED_HAR_PATH))

    # No Hailo normalization layer is added because the ONNX model and
    # calibration data already use the [-1, 1] input range.
    runner.optimize(calibration_data)
    runner.save_har(str(QUANTIZED_HAR_PATH))

    print(f"Saved: {QUANTIZED_HAR_PATH}")

    return runner


def compile_model(runner):
    print("\n[3/3] Compiling for Hailo-8L")

    hef_binary = runner.compile()
    HEF_PATH.write_bytes(hef_binary)
    runner.save_har(str(COMPILED_HAR_PATH))

    print(f"Saved: {COMPILED_HAR_PATH}")
    print(f"Saved: {HEF_PATH}")


def main():
    print("Starry Night Style Transfer: ONNX to Hailo-8L HEF")
    print(f"ONNX: {ONNX_PATH}")
    print(f"Calibration: {CALIBRATION_PATH}")

    calibration_data = load_calibration_data()
    parse_onnx()
    runner = optimize_model(calibration_data)
    compile_model(runner)

    print("\nHEF compilation complete")
    print(f"Final HEF: {HEF_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nHEF compilation failed: {error}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
