"""Benchmark ONNX Runtime CPU and Hailo-8L with calibration images."""

from pathlib import Path
import csv
import os
import time

import cv2
import numpy as np
import onnxruntime as ort
from hailo_platform import (
    HEF,
    VDevice,
    ConfigureParams,
    FormatType,
    HailoStreamInterface,
    InferVStreams,
    InputVStreamParams,
    OutputVStreamParams,
)


WARMUP_RUNS = 10
MAX_IMAGES = 100

PROJECT_DIR = Path(__file__).resolve().parent
CALIBRATION_DIR = PROJECT_DIR / "calibration"
ONNX_PATH = PROJECT_DIR / "starry_night.onnx"
HEF_PATH = PROJECT_DIR / "starry_night.hef"
OUTPUT_DIR = PROJECT_DIR / "benchmark_output"
RAW_CSV_PATH = OUTPUT_DIR / "cpu_npu_latency.csv"
SUMMARY_CSV_PATH = OUTPUT_DIR / "cpu_npu_summary.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def calculate_stats(latencies_ms):
    values = np.asarray(latencies_ms, dtype=np.float64)

    return {
        "mean_ms": float(np.mean(values)),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
        "std_ms": float(np.std(values)),
        "fps": float(1000.0 / np.mean(values)),
    }


def print_stats(name, stats):
    print(f"\n[{name}]")
    print(f"Mean:   {stats['mean_ms']:.4f} ms")
    print(f"Median: {stats['median_ms']:.4f} ms")
    print(f"P95:    {stats['p95_ms']:.4f} ms")
    print(f"Min:    {stats['min_ms']:.4f} ms")
    print(f"Max:    {stats['max_ms']:.4f} ms")
    print(f"Std:    {stats['std_ms']:.4f} ms")
    print(f"FPS:    {stats['fps']:.2f}")


def convert_npu_output_to_nchw(output):
    output = np.asarray(output)

    if output.ndim == 3:
        output = np.expand_dims(output, axis=0)

    if output.shape[-1] == 3:
        output = np.transpose(output, (0, 3, 1, 2))

    return output


image_files = []

for extension in ("*.jpg", "*.jpeg", "*.png"):
    image_files.extend(CALIBRATION_DIR.glob(extension))

image_files = sorted(set(image_files))[:MAX_IMAGES]

if not image_files:
    raise FileNotFoundError(f"No calibration images found: {CALIBRATION_DIR}")

for path in (ONNX_PATH, HEF_PATH):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

hef = HEF(str(HEF_PATH))
input_info = hef.get_input_vstream_infos()[0]
output_info = hef.get_output_vstream_infos()[0]
input_height, input_width, input_channels = input_info.shape

if input_channels != 3:
    raise ValueError(f"Expected 3 input channels, got {input_channels}")

cpu_inputs = []
npu_inputs = []

for image_path in image_files:
    image_bgr = cv2.imread(str(image_path))

    if image_bgr is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (input_width, input_height))

    # Match the [-1, 1] range used by training, ONNX, and HEF calibration.
    input_nhwc = image_rgb.astype(np.float32) / 127.5 - 1.0
    input_nchw = np.transpose(input_nhwc, (2, 0, 1))

    npu_inputs.append(np.expand_dims(input_nhwc, axis=0))
    cpu_inputs.append(np.expand_dims(input_nchw, axis=0))

print("=" * 72)
print("Style Transfer CPU vs Hailo-8L Benchmark")
print("=" * 72)
print(f"Images: {len(image_files)}")
print(f"CPU cores: {os.cpu_count()}")
print(f"Warm-up runs: {WARMUP_RUNS}")
print(f"HEF input: {input_info.name} {input_info.shape}")
print(f"HEF output: {output_info.name} {output_info.shape}")

session = ort.InferenceSession(
    str(ONNX_PATH),
    providers=["CPUExecutionProvider"],
)
cpu_input_info = session.get_inputs()[0]
cpu_output_info = session.get_outputs()[0]

expected_cpu_shape = (1, 3, input_height, input_width)

if tuple(cpu_inputs[0].shape) != expected_cpu_shape:
    raise ValueError(
        f"Expected CPU input shape {expected_cpu_shape}, "
        f"got {cpu_inputs[0].shape}"
    )

for _ in range(WARMUP_RUNS):
    session.run(
        [cpu_output_info.name],
        {cpu_input_info.name: cpu_inputs[0]},
    )

cpu_latencies = []
npu_latencies = []
mean_errors = []
max_errors = []
raw_rows = []

with VDevice() as target:
    configure_params = ConfigureParams.create_from_hef(
        hef=hef,
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
            for _ in range(WARMUP_RUNS):
                pipeline.infer({input_info.name: npu_inputs[0]})

            for image_path, cpu_input, npu_input in zip(
                image_files,
                cpu_inputs,
                npu_inputs,
            ):
                start = time.perf_counter()
                cpu_output = session.run(
                    [cpu_output_info.name],
                    {cpu_input_info.name: cpu_input},
                )[0]
                cpu_latency_ms = (time.perf_counter() - start) * 1000.0

                start = time.perf_counter()
                npu_results = pipeline.infer({input_info.name: npu_input})
                npu_latency_ms = (time.perf_counter() - start) * 1000.0

                npu_output = convert_npu_output_to_nchw(
                    npu_results[output_info.name]
                )

                if cpu_output.shape != npu_output.shape:
                    raise ValueError(
                        f"Output shape mismatch: CPU {cpu_output.shape}, "
                        f"NPU {npu_output.shape}"
                    )

                absolute_error = np.abs(cpu_output - npu_output)
                mean_error = float(absolute_error.mean())
                max_error = float(absolute_error.max())

                cpu_latencies.append(cpu_latency_ms)
                npu_latencies.append(npu_latency_ms)
                mean_errors.append(mean_error)
                max_errors.append(max_error)

                raw_rows.append(
                    {
                        "image": image_path.name,
                        "cpu_latency_ms": cpu_latency_ms,
                        "npu_latency_ms": npu_latency_ms,
                        "speedup": cpu_latency_ms / npu_latency_ms,
                        "mean_abs_error": mean_error,
                        "max_abs_error": max_error,
                    }
                )

                print(
                    f"{image_path.name}: "
                    f"CPU {cpu_latency_ms:.3f} ms, "
                    f"NPU {npu_latency_ms:.3f} ms"
                )

cpu_stats = calculate_stats(cpu_latencies)
npu_stats = calculate_stats(npu_latencies)
speedup = cpu_stats["mean_ms"] / npu_stats["mean_ms"]

print_stats("CPU ONNX Runtime FP32", cpu_stats)
print_stats("Hailo-8L INT8", npu_stats)

print("\n[COMPARISON]")
print(f"Speedup: {speedup:.2f}x")
print(f"Mean absolute error: {np.mean(mean_errors):.8f}")
print(f"Maximum absolute error: {np.max(max_errors):.8f}")

with RAW_CSV_PATH.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=raw_rows[0].keys())
    writer.writeheader()
    writer.writerows(raw_rows)

summary_rows = []

for device, precision, stats in (
    ("CPU", "FP32", cpu_stats),
    ("Hailo-8L", "INT8", npu_stats),
):
    summary_rows.append(
        {
            "device": device,
            "model": "Starry Night Style Transfer",
            "precision": precision,
            "images": len(image_files),
            "warmup_runs": WARMUP_RUNS,
            **stats,
        }
    )

with SUMMARY_CSV_PATH.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=summary_rows[0].keys())
    writer.writeheader()
    writer.writerows(summary_rows)

print("\n[SAVED]")
print(RAW_CSV_PATH)
print(SUMMARY_CSV_PATH)
