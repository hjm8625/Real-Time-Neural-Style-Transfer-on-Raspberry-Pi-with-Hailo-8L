# Real-Time Neural Style Transfer on an Edge NPU

This project transforms Raspberry Pi 5 camera frames into the style of Vincent van Gogh's *The Starry Night* in real time using a Hailo-8L NPU.

<p align="center">
  <img src="images/starry_night.jpg" alt="Vincent van Gogh - The Starry Night" width="280">
</p>

The PyTorch transformation network is based on Johnson, Alahi, and Fei-Fei's *Perceptual Losses for Real-Time Style Transfer and Super-Resolution*. Several operations were adapted for Hailo-8L deployment compatibility. Rather than proposing a new style transfer algorithm, this project focuses on training a paper-based model, exporting it to ONNX, applying INT8 quantization, deploying it to an NPU, and evaluating its performance.

## Goal

The goal is to implement and train a PyTorch model, export it to ONNX, compile it into a Hailo-8L HEF, and run it on the Raspberry Pi NPU module.

## Model Architecture

Both the input and output shapes are `[1, 3, 256, 256]`. The model has `1,677,635` trainable parameters.

| Stage | Operation |
|---|---|
| Encoder | 9×9 Conv, 3 → 32 + BatchNorm + ReLU |
| Downsampling | 3×3 Conv, 32 → 64, stride 2 + BatchNorm + ReLU |
| Downsampling | 3×3 Conv, 64 → 128, stride 2 + BatchNorm + ReLU |
| Transformation | Five 128-channel residual blocks |
| Upsampling | Nearest-neighbor resize ×2 + 3×3 Conv, 128 → 64 + BatchNorm + ReLU |
| Upsampling | Nearest-neighbor resize ×2 + 3×3 Conv, 64 → 32 + BatchNorm + ReLU |
| Output | 9×9 Conv, 32 → 3 + Tanh |

### Changes from the Paper Architecture

The overall channel configuration and number of residual blocks are retained, while the following changes improve Hailo-8L deployment compatibility:

- Zero padding is used instead of reflection padding.
- Nearest-neighbor resize followed by a regular convolution is used instead of fractionally strided convolution.
- A fixed input resolution of 256×256 is used.

Reflection padding may be unsupported by the Hailo Dataflow Compiler or replaced with zero padding. The padding and `output_padding` behavior of PyTorch `ConvTranspose2d` can also cause output-alignment issues during Hailo conversion. This deployment-oriented architecture avoids those operations.

## Training Method

Only the transformation network is trained. VGG-16 is used as a fixed feature network for calculating perceptual losses and is not included in the deployed model.

```text
Total Loss = Content Weight × Content Loss
           + Style Weight × Style Loss
```

### Content Loss

The VGG `relu3_3` features of the content and generated images are compared using mean squared error.

### Style Loss

The Gram matrices of the generated image and *The Starry Night* are compared and summed at the following VGG layers:

- `relu1_2`
- `relu2_2`
- `relu3_3`
- `relu4_3`

A Gram matrix represents correlations between feature channels and captures properties such as color, texture, and repeated patterns.

### VGG-16

The training code uses `VGG16_Weights.IMAGENET1K_FEATURES` from torchvision. Inputs are restored to `[0, 1]` and then normalized with the mean and standard deviation defined for those weights.

```text
mean = [0.48235, 0.45882, 0.40784]
std  = [1/255, 1/255, 1/255]
```

## Final Training Configuration

| Item | Setting |
|---|---:|
| Training images | 3,000 |
| Validation images | 100 |
| Style image | One *The Starry Night* image |
| Input resolution | 256×256 |
| Batch size | 8 |
| Epochs | 10 |
| Optimizer | Adam |
| Learning rate | `1e-3` |
| Content weight | `1.0` |
| Style weight | `5.0` |
| Random seed | `42` |

At the end of each epoch, a checkpoint and a sample generated from the same validation input are saved in `outputs/`.

```text
outputs/
├── model_epoch_001.pt
├── sample_epoch_001.jpg
├── ...
├── model_epoch_010.pt
└── sample_epoch_010.jpg
```

The final PyTorch checkpoint is `outputs/model_epoch_010.pt`.

## Results by Epoch

The following images show how the same validation image, which was excluded from the training split, changes over the ten epochs.

<p align="center">
  <img src="images/000000000802.jpg" alt="Original validation image" width="280">
  <br>
  <sub>Original validation image</sub>
</p>

| Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 | Epoch 5 |
|---|---|---|---|---|
| ![Epoch 1](outputs/sample_epoch_001.jpg) | ![Epoch 2](outputs/sample_epoch_002.jpg) | ![Epoch 3](outputs/sample_epoch_003.jpg) | ![Epoch 4](outputs/sample_epoch_004.jpg) | ![Epoch 5](outputs/sample_epoch_005.jpg) |

| Epoch 6 | Epoch 7 | Epoch 8 | Epoch 9 | Epoch 10 |
|---|---|---|---|---|
| ![Epoch 6](outputs/sample_epoch_006.jpg) | ![Epoch 7](outputs/sample_epoch_007.jpg) | ![Epoch 8](outputs/sample_epoch_008.jpg) | ![Epoch 9](outputs/sample_epoch_009.jpg) | ![Epoch 10](outputs/sample_epoch_010.jpg) |

The blue color palette and brushstroke-like textures become more stable during training. The Epoch 10 model was selected as the final PyTorch reference model.

## Epoch 10 Examples

The final Epoch 10 model was applied to five COCO images.

| Example 1 | Example 2 | Example 3 | Example 4 | Example 5 |
|---|---|---|---|---|
| ![Original 1](images/001.jpg) | ![Original 2](images/002.jpg) | ![Original 3](images/003.jpg) | ![Original 4](images/004.jpg) | ![Original 5](images/005.jpg) |
| ![Result 1](images/result_001.jpg) | ![Result 2](images/result_002.jpg) | ![Result 3](images/result_003.jpg) | ![Result 4](images/result_004.jpg) | ![Result 5](images/result_005.jpg) |

The original images are shown in the first row, and their Starry Night transformations are shown in the second row.

## Dataset

### Content Images

- Dataset: COCO 2017 validation images
- Official page: <https://cocodataset.org/#download>
- Download: <http://images.cocodataset.org/zips/val2017.zip>
- Split: `train.py` shuffles the images in `data/val2017` with seed `42`, then selects 3,000 training images and 100 validation images.
- Annotations: Not used

Download and extract the dataset in Colab with the following commands:

```bash
!mkdir -p data
!curl -L --fail --retry 3 -o data/val2017.zip http://images.cocodataset.org/zips/val2017.zip
!unzip -q data/val2017.zip -d data
```

The extracted images are stored in `data/val2017`. Neural style transfer only requires content images, so COCO annotations are neither downloaded nor used. `train.py` only splits and trains on an existing dataset; it does not download data.

Downloaded images in `data/` are excluded from Git.

### Style Image

- Artwork: Vincent van Gogh, *The Starry Night* (1889)
- Source: <https://commons.wikimedia.org/wiki/File:TheStarryNightByVincentVanGogh.jpg>
- Rights statement: Public Domain Mark 1.0

The VGG features and Gram matrices of this single style image are calculated once and reused as the style targets for every content batch.

## Repository Structure

```text
npu_project/
├── README.md
├── model.py
├── train.py
├── export_onnx.py
├── verify_onnx.py
├── calibration.py
├── compile_hef.py
├── starry_night.onnx
├── calibration/
├── calibration_output/
├── raspberry_pi/
│   ├── inference_image.py
│   ├── realtime_camera.py
│   ├── realtime_comparison.py
│   ├── cpu_npu_benchmark.py
│   ├── cpu_npu_latency.csv
│   └── cpu_npu_summary.csv
├── images/
│   ├── 000000000802.jpg
│   ├── 001.jpg ... 005.jpg
│   ├── result_001.jpg ... result_005.jpg
│   └── starry_night.jpg
├── outputs/
│   ├── model_epoch_001.pt
│   ├── sample_epoch_001.jpg
│   ├── ...
│   ├── model_epoch_010.pt
│   ├── sample_epoch_010.jpg
│   └── results.gif
└── data/
    └── val2017/
```

The dataset, calibration data, generated HAR files, and PyTorch checkpoints are excluded from Git. The ONNX model, sample images, benchmark CSV files, and real-time result GIF are included.

## PyTorch Validation

- The Epoch 10 checkpoint loads into the current `StyleTransferNet` without missing or unexpected state-dictionary keys.
- The input and output shapes are both `[1, 3, 256, 256]`.
- The output contains no NaN or Inf values.
- Style transfer was checked on multiple validation images.

## ONNX Validation

- File: `starry_night.onnx`
- Opset: Selected by the installed PyTorch exporter and therefore dependent on the PyTorch version
- Input and output shapes: `[1, 3, 256, 256]`
- ONNX checker: Passed
- ONNX Runtime inference: Passed
- Maximum absolute difference from PyTorch: `0.0000167489`
- Mean absolute difference from PyTorch: `0.0000006151`
- Operators: `Conv`, `Relu`, `Add`, `Resize`, `Tanh`, `Constant`
- Reflection padding and `ConvTranspose2d` are not used.

## Raspberry Pi CPU vs. Hailo-8L Performance

ONNX Runtime CPU FP32 inference and Hailo-8L INT8 inference were compared using 100 calibration images. The first image was used for 10 warm-up runs on each device before measurement.

| Device | Precision | Mean latency | Median | Minimum | Maximum | Standard deviation | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| CPU | FP32 | 244.988 ms | 244.884 ms | 241.268 ms | 252.237 ms | 1.709 ms | 4.08 |
| Hailo-8L | INT8 | 5.841 ms | 5.756 ms | 5.719 ms | 9.261 ms | 0.491 ms | 171.21 |

Based on mean inference latency, the Hailo-8L was approximately **41.94×** faster than the CPU. The total measured inference time for all 100 images was approximately `24.499 seconds` on the CPU and `0.584 seconds` on the Hailo-8L.

The FPS values are calculated as the reciprocal of mean inference latency. They represent model throughput only and exclude camera capture, resizing, preprocessing, postprocessing, and display time.

CPU and NPU outputs were compared in the same `[-1, 1]` range.

- Mean of per-image mean absolute error: `0.041421`
- Range of per-image mean absolute error: `0.020591–0.078583`
- Maximum absolute error observed across all images: `0.875812`

Detailed measurements are available in:

- [`raspberry_pi/cpu_npu_latency.csv`](raspberry_pi/cpu_npu_latency.csv)
- [`raspberry_pi/cpu_npu_summary.csv`](raspberry_pi/cpu_npu_summary.csv)

## Deployment Flow

```text
PyTorch checkpoint
        ↓
ONNX export and ONNX Runtime validation
        ↓
Hailo calibration and INT8 optimization
        ↓
HEF compilation
        ↓
Raspberry Pi Camera + HailoRT real-time inference
```

## AI Usage

The HailoRT inference and camera execution code for Raspberry Pi was developed with the assistance of generative AI, then adapted to the project environment and validated directly.

## References

- Justin Johnson, Alexandre Alahi, and Li Fei-Fei, *Perceptual Losses for Real-Time Style Transfer and Super-Resolution*, ECCV 2016
- Paper: <https://arxiv.org/abs/1603.08155>
- Official implementation: <https://github.com/jcjohnson/fast-neural-style>
- Seoul Metropolitan Area ICT Innovation Square NPU course

## Real-Time Inference Result

<p align="center">
  <img src="outputs/results.gif" alt="Raspberry Pi Hailo-8L real-time style transfer result" width="800">
</p>
