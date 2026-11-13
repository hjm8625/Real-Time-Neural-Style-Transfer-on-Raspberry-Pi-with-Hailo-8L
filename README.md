# Edge NPU 기반 실시간 Neural Style Transfer

Raspberry Pi 5와 Hailo-8L NPU를 이용해 카메라 영상을 Vincent van Gogh의 *The Starry Night* 화풍으로 실시간 변환하는 프로젝트입니다.

Johnson, Alahi, Fei-Fei의 *Perceptual Losses for Real-Time Style Transfer and Super-Resolution*을 기반으로 transformation network를 PyTorch로 구현하고, Hailo-8L 배포 호환성을 고려해 일부 연산을 변경했습니다. 새로운 스타일 변환 알고리즘을 제안하기보다 논문 기반 모델의 학습, ONNX 변환, INT8 양자화, NPU 배포 및 성능 평가에 초점을 둡니다.

## 목표

직접 구현하고 학습한 PyTorch 모델을 ONNX로 변환한 뒤 Hailo-8L용 HEF로 컴파일하여 Raspberry Pi의 NPU 모듈에서 사용하는 것을 목표로 합니다.

## 모델 구조

입력과 출력은 모두 `[1, 3, 256, 256]`이며, 학습 가능한 파라미터 수는 `1,677,635`개입니다.

| 단계 | 연산 |
|---|---|
| Encoder | 9×9 Conv, 3 → 32 + BatchNorm + ReLU |
| Downsampling | 3×3 Conv, 32 → 64, stride 2 + BatchNorm + ReLU |
| Downsampling | 3×3 Conv, 64 → 128, stride 2 + BatchNorm + ReLU |
| Transformation | 128채널 Residual Block × 5 |
| Upsampling | Nearest-neighbor resize ×2 + 3×3 Conv, 128 → 64 + BatchNorm + ReLU |
| Upsampling | Nearest-neighbor resize ×2 + 3×3 Conv, 64 → 32 + BatchNorm + ReLU |
| Output | 9×9 Conv, 32 → 3 + Tanh |

### 논문 구조에서 변경한 부분

논문의 전체 채널 구성과 residual block 수는 유지하되 Hailo-8L 호환성을 위해 다음을 변경했습니다.

- Reflection padding 대신 convolution의 zero padding 사용
- Fractionally-strided convolution 대신 nearest-neighbor resize와 일반 convolution 사용
- 고정 입력 크기 256×256 사용

Reflection padding은 DFC에서 지원되지 않거나 zero padding으로 대체될 수 있으며, PyTorch `ConvTranspose2d`의 padding 및 `output_padding`은 Hailo에서 출력 정렬 문제가 발생할 수 있습니다. 현재 구조는 이러한 위험을 줄이기 위한 배포용 변형입니다.

## 학습 방법

Transformation network만 학습하며 VGG-16은 perceptual loss 계산을 위한 고정 feature network로 사용합니다. VGG는 배포 모델에 포함되지 않습니다.

```text
Total Loss = Content Weight × Content Loss
           + Style Weight × Style Loss
```

### Content loss

콘텐츠 이미지와 생성 이미지의 VGG `relu3_3` feature를 MSE로 비교합니다.

### Style loss

다음 VGG layer에서 생성 이미지와 *The Starry Night*의 Gram matrix를 비교하고 합산합니다.

- `relu1_2`
- `relu2_2`
- `relu3_3`
- `relu4_3`

Gram matrix는 feature channel 간 상관관계를 나타내며 이미지의 색상, 질감 및 반복 패턴을 표현합니다.

### VGG-16

torchvision의 `VGG16_Weights.IMAGENET1K_FEATURES`를 사용합니다. 입력을 `[0, 1]`로 복원한 뒤 해당 weights에 정의된 mean과 standard deviation으로 정규화합니다.

```text
mean = [0.48235, 0.45882, 0.40784]
std  = [1/255, 1/255, 1/255]
```

## 최종 학습 설정

| 항목 | 설정 |
|---|---:|
| Training images | 3,000장 |
| Validation images | 100장 |
| Style image | *The Starry Night* 1장 |
| Input resolution | 256×256 |
| Batch size | 8 |
| Epochs | 10 |
| Optimizer | Adam |
| Learning rate | `1e-3` |
| Content weight | `1.0` |
| Style weight | `5.0` |
| Random seed | `42` |


각 epoch가 끝나면 `outputs/`에 checkpoint와 동일한 validation 입력에 대한 sample 이미지가 저장됩니다.

```text
outputs/
├── model_epoch_001.pt
├── sample_epoch_001.jpg
├── ...
├── model_epoch_010.pt
└── sample_epoch_010.jpg
```

최종 PyTorch 모델은 `outputs/model_epoch_010.pt`를 사용합니다.

## Epoch별 결과 변화

아래 이미지는 학습에 포함되지 않은 동일한 validation 사진을 각 epoch의 모델로 변환한 결과입니다.

| Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 | Epoch 5 |
|---|---|---|---|---|
| ![Epoch 1](outputs/sample_epoch_001.jpg) | ![Epoch 2](outputs/sample_epoch_002.jpg) | ![Epoch 3](outputs/sample_epoch_003.jpg) | ![Epoch 4](outputs/sample_epoch_004.jpg) | ![Epoch 5](outputs/sample_epoch_005.jpg) |

| Epoch 6 | Epoch 7 | Epoch 8 | Epoch 9 | Epoch 10 |
|---|---|---|---|---|
| ![Epoch 6](outputs/sample_epoch_006.jpg) | ![Epoch 7](outputs/sample_epoch_007.jpg) | ![Epoch 8](outputs/sample_epoch_008.jpg) | ![Epoch 9](outputs/sample_epoch_009.jpg) | ![Epoch 10](outputs/sample_epoch_010.jpg) |

학습이 진행되면서 Starry Night의 푸른 색조와 붓 터치가 안정화되며, Epoch 10 모델을 최종 PyTorch 기준 모델로 선택했습니다.

## 데이터셋

### Content images

- Dataset: COCO 2017 validation images
- 공식 페이지: <https://cocodataset.org/#download>
- 다운로드: <http://images.cocodataset.org/zips/val2017.zip>
- 사용 방식: `data/val2017`의 이미지를 seed `42`로 섞은 뒤 `train.py`가 training 3,000장과 validation 100장으로 분리
- Annotation: 사용하지 않음

Colab에서 다음 명령으로 데이터셋을 내려받고 압축을 해제합니다.

```bash
!mkdir -p data
!curl -L --fail --retry 3 -o data/val2017.zip http://images.cocodataset.org/zips/val2017.zip
!unzip -q data/val2017.zip -d data
```

압축을 해제하면 이미지가 `data/val2017`에 저장됩니다. Neural style transfer는 콘텐츠 사진만 필요하므로 COCO annotation은 내려받거나 사용하지 않습니다. `train.py`는 다운로드를 수행하지 않으며, 준비된 이미지의 분할과 학습만 담당합니다.

`data/`의 실제 이미지 파일은 Git에 포함하지 않으며 출처 정보만 저장합니다.

### Style image

- 작품: Vincent van Gogh, *The Starry Night* (1889)
- 파일 출처: <https://commons.wikimedia.org/wiki/File:TheStarryNightByVincentVanGogh.jpg>
- 권리 표시: Public Domain Mark 1.0

스타일 이미지 한 장에서 VGG feature와 Gram matrix를 한 번 계산하고, 모든 콘텐츠 batch에 동일한 style target을 사용합니다.

## 저장소 구조

```text
npu_project/
├── README.md
├── model.py
├── train.py
├── export_onnx.py
├── verify_onnx.py
├── inference_image.py
├── realtime_camera.py
├── starry_night.onnx
├── images/
│   ├── 000000000802.jpg
│   └── starry_night.jpg
├── outputs/
│   ├── model_epoch_001.pt
│   ├── sample_epoch_001.jpg
│   ├── ...
│   ├── model_epoch_010.pt
│   └── sample_epoch_010.jpg
└── data/
    └── SOURCES.md
```

## PyTorch 검증 결과

- Epoch 10 checkpoint를 현재 `StyleTransferNet`에 정상 로드
- 누락된 state dictionary key 없음
- 예상하지 않은 state dictionary key 없음
- 입력 및 출력 shape `[1, 3, 256, 256]` 확인
- 출력에서 NaN 또는 Inf 없음
- 여러 validation 이미지에 대한 스타일 변환 확인


## ONNX 검증 결과

- 파일: `starry_night.onnx`
- Opset: PyTorch exporter가 선택하며 설치된 PyTorch 버전에 따라 달라질 수 있음
- 입력 및 출력 shape: `[1, 3, 256, 256]`
- ONNX checker: 통과
- ONNX Runtime 추론: 통과
- PyTorch 대비 최대 절대 오차: `0.0000167489`
- PyTorch 대비 평균 절대 오차: `0.0000006151`
- 연산자: `Conv`, `Relu`, `Add`, `Resize`, `Tanh`, `Constant`

## 배포 계획

```text
PyTorch checkpoint
        ↓
ONNX export 및 ONNX Runtime 검증
        ↓
Hailo calibration 및 INT8 optimization
        ↓
HEF compile
        ↓
Raspberry Pi Camera + HailoRT 실시간 추론
```

다음 단계에서는 검증된 ONNX 모델을 Hailo DFC로 parsing한 뒤 실제 사진으로 calibration을 수행하고 Hailo-8L용 HEF를 생성합니다.

## 성능 평가 계획

- ONNX Runtime CPU와 Hailo-8L 추론 latency 및 FPS 비교
- 전처리와 화면 출력을 포함한 end-to-end latency 측정
- FP32와 INT8 출력의 시각적 비교
- PSNR 및 SSIM 측정
- 256×256 기준 10 FPS 이상 목표
- 30 FPS는 확장 목표

측정하지 않은 성능을 결과로 주장하지 않으며 실제 장치, 소프트웨어 버전 및 측정 조건을 함께 기록합니다.

## 참고 자료

- Justin Johnson, Alexandre Alahi, Li Fei-Fei, *Perceptual Losses for Real-Time Style Transfer and Super-Resolution*, ECCV 2016
- 논문: <https://arxiv.org/abs/1603.08155>
- 공식 구현: <https://github.com/jcjohnson/fast-neural-style>

## 라이선스 및 출처

본 프로젝트는 Johnson et al.의 연구와 공식 구현을 참고한 재현 및 배포 프로젝트입니다. 참고한 연구, 외부 코드, COCO 데이터셋 및 스타일 이미지의 출처와 이용 조건을 명시하고, 직접 구현하거나 Hailo 배포를 위해 변경한 부분과 구분합니다.
