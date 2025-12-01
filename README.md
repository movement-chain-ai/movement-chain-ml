# Movement Chain AI - Machine Learning

ML models and training pipelines for real-time movement analysis, error detection, and corrective feedback in the Movement Chain AI system.

## Overview

This repository contains:

- **Pose estimation models** (RTMPose-m, YOLO11 Pose)
- **Temporal modeling** (LSTM + Transformer hybrid for movement sequences)
- **Error detection algorithms** (kinematic, timing, muscle pattern analysis)
- **Model optimization** (ONNX conversion for cross-platform deployment)
- **Training pipelines** (data preprocessing, augmentation, evaluation)

## Architecture

See [full documentation](https://movement-chain-ai.github.io/system-documentation/) for system architecture.

### Model Pipeline

```
Camera Input (60fps) → RTMPose-m → 17 keypoints (34D)
IMU Data (100Hz) → Feature Extraction → 6D vector
EMG Data (200Hz) → Signal Processing → 4D vector
Metadata → User Profile + Context → 7D vector
                    ↓
        Feature Fusion → 51D input vector
                    ↓
    LSTM + Transformer → Movement sequence analysis
                    ↓
    Error Detection → Correction suggestions
```

## Planned Directory Structure

```
movement-chain-ml/
├── training/
│   ├── datasets/
│   │   ├── golf_swing/              # Golf movement datasets
│   │   ├── bicep_curl/              # Workout datasets (MVP)
│   │   └── data_loaders.py          # PyTorch data loading
│   ├── models/
│   │   ├── pose_estimation/
│   │   │   ├── rtmpose_m.py         # RTMPose-m architecture
│   │   │   └── yolo11_pose.py       # YOLO11 Pose alternative
│   │   ├── temporal/
│   │   │   ├── lstm_transformer.py  # Hybrid temporal model
│   │   │   └── attention.py         # Custom attention layers
│   │   └── error_detection/
│   │       ├── kinematic_analyzer.py
│   │       └── form_classifier.py
│   ├── train.py                     # Training script
│   ├── evaluate.py                  # Model evaluation
│   └── config/
│       ├── golf_config.yaml         # Golf-specific config
│       └── workout_config.yaml      # Workout config
├── inference/
│   ├── onnx_export.py               # ONNX model conversion
│   ├── tflite_export.py             # TensorFlow Lite export
│   ├── quantization.py              # Model quantization (INT8)
│   └── benchmark.py                 # Performance benchmarking
├── models/
│   ├── rtmpose_m_golf.onnx         # Compiled models
│   ├── lstm_transformer.onnx
│   └── error_detector.tflite
├── data/
│   ├── preprocessing/
│   │   ├── camera_calibration.py   # Camera intrinsics
│   │   ├── imu_filtering.py        # Sensor fusion
│   │   └── emg_normalization.py    # EMG signal processing
│   ├── augmentation/
│   │   ├── pose_augment.py         # Keypoint augmentation
│   │   └── temporal_augment.py     # Sequence augmentation
│   └── annotation/
│       ├── label_studio_export.py  # Annotation tools
│       └── ground_truth.py         # Expert labels
├── notebooks/
│   ├── exploratory_analysis.ipynb  # Data exploration
│   ├── model_comparison.ipynb      # Benchmark results
│   └── error_patterns.ipynb        # Error taxonomy analysis
├── tests/
│   ├── test_models.py              # Unit tests
│   ├── test_inference.py           # Inference tests
│   └── test_data_pipeline.py       # Data loading tests
├── requirements.txt                 # Python dependencies
├── setup.py                         # Package installation
├── .gitignore
├── LICENSE                          # Apache 2.0
└── README.md
```

## Development Setup

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU training)
- 16GB+ RAM (32GB recommended)

### Installation

```bash
# Clone repository
git clone https://github.com/movement-chain-ai/movement-chain-ml.git
cd movement-chain-ml

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Dependencies

```
# Core ML frameworks
torch>=2.0.0
onnx>=1.14.0
onnxruntime>=1.15.0
tensorflow-lite>=2.13.0

# Pose estimation
mmpose>=1.1.0
mmcv>=2.0.0
ultralytics>=8.0.0

# Data processing
numpy>=1.24.0
pandas>=2.0.0
opencv-python>=4.8.0
scikit-learn>=1.3.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0

# Development
pytest>=7.4.0
black>=23.7.0
flake8>=6.0.0
```

## Model Zoo

### Pose Estimation Models

| Model | Input | Output | FPS (Intel i7) | AP | Status |
|-------|-------|--------|----------------|-----|--------|
| RTMPose-m | 256x192 | 17 keypoints | 90+ | 75.8% | ✅ Recommended |
| YOLO11 Pose | 640x640 | 17 keypoints | 60+ | 89.4% | ⚠️ Alternative |

### Temporal Models

| Model | Input | Output | Latency | Accuracy | Status |
|-------|-------|--------|---------|----------|--------|
| LSTM + Transformer | 51D × 30 frames | Error probabilities | 15ms | TBD | 🚧 In Progress |

### Deployment Formats

- **ONNX Runtime** (Mobile app inference)
- **TensorFlow Lite** (Mobile GPU acceleration)
- **TFLite Micro** (ESP32-S3 edge inference - future)

## Training Workflow

```bash
# 1. Prepare dataset
python data/preprocessing/prepare_dataset.py --movement golf_swing

# 2. Train pose estimation model
python training/train.py --config config/golf_config.yaml --model rtmpose_m

# 3. Train temporal model
python training/train.py --config config/golf_config.yaml --model lstm_transformer

# 4. Export to ONNX
python inference/onnx_export.py --checkpoint checkpoints/best_model.pth

# 5. Benchmark performance
python inference/benchmark.py --model models/rtmpose_m_golf.onnx
```

## Model Deployment

### ONNX Runtime (Mobile App)

```python
import onnxruntime as ort

session = ort.InferenceSession("models/rtmpose_m_golf.onnx")
outputs = session.run(None, {"input": camera_frame})
keypoints = outputs[0]
```

### TensorFlow Lite (Flutter)

```dart
import 'package:tflite_flutter/tflite_flutter.dart';

final interpreter = await Interpreter.fromAsset('rtmpose_m_golf.tflite');
interpreter.run(inputTensor, outputTensor);
```

## Performance Targets

- **Pose Estimation:** <30ms latency on mobile GPU
- **Temporal Analysis:** <50ms for 30-frame window
- **End-to-End:** <100ms total feedback latency
- **Accuracy:** >85% error detection rate

See [performance targets documentation](https://movement-chain-ai.github.io/system-documentation/latest/architecture/hld/04-performance-targets/).

## Contributing

We welcome contributions! Submit pull requests with:
- ✅ Unit tests for new models
- ✅ Benchmark results comparison
- ✅ Documentation updates

Branch protection requires:
- At least 1 approving review
- All CI checks passing

## License

Apache License 2.0 - see [LICENSE](LICENSE) file for details.

## Documentation

Full system documentation: https://movement-chain-ai.github.io/system-documentation/

## Related Repositories

- [movement-chain-mobile](https://github.com/movement-chain-ai/movement-chain-mobile) - Flutter app (model deployment)
- [movement-chain-firmware](https://github.com/movement-chain-ai/movement-chain-firmware) - ESP32 firmware
- [movement-chain-hardware](https://github.com/movement-chain-ai/movement-chain-hardware) - Hardware schematics
