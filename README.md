# Movement Chain ML

Golf swing analysis pipeline combining vision (MediaPipe) and IMU sensor data.

## Features

- **Vision Analysis**: MediaPipe Pose Landmarker for 33-point skeleton tracking
- **IMU Processing**: 8-phase swing detection with 7 performance metrics
- **Sensor Fusion**: Time-aligned Vision + IMU data using Impact as anchor
- **3D Visualization**: Rerun SDK for interactive swing playback
- **AI Coaching**: Structured Kinematic Prompts for LLM feedback generation

## Quick Start

### Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package with all dependencies
pip install -e ".[all]"

# Or install minimal + specific extras
pip install -e .              # Core only
pip install -e ".[viz]"       # + Rerun visualization
pip install -e ".[dev]"       # + Development tools

# Set up pre-commit hooks
pre-commit install
pre-commit install --hook-type commit-msg
```

### Usage

#### CLI

```bash
# Run the analysis pipeline
movement-chain --video data/swing.mp4 --imu data/imu_data.csv

# With Rerun visualization
movement-chain --video data/swing.mp4 --imu data/imu_data.csv --spawn-viewer

# Specify output directory
movement-chain --video swing.mp4 --imu swing.csv --output output/session1
```

#### Python API

```python
from movement_chain import run_pipeline, VisionAnalyzer, AICoach

# Full pipeline
result = run_pipeline("video.mp4", "imu.csv")
print(result.ai_prompt)

# Individual components
analyzer = VisionAnalyzer()
vision_result = analyzer.analyze_video("video.mp4")
```

## Project Structure

```
movement-chain-ml/
├── src/
│   └── movement_chain/       # Main package
│       ├── __init__.py       # Package exports
│       ├── schemas.py        # Data structures & constants
│       ├── vision_analyzer.py    # MediaPipe pose analysis
│       ├── imu_swing_analyzer.py # IMU phase detection
│       ├── sensor_fusion.py      # Vision + IMU alignment
│       ├── rerun_visualizer.py   # 3D visualization
│       ├── ai_coach.py           # Kinematic prompt generation
│       └── pipeline.py           # CLI & orchestration
├── scripts/
│   └── test_mediapipe.py     # Standalone MediaPipe testing tool
├── tests/                    # Unit tests
├── docs/                     # Documentation
│   ├── FUSION_PIPELINE_PLAN.md
│   ├── IMPLEMENTATION_SUMMARY.md
│   ├── QUICK_REFERENCE.md
│   └── SETUP_INSTRUCTIONS.md
├── data/                     # Sample data
├── models/                   # ML model files
├── notebooks/                # Jupyter notebooks
├── output/                   # Analysis outputs
├── .github/workflows/        # CI/CD
├── pyproject.toml           # Project config
├── .pre-commit-config.yaml  # Git hooks
└── requirements.txt         # Dependencies
```

## Development

### Setup

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
pre-commit install --hook-type commit-msg
```

### Code Quality

```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint
ruff check --fix src/ tests/

# Type check
mypy src/

# Run all checks
pre-commit run --all-files
```

### Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=src/movement_chain --cov-report=html

# Specific test
pytest tests/test_example.py -v
```

### Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```bash
git commit -m "feat(analysis): add new swing metric"
git commit -m "fix(vision): resolve keypoint detection issue"
git commit -m "docs: update API documentation"
```

**Types:** `feat`, `fix`, `docs`, `chore`, `refactor`, `test`, `perf`, `ci`, `style`

## Pipeline Architecture

```
Video (30fps) ─┬─→ VisionAnalyzer ──┐
               │   (MediaPipe)       │
               │                     ├─→ SensorFusion ──┬─→ RerunVisualizer
               │                     │   (Impact sync)   │   (3D playback)
IMU (1000Hz) ──┴─→ IMUAnalyzer ─────┘                   │
                   (8 phases)                           └─→ AICoach
                                                            (Kinematic Prompt)
```

## Documentation

- [Fusion Pipeline Plan](docs/FUSION_PIPELINE_PLAN.md) - Architecture & design
- [Implementation Summary](docs/IMPLEMENTATION_SUMMARY.md) - Module details
- [Quick Reference](docs/QUICK_REFERENCE.md) - Developer reference
- [Setup Instructions](docs/SETUP_INSTRUCTIONS.md) - Detailed setup guide

## License

MIT
