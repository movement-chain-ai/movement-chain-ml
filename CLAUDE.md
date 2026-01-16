# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Movement Chain ML is a golf swing analysis platform that combines computer vision (MediaPipe) and IMU sensor data to provide AI-powered coaching feedback. The system fuses 30fps video with 1000Hz IMU data, detects 8 swing phases, and generates structured kinematic prompts for LLM-based coaching.

## Development Commands

```bash
# Install (choose one)
pip install -e .              # Core only
pip install -e ".[viz]"       # With Rerun visualization
pip install -e ".[dev]"       # Development tools
pip install -e ".[all]"       # Everything

# Testing
pytest                                    # All tests with coverage
pytest tests/test_example.py -v           # Single test file
pytest -x                                 # Stop on first failure

# Code Quality
black src/ tests/                         # Format code
ruff check --fix src/ tests/              # Lint and auto-fix
isort src/ tests/                         # Sort imports
mypy src/                                 # Type checking
pre-commit run --all-files                # Run all checks

# CLI Usage
movement-chain --video data/swing.mp4 --imu data/imu.csv --spawn-viewer
```

## Architecture

```
Video (30fps) ──┐
                ├─→ VisionAnalyzer (MediaPipe) ──┐
                │                                 ├─→ SensorFusion ──┐
IMU (1000Hz) ──┘                                 │   (Impact sync)  ├─→ RerunVisualizer
                └─→ IMUAnalyzer (8 phases) ─────┘                   │   (3D playback)
                                                  └─→ AICoach
                                                     (Kinematic Prompt)
```

### Core Modules (`src/movement_chain/`)

| Module | Purpose |
|--------|---------|
| `schemas.py` | Dataclasses for all data types (Chinese docstrings) |
| `pipeline.py` | CLI entry point + orchestration |
| `vision_analyzer.py` | MediaPipe pose detection (33 landmarks) |
| `imu_swing_analyzer.py` | 8-phase detection + swing metrics |
| `sensor_fusion.py` | Time alignment via impact synchronization |
| `rerun_visualizer.py` | 3D visualization export (.rrd files) |
| `ai_coach.py` | Kinematic prompt generation for LLM |

### Data Flow

1. **Vision**: Video → 33-point skeleton per frame → arm angles, x-factor, head movement
2. **IMU**: CSV → gyro magnitude peaks → 8 phases (Address→Finish) + 7 metrics
3. **Fusion**: Align at impact moment, interpolate vision to IMU timestamps
4. **Output**: `FusedSwingData` → Kinematic prompt JSON + optional .rrd visualization

### Swing Phases (8 total)
Address → Takeaway → Backswing → Top → Downswing → Impact → Follow-through → Finish

### Key Metrics
- Vision: arm angles, x-factor, head stability, lead arm extension
- IMU: peak velocity (°/s), tempo ratio, backswing/downswing duration, wrist release point

## Important Patterns

### MediaPipe Usage
Uses Tasks API (0.10.x), NOT legacy solutions API. Model file: `models/pose_landmarker_full.task`

### Type System
All data structures in `schemas.py` use Python dataclasses with full type hints. Schema documentation is in Chinese.

### Optional Visualization
Rerun is an optional dependency. Code gracefully falls back if not installed.

### Conventional Commits
Format: `<type>(<scope>): <subject>`
Types: `feat`, `fix`, `docs`, `chore`, `refactor`, `test`, `perf`, `ci`
Example: `feat(analysis): add fusion pipeline`

## Pre-commit Hooks

Enforced on every commit:
- **commit-msg**: Validates Conventional Commits format
- **pre-commit**: Black, Ruff, isort, nbstripout
- **pre-push**: mypy + pytest (optional)

## Test Data

Sample files in `data/`:
- `IMG_1377.MOV` - Test video
- `imu_20260112_171520.csv` - Test IMU data
