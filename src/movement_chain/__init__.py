"""Movement Chain ML - Golf Swing Analysis Pipeline.

This package provides tools for analyzing golf swings using:
- Vision analysis (MediaPipe pose detection)
- IMU sensor data processing
- Sensor fusion for combined analysis
- 3D visualization with Rerun
- AI-powered coaching feedback

Usage:
    from movement_chain import run_pipeline
    result = run_pipeline("video.mp4", "imu.csv")
"""

__version__ = "2.0.0"

from .ai_coach import AICoach
from .imu_swing_analyzer import analyze_swing
from .pipeline import run_pipeline
from .rerun_visualizer import RerunVisualizer
from .schemas import (
    BENCHMARKS,
    FusedFrame,
    FusedSwingData,
    Issue,
    KinematicPrompt,
    PhaseAnalysis,
    PoseFrame,
    VisionMetrics,
    VisionResult,
)
from .sensor_fusion import SensorFusion
from .vision_analyzer import VisionAnalyzer

__all__ = [
    # Main pipeline
    "run_pipeline",
    # Classes
    "AICoach",
    "RerunVisualizer",
    "SensorFusion",
    "VisionAnalyzer",
    # Functions
    "analyze_swing",
    # Data structures
    "FusedFrame",
    "FusedSwingData",
    "Issue",
    "KinematicPrompt",
    "PhaseAnalysis",
    "PoseFrame",
    "VisionMetrics",
    "VisionResult",
    # Constants
    "BENCHMARKS",
]
