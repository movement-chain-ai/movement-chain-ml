"""
Movement Chain - 统一数据结构定义

定义 Vision、IMU、融合数据的统一 Schema，用于 Pipeline 各模块间数据交换。

Author: Movement Chain AI Team
Date: 2025-01-15
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# ============================================================
# Vision 数据结构
# ============================================================


@dataclass
class PoseFrame:
    """单帧姿态数据 (来自 MediaPipe)"""

    frame_idx: int
    timestamp_ms: float
    landmarks: list[list[float]]  # 33 关键点 [[x, y, z, visibility], ...]

    # 计算得出的角度指标
    left_arm_angle: float | None = None  # 肩-肘-腕夹角 (度)
    right_arm_angle: float | None = None
    x_factor: float | None = None  # 肩线与髋线夹角 (度)

    # 关键点速度 (用于 Impact 检测)
    wrist_speed: float | None = None  # 手腕速度 (像素/帧)


@dataclass
class VisionMetrics:
    """视觉分析指标汇总 (6 项核心指标)"""

    # 姿势评估
    address_posture_score: float | None = None  # 准备姿势评分 (0-100)
    spine_angle_deg: float | None = None  # 脊柱倾斜角 (度)

    # 稳定性
    head_movement_cm: float | None = None  # 头部移动距离 (厘米)

    # 旋转
    x_factor_max_deg: float | None = None  # 最大 X-Factor (度)
    hip_rotation_deg: float | None = None  # 髋部旋转角度 (度)

    # 手臂
    lead_arm_extension_pct: float | None = None  # 前臂伸展百分比 (0-100)

    # 顶点位置
    top_position_frame: int | None = None  # 顶点帧索引
    impact_frame: int | None = None  # Impact 帧索引


@dataclass
class VisionResult:
    """视频分析完整结果"""

    video_file: str
    fps: float
    total_frames: int
    width: int
    height: int

    frames: list[PoseFrame]
    metrics: VisionMetrics

    # Impact 检测
    impact_frame_idx: int | None = None
    impact_confidence: float | None = None


# ============================================================
# IMU 数据结构 (兼容 imu_swing_analyzer.py)
# ============================================================


@dataclass
class IMUFrame:
    """单帧 IMU 数据"""

    timestamp_ms: float
    gyro_dps: tuple[float, float, float]  # GyX, GyY, GyZ (度/秒)
    accel_g: tuple[float, float, float]  # AcX, AcY, AcZ (g)
    gyro_magnitude: float  # 陀螺仪合成值


# 注意: SwingPhase 和 SwingMetrics 定义在 imu_swing_analyzer.py 中
# 这里导入使用，避免重复定义


# ============================================================
# 融合数据结构
# ============================================================


@dataclass
class FusedFrame:
    """单帧融合数据 (Vision + IMU 时间对齐后)"""

    timestamp_ms: float  # 统一时间戳 (IMU 时钟基准)
    frame_idx: int  # 对应的 IMU 数据索引

    # Vision 数据 (插值后)
    has_vision: bool = False
    pose_landmarks: list[list[float]] | None = None  # 33 关键点
    left_arm_angle: float | None = None
    right_arm_angle: float | None = None
    x_factor: float | None = None

    # IMU 数据
    gyro_dps: tuple[float, float, float] | None = None
    accel_g: tuple[float, float, float] | None = None
    gyro_magnitude: float | None = None

    # 阶段标记
    phase_name: str | None = None
    phase_name_cn: str | None = None


@dataclass
class FusedSwingData:
    """完整挥杆融合数据"""

    # 元数据
    session_id: str
    video_file: str
    imu_file: str
    analysis_time: str
    alignment_offset_ms: float  # Vision 时间偏移量

    # 融合帧数据
    frames: list[FusedFrame]

    # Vision 结果
    vision_fps: float
    vision_metrics: VisionMetrics

    # IMU 结果 (使用字典存储，兼容 imu_swing_analyzer 输出)
    imu_phases: list[dict[str, Any]]  # SwingPhase 转换为 dict
    imu_metrics: dict[str, Any]  # SwingMetrics 转换为 dict
    imu_report: dict[str, Any]  # 完整报告

    def to_dict(self) -> dict[str, Any]:
        """转换为字典 (用于 JSON 序列化)"""
        return {
            "session_id": self.session_id,
            "video_file": self.video_file,
            "imu_file": self.imu_file,
            "analysis_time": self.analysis_time,
            "alignment_offset_ms": self.alignment_offset_ms,
            "vision_fps": self.vision_fps,
            "vision_metrics": asdict(self.vision_metrics),
            "imu_phases": self.imu_phases,
            "imu_metrics": self.imu_metrics,
            "frame_count": len(self.frames),
            # 不保存完整帧数据，太大
        }


# ============================================================
# AI Coach 数据结构
# ============================================================


@dataclass
class Issue:
    """检测到的问题"""

    category: str  # tempo, posture, rotation, timing, stability
    severity: str  # info, warning, critical
    metric_name: str
    actual_value: float
    benchmark_range: str
    description: str
    suggestion: str


@dataclass
class PhaseAnalysis:
    """单阶段分析结果"""

    phase: str
    phase_cn: str
    duration_ms: float
    issues: list[str]
    benchmark_comparison: str


@dataclass
class KinematicPrompt:
    """AI 输入结构 (Kinematic Prompt)"""

    session_id: str
    analysis_time: str

    # 摘要
    overall_level: str  # 初级, 中级, 高级, 职业
    key_issues: list[str]

    # 完整指标
    imu_metrics: dict[str, Any]
    vision_metrics: dict[str, Any]

    # 阶段分析
    phase_analysis: list[PhaseAnalysis]

    # 检测到的问题
    issues: list[Issue]

    # 可视化文件
    visualization_file: str | None = None

    # 生成的 AI Prompt 文本
    ai_prompt: str = ""

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "session_id": self.session_id,
            "analysis_time": self.analysis_time,
            "summary": {
                "overall_level": self.overall_level,
                "key_issues": self.key_issues,
            },
            "metrics": {
                "imu": self.imu_metrics,
                "vision": self.vision_metrics,
            },
            "phase_analysis": [asdict(p) for p in self.phase_analysis],
            "issues": [asdict(i) for i in self.issues],
            "visualization_file": self.visualization_file,
            "ai_prompt": self.ai_prompt,
        }

    def save(self, output_path: str) -> None:
        """保存为 JSON 文件"""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"✅ Kinematic Prompt 已保存: {output_path}")


# ============================================================
# 常量定义
# ============================================================

# BlazePose 33 关键点连接定义
POSE_CONNECTIONS = [
    # 面部
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),  # 左眼
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),  # 右眼
    (9, 10),  # 嘴巴
    # 躯干
    (11, 12),  # 肩膀连线
    (11, 23),
    (12, 24),  # 肩膀到髋部
    (23, 24),  # 髋部连线
    # 左臂
    (11, 13),
    (13, 15),  # 肩-肘-腕
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),  # 手部
    # 右臂
    (12, 14),
    (14, 16),  # 肩-肘-腕
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),  # 手部
    # 左腿
    (23, 25),
    (25, 27),  # 髋-膝-踝
    (27, 29),
    (27, 31),
    (29, 31),  # 脚部
    # 右腿
    (24, 26),
    (26, 28),  # 髋-膝-踝
    (28, 30),
    (28, 32),
    (30, 32),  # 脚部
]

# 高尔夫关键点索引
GOLF_KEY_LANDMARKS = {
    0: "NOSE",
    11: "LEFT_SHOULDER",
    12: "RIGHT_SHOULDER",
    13: "LEFT_ELBOW",
    14: "RIGHT_ELBOW",
    15: "LEFT_WRIST",
    16: "RIGHT_WRIST",
    23: "LEFT_HIP",
    24: "RIGHT_HIP",
    25: "LEFT_KNEE",
    26: "RIGHT_KNEE",
    27: "LEFT_ANKLE",
    28: "RIGHT_ANKLE",
}

# 挥杆阶段颜色 (用于 Rerun 可视化)
PHASE_COLORS = {
    "address": (0.5, 0.5, 0.5),  # 灰色
    "takeaway": (0.2, 0.6, 1.0),  # 蓝色
    "backswing": (0.2, 0.8, 0.2),  # 绿色
    "top": (1.0, 0.8, 0.0),  # 黄色
    "downswing": (1.0, 0.5, 0.0),  # 橙色
    "impact": (1.0, 0.0, 0.0),  # 红色
    "follow_through": (0.8, 0.2, 0.8),  # 紫色
    "finish": (0.4, 0.4, 0.4),  # 深灰
}

# 职业选手基准数据
BENCHMARKS = {
    # IMU 指标
    "peak_angular_velocity_dps": {
        "beginner": (0, 600),
        "amateur": (600, 1000),
        "advanced": (1000, 1500),
        "professional": (1500, 3000),
    },
    "tempo_ratio": {
        "ideal": (2.5, 3.5),
        "amateur": (2.0, 2.5),
    },
    "backswing_duration_ms": {
        "beginner": (1000, 1500),
        "amateur": (850, 1000),
        "advanced": (700, 850),
        "professional": (700, 800),
    },
    "downswing_duration_ms": {
        "beginner": (350, 500),
        "amateur": (300, 350),
        "advanced": (250, 300),
        "professional": (230, 280),
    },
    # Vision 指标
    "x_factor_max_deg": {
        "beginner": (20, 35),
        "amateur": (35, 45),
        "professional": (45, 60),
    },
    "head_movement_cm": {
        "excellent": (0, 5),
        "good": (5, 10),
        "poor": (10, 20),
    },
    "lead_arm_extension_pct": {
        "poor": (0, 70),
        "amateur": (70, 85),
        "professional": (85, 100),
    },
}
