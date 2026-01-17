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


@dataclass
class SwingPhase:
    """挥杆阶段数据"""

    name: str  # 阶段英文名 (Address, Takeaway, Backswing, Top, Downswing, Impact, Follow-through, Finish)
    name_cn: str  # 阶段中文名
    start_idx: int  # 起始索引
    end_idx: int  # 结束索引
    start_time_ms: float  # 起始时间 (毫秒)
    end_time_ms: float  # 结束时间 (毫秒)
    duration_ms: float  # 持续时间 (毫秒)
    peak_gyro_dps: float  # 阶段内峰值角速度 (度/秒)

    def __post_init__(self) -> None:
        """Validate invariants at construction time."""
        import numpy as np

        if self.end_idx < self.start_idx:
            raise ValueError(
                f"SwingPhase '{self.name}': end_idx ({self.end_idx}) < start_idx ({self.start_idx})"
            )
        if self.end_time_ms < self.start_time_ms:
            raise ValueError(
                f"SwingPhase '{self.name}': end_time_ms ({self.end_time_ms}) < start_time_ms ({self.start_time_ms})"
            )
        if self.peak_gyro_dps < 0 or (
            isinstance(self.peak_gyro_dps, float) and np.isnan(self.peak_gyro_dps)
        ):
            raise ValueError(
                f"SwingPhase '{self.name}': invalid peak_gyro_dps ({self.peak_gyro_dps})"
            )


@dataclass
class SwingMetrics:
    """挥杆指标 - 完整 7 项 IMU 指标"""

    # 核心 5 项
    peak_angular_velocity_dps: float  # 峰值角速度 (度/秒)
    backswing_duration_ms: float  # 上杆时长 (毫秒)
    downswing_duration_ms: float  # 下杆时长 (毫秒)
    total_swing_time_ms: float  # 总挥杆时间 (毫秒)
    tempo_ratio: float | None  # 节奏比 (上杆/下杆), 下杆为0时为None

    # 新增 2 项
    wrist_release_point_pct: float | None  # 手腕释放点 (下杆完成百分比)
    acceleration_time_ms: float | None  # 加速时段 (毫秒)

    # 评估等级
    velocity_level: str  # 峰值速度等级
    tempo_level: str  # 节奏等级
    wrist_release_level: str  # 手腕释放等级
    acceleration_level: str  # 加速等级
    overall_level: str  # 综合等级

    def __post_init__(self) -> None:
        """Validate invariants at construction time."""
        if self.peak_angular_velocity_dps < 0:
            raise ValueError(
                f"peak_angular_velocity_dps must be non-negative, got {self.peak_angular_velocity_dps}"
            )
        if self.backswing_duration_ms < 0:
            raise ValueError(
                f"backswing_duration_ms must be non-negative, got {self.backswing_duration_ms}"
            )
        if self.downswing_duration_ms < 0:
            raise ValueError(
                f"downswing_duration_ms must be non-negative, got {self.downswing_duration_ms}"
            )
        if self.wrist_release_point_pct is not None:
            if not (0 <= self.wrist_release_point_pct <= 100):
                raise ValueError(
                    f"wrist_release_point_pct must be in [0, 100], got {self.wrist_release_point_pct}"
                )


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

    # IMU 结果 (直接使用 dataclass，在序列化时转换为 dict)
    imu_phases: list[SwingPhase]
    imu_metrics: SwingMetrics
    imu_report: dict[str, Any]  # 完整报告

    # V2: 逐阶段指标 (由 SensorFusion.aggregate_phase_metrics 填充)
    phase_metrics: list[Any] | None = None  # list[PerPhaseMetrics], 使用 Any 避免循环引用

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
            "imu_phases": [asdict(p) for p in self.imu_phases],
            "imu_metrics": asdict(self.imu_metrics),
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
# V2 Per-Phase Metrics 数据结构
# ============================================================


@dataclass
class SensorAvailability:
    """传感器可用性状态"""

    vision: bool = False
    imu: bool = False
    emg: bool = False  # 预留 EMG 扩展
    vision_frame_count: int = 0
    imu_frame_count: int = 0
    emg_frame_count: int = 0


@dataclass
class PhaseTimingMetrics:
    """阶段时间指标"""

    start_ms: float
    end_ms: float
    duration_ms: float
    frame_count: int


@dataclass
class PhaseIMUMetrics:
    """阶段 IMU 指标 (陀螺仪统计)"""

    gyro_magnitude_max: float | None = None
    gyro_magnitude_avg: float | None = None
    gyro_magnitude_min: float | None = None
    gyro_stability_score: float | None = None  # 标准差的倒数归一化
    gyro_x_max: float | None = None
    gyro_y_max: float | None = None
    gyro_z_max: float | None = None


@dataclass
class PhaseVisionMetrics:
    """阶段视觉指标"""

    x_factor_start: float | None = None
    x_factor_end: float | None = None
    x_factor_max: float | None = None
    x_factor_min: float | None = None
    x_factor_delta: float | None = None
    left_arm_angle_avg: float | None = None
    right_arm_angle_avg: float | None = None
    head_displacement_cm: float | None = None


@dataclass
class PhaseEMGMetrics:
    """阶段 EMG 指标 (预留扩展)"""

    core_activation_pct: float | None = None
    forearm_activation_pct: float | None = None
    timing_gap_ms: float | None = None


# ============================================================
# 阶段特定指标 (8 个阶段)
# ============================================================


@dataclass
class AddressPhaseMetrics:
    """Address 阶段特定指标 (准备)"""

    stability_score: float | None = None  # 静止稳定性 0-100
    spine_angle_deg: float | None = None  # 脊柱前倾角
    stance_width_ratio: float | None = None  # 站位宽度/肩宽


@dataclass
class TakeawayPhaseMetrics:
    """Takeaway 阶段特定指标 (启动)"""

    initial_acceleration_dps2: float | None = None  # 初始加速度
    rotation_start_ms: float | None = None  # 旋转开始时间


@dataclass
class BackswingPhaseMetrics:
    """Backswing 阶段特定指标 (上杆)"""

    x_factor_buildup_rate: float | None = None  # X-Factor 增长速率 (°/s)
    shoulder_turn_deg: float | None = None  # 肩部旋转角度
    hip_turn_deg: float | None = None  # 髋部旋转角度
    sway_cm: float | None = None  # 横向晃动距离


@dataclass
class TopPhaseMetrics:
    """Top 阶段特定指标 (顶点)"""

    x_factor_max_deg: float | None = None  # 最大 X-Factor
    lead_arm_extension_pct: float | None = None  # 前臂伸展百分比
    pause_duration_ms: float | None = None  # 顶点停顿时间


@dataclass
class DownswingPhaseMetrics:
    """Downswing 阶段特定指标 (下杆)"""

    peak_velocity_dps: float | None = None  # 峰值角速度
    acceleration_rate_dps2: float | None = None  # 加速率
    hip_lead_ms: float | None = None  # 髋部领先时间
    wrist_release_point_pct: float | None = None  # 手腕释放点 (0-100%)


@dataclass
class ImpactPhaseMetrics:
    """Impact 阶段特定指标 (击球)"""

    velocity_at_impact_dps: float | None = None  # 击球时角速度
    head_stable: bool | None = None  # 头部是否稳定
    weight_shift_pct: float | None = None  # 重心转移百分比


@dataclass
class FollowThroughPhaseMetrics:
    """Follow-through 阶段特定指标 (送杆)"""

    deceleration_rate_dps2: float | None = None  # 减速率
    rotation_completion_pct: float | None = None  # 旋转完成度


@dataclass
class FinishPhaseMetrics:
    """Finish 阶段特定指标 (收杆)"""

    final_stability_score: float | None = None  # 最终稳定性
    balance_score: float | None = None  # 平衡评分


@dataclass
class PerPhaseMetrics:
    """单阶段完整指标容器"""

    phase_name: str
    phase_name_cn: str
    timing: PhaseTimingMetrics

    # 通用传感器指标 (可选)
    imu: PhaseIMUMetrics | None = None
    vision: PhaseVisionMetrics | None = None
    emg: PhaseEMGMetrics | None = None

    # 阶段特定指标 (仅一个非空)
    address: AddressPhaseMetrics | None = None
    takeaway: TakeawayPhaseMetrics | None = None
    backswing: BackswingPhaseMetrics | None = None
    top: TopPhaseMetrics | None = None
    downswing: DownswingPhaseMetrics | None = None
    impact: ImpactPhaseMetrics | None = None
    follow_through: FollowThroughPhaseMetrics | None = None
    finish: FinishPhaseMetrics | None = None


# ============================================================
# V2 规则触发器 (纯布尔值)
# ============================================================


@dataclass
class DiagnosticRuleTriggers:
    """诊断规则触发器 (仅布尔值，无硬编码文本)"""

    # P0 - 关键问题
    tempo_ratio_outside_ideal: bool = False
    head_movement_excessive: bool = False
    x_factor_insufficient: bool = False

    # P1 - 警告
    backswing_too_fast: bool = False
    downswing_too_slow: bool = False
    lead_arm_bent: bool = False
    early_wrist_release: bool = False

    # P2 - 信息
    velocity_below_amateur: bool = False

    # 统计
    rules_evaluated: int = 0
    rules_triggered: int = 0


@dataclass
class KinematicPromptV2:
    """V2 AI 输入结构 (纯数据 + 布尔规则触发)"""

    session_id: str
    analysis_time: str
    schema_version: str = "2.0"

    # 传感器状态
    sensors: SensorAvailability = None  # type: ignore

    # 摘要
    overall_level: str = ""

    # 全局指标
    imu_global_metrics: dict[str, Any] = None  # type: ignore
    vision_global_metrics: dict[str, Any] = None  # type: ignore

    # 逐阶段指标
    phases: list[PerPhaseMetrics] = None  # type: ignore

    # 规则触发器
    rule_triggers: DiagnosticRuleTriggers = None  # type: ignore

    # 可视化文件
    visualization_file: str | None = None

    def __post_init__(self):
        """初始化默认值"""
        if self.sensors is None:
            self.sensors = SensorAvailability()
        if self.imu_global_metrics is None:
            self.imu_global_metrics = {}
        if self.vision_global_metrics is None:
            self.vision_global_metrics = {}
        if self.phases is None:
            self.phases = []
        if self.rule_triggers is None:
            self.rule_triggers = DiagnosticRuleTriggers()

    def to_dict(self) -> dict[str, Any]:
        """转换为字典 (符合 V2 JSON 结构)"""
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "analysis_time": self.analysis_time,
            "sensors": asdict(self.sensors),
            "summary": {
                "overall_level": self.overall_level,
                "rules_triggered_count": self.rule_triggers.rules_triggered,
            },
            "metrics": {
                "global": {
                    "imu": self.imu_global_metrics,
                    "vision": self.vision_global_metrics,
                },
                "per_phase": [self._phase_to_dict(p) for p in self.phases],
            },
            "rule_triggers": asdict(self.rule_triggers),
            "visualization_file": self.visualization_file,
        }

    def _phase_to_dict(self, phase: PerPhaseMetrics) -> dict[str, Any]:
        """将单阶段指标转换为字典"""
        result = {
            "phase_name": phase.phase_name,
            "phase_name_cn": phase.phase_name_cn,
            "timing": asdict(phase.timing),
            "imu": asdict(phase.imu) if phase.imu else None,
            "vision": asdict(phase.vision) if phase.vision else None,
            "emg": asdict(phase.emg) if phase.emg else None,
        }

        # 添加阶段特定指标
        phase_specific = None
        for attr in [
            "address",
            "takeaway",
            "backswing",
            "top",
            "downswing",
            "impact",
            "follow_through",
            "finish",
        ]:
            specific = getattr(phase, attr, None)
            if specific is not None:
                phase_specific = asdict(specific)
                break

        result["phase_specific"] = phase_specific
        return result

    def save(self, output_path: str) -> None:
        """保存为 JSON 文件"""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"✅ Kinematic Prompt V2 已保存: {output_path}")


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
