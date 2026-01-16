# Movement Chain - 传感器融合 Pipeline 实现计划

> 状态: 待审核
> 日期: 2025-01-15
> 作者: Claude Code

---

## 1. 目标概述

构建完整的 **视频 + IMU → 融合 → Rerun 可视化 → AI 反馈** 流水线。

```
┌─────────────┐    ┌─────────────┐
│   视频文件   │    │  IMU CSV    │
│  (.mp4)     │    │   数据      │
└──────┬──────┘    └──────┬──────┘
       │                  │
       ▼                  ▼
┌─────────────┐    ┌─────────────┐
│  Vision     │    │    IMU      │
│  Analyzer   │    │  Analyzer   │
│ (MediaPipe) │    │  (已有)     │
└──────┬──────┘    └──────┬──────┘
       │                  │
       └────────┬─────────┘
                ▼
        ┌───────────────┐
        │ Sensor Fusion │
        │ (时间对齐)     │
        └───────┬───────┘
                │
       ┌────────┴────────┐
       ▼                 ▼
┌─────────────┐   ┌─────────────┐
│   Rerun     │   │  AI Coach   │
│ Visualizer  │   │ (Kinematic  │
│  (3D骨架)   │   │  Prompts)   │
└─────────────┘   └─────────────┘
```

---

## 2. 文件结构 (新增/修改)

```
analysis/
├── imu_swing_analyzer.py     # [已有] IMU 分析器
├── test_mediapipe.py         # [已有] MediaPipe 测试脚本
├── vision_analyzer.py        # [新增] 完整视频分析器
├── sensor_fusion.py          # [新增] 传感器融合 + 时间对齐
├── rerun_visualizer.py       # [新增] Rerun 3D 可视化
├── ai_coach.py               # [新增] AI 反馈生成器
├── pipeline.py               # [新增] 主 Pipeline 入口
└── schemas.py                # [新增] 统一数据结构定义
```

---

## 3. 核心设计决策

### 3.1 时间对齐策略: Impact-Based Synchronization

**问题**: 视频 (30fps) 和 IMU (1000Hz+) 时钟不同步

**解决方案**: 使用 **Impact 时刻** 作为对齐锚点

| 传感器 | Impact 检测方法 |
|--------|----------------|
| IMU | `np.argmax(gyro_magnitude)` - 陀螺仪合成角速度峰值 |
| Vision | 手腕关键点速度峰值 或 用户手动标记帧 |

```python
# 对齐算法伪代码
imu_impact_time = imu_phases["impact"].start_time_ms
vision_impact_frame = detect_impact_frame(vision_data)
vision_impact_time = vision_impact_frame * (1000 / fps)

time_offset = imu_impact_time - vision_impact_time
# 所有 Vision 时间戳 += time_offset
```

### 3.2 统一数据结构

```python
@dataclass
class FusedFrame:
    """单帧融合数据"""
    timestamp_ms: float              # 统一时间戳 (IMU 时钟)

    # Vision 数据 (插值到 IMU 采样率)
    pose_landmarks: List[float]      # 33 关键点 [x, y, z, visibility]
    left_arm_angle: Optional[float]
    right_arm_angle: Optional[float]
    x_factor: Optional[float]

    # IMU 数据
    gyro_dps: Tuple[float, float, float]  # GyX, GyY, GyZ
    accel_g: Tuple[float, float, float]   # AcX, AcY, AcZ
    gyro_magnitude: float

@dataclass
class FusedSwingData:
    """完整挥杆融合数据"""
    frames: List[FusedFrame]

    # IMU 阶段 & 指标 (来自 imu_swing_analyzer)
    imu_phases: List[SwingPhase]
    imu_metrics: SwingMetrics

    # Vision 指标
    vision_metrics: VisionMetrics

    # 元数据
    video_file: str
    imu_file: str
    alignment_offset_ms: float
```

### 3.3 Vision 指标定义 (6 项)

基于文档 `metrics-calculation.md` 定义:

| 指标 | 计算方法 | 说明 |
|------|----------|------|
| `address_posture_score` | 脊柱角度 + 膝盖弯曲度评分 | 0-100 |
| `spine_angle_deg` | 肩膀中点到髋部中点的倾斜角 | 度数 |
| `head_movement_cm` | 鼻尖位置在挥杆过程中的移动距离 | 厘米 |
| `x_factor_max_deg` | 肩线与髋线最大夹角 | 度数 |
| `lead_arm_extension_pct` | 前臂在顶点时的伸展百分比 | 0-100% |
| `hip_rotation_deg` | 髋部旋转角度 | 度数 |

### 3.4 Rerun 可视化内容

1. **3D 骨架动画** - 33 关键点 + 连接线
2. **时间序列图** - 左臂角度、右臂角度、X-Factor、陀螺仪
3. **阶段标注** - 8 阶段颜色区分 (Address, Takeaway, Backswing, Top, Downswing, Impact, Follow-through, Finish)
4. **同步播放** - 原视频与骨架 overlay

### 3.5 AI Coach 输出格式 (Kinematic Prompts)

```json
{
  "session_id": "2025-01-15_swing_001",
  "summary": {
    "overall_level": "中级",
    "key_issues": ["上杆过快", "头部移动过大"]
  },
  "metrics": {
    "imu": {
      "peak_angular_velocity_dps": 1200,
      "tempo_ratio": 2.8,
      "...": "..."
    },
    "vision": {
      "x_factor_max_deg": 45,
      "head_movement_cm": 8.5,
      "...": "..."
    }
  },
  "phase_analysis": [
    {
      "phase": "backswing",
      "duration_ms": 650,
      "issues": ["速度过快"],
      "benchmark_comparison": "比职业选手快 15%"
    }
  ],
  "ai_prompt": "基于以下高尔夫挥杆分析数据，提供具体的改进建议...",
  "visualization_url": "file://output/swing_001.rrd"
}
```

---

## 4. 实现细节

### 4.1 `schemas.py` - 统一数据结构

```python
# 核心 dataclass 定义
- PoseFrame          # 单帧姿态数据 (来自 MediaPipe)
- VisionMetrics      # 视觉指标汇总
- FusedFrame         # 单帧融合数据
- FusedSwingData     # 完整挥杆融合数据
- KinematicPrompt    # AI 输入结构
```

### 4.2 `vision_analyzer.py` - 视频分析器

基于 `test_mediapipe.py` 扩展:

```python
class VisionAnalyzer:
    def __init__(self, model_path: Path):
        # 初始化 MediaPipe PoseLandmarker

    def analyze_video(self, video_path: str) -> VisionResult:
        # 逐帧分析，返回所有帧的姿态数据

    def detect_impact_frame(self, frames: List[PoseFrame]) -> int:
        # 检测 Impact 帧 (手腕速度峰值)

    def calculate_vision_metrics(self, frames: List[PoseFrame], phases: List) -> VisionMetrics:
        # 计算 6 项视觉指标
```

### 4.3 `sensor_fusion.py` - 传感器融合

```python
class SensorFusion:
    def __init__(self):
        pass

    def align_timestamps(
        self,
        vision_frames: List[PoseFrame],
        imu_df: pd.DataFrame,
        vision_fps: float,
        imu_impact_idx: int,
        vision_impact_frame: int
    ) -> float:
        # 计算时间偏移量

    def interpolate_vision_to_imu(
        self,
        vision_frames: List[PoseFrame],
        imu_timestamps: np.ndarray,
        time_offset: float,
        vision_fps: float
    ) -> List[Optional[PoseFrame]]:
        # 将 Vision 数据插值到 IMU 时间戳

    def fuse(
        self,
        vision_result: VisionResult,
        imu_result: Tuple[pd.DataFrame, List[SwingPhase], SwingMetrics, Dict]
    ) -> FusedSwingData:
        # 主融合函数
```

### 4.4 `rerun_visualizer.py` - Rerun 可视化

```python
class RerunVisualizer:
    def __init__(self, output_path: str = "output/swing.rrd"):
        rr.init("movement-chain-swing")
        rr.save(output_path)

    def log_frame(self, fused_frame: FusedFrame, frame_idx: int):
        # 记录单帧数据到 Rerun
        # - 3D 骨架点
        # - 连接线
        # - 时间序列值

    def log_phases(self, phases: List[SwingPhase]):
        # 记录阶段区间标注

    def visualize_swing(self, fused_data: FusedSwingData):
        # 完整可视化流程
```

### 4.5 `ai_coach.py` - AI 反馈生成

```python
class AICoach:
    def __init__(self):
        self.benchmarks = load_benchmarks()  # 职业选手基准数据

    def analyze_issues(self, fused_data: FusedSwingData) -> List[Issue]:
        # 与基准对比，识别问题

    def generate_kinematic_prompt(self, fused_data: FusedSwingData) -> KinematicPrompt:
        # 生成结构化 AI 输入

    def generate_feedback(self, prompt: KinematicPrompt) -> str:
        # 调用 LLM 生成反馈文本 (可选)
```

### 4.6 `pipeline.py` - 主入口

```python
def run_pipeline(
    video_path: str,
    imu_path: str,
    output_dir: str = "output",
    show_rerun: bool = True,
    generate_ai_feedback: bool = True
) -> KinematicPrompt:
    """
    完整 Pipeline:
    1. 分析视频 (VisionAnalyzer)
    2. 分析 IMU (imu_swing_analyzer.analyze_swing)
    3. 融合数据 (SensorFusion)
    4. 可视化 (RerunVisualizer)
    5. 生成 AI 反馈 (AICoach)
    """
```

---

## 5. 假设 & 待确认

### 5.1 数据格式假设

| 假设 | 说明 | 如果不同... |
|------|------|-------------|
| 视频格式 | `.mp4`, `.mov`, `.avi` | 请确认实际格式 |
| 视频帧率 | 30fps | 请确认实际帧率 |
| IMU 格式 | CSV, 与 `imu_swing_analyzer.py` 兼容 | 请确认 |
| 同步录制 | 视频和 IMU 大致同时开始录制 | 如有大偏移需手动标记 |

### 5.2 待确认问题

1. **视频中是否只有一人?** (假设: 是, 只检测一个人)

2. **Impact 帧检测方法**:
   - 自动检测 (手腕速度峰值)
   - 还是需要用户手动标记?

3. **AI 反馈生成**:
   - 本地 LLM (Ollama)?
   - Claude API?
   - 还是只生成结构化 Prompt, 由用户手动调用?

4. **Rerun 输出**:
   - 保存 `.rrd` 文件?
   - 还是直接启动 Rerun Viewer?
   - 或两者都支持?

---

## 6. 实现顺序

```
Phase 1: schemas.py           # 定义所有数据结构
    ↓
Phase 2: vision_analyzer.py   # 完整视频分析
    ↓
Phase 3: sensor_fusion.py     # 时间对齐 + 融合
    ↓
Phase 4: rerun_visualizer.py  # Rerun 可视化
    ↓
Phase 5: ai_coach.py          # AI 反馈生成
    ↓
Phase 6: pipeline.py          # 主入口整合
    ↓
Phase 7: 测试 & 调试          # 用真实数据验证
```

---

## 7. 预计代码量

| 文件 | 预计行数 | 复杂度 |
|------|----------|--------|
| `schemas.py` | ~150 | 低 |
| `vision_analyzer.py` | ~350 | 中 |
| `sensor_fusion.py` | ~250 | 高 |
| `rerun_visualizer.py` | ~300 | 中 |
| `ai_coach.py` | ~200 | 低 |
| `pipeline.py` | ~150 | 低 |
| **总计** | **~1400** | |

---

## 8. 下一步

**请审核以上计划:**

1. 架构设计是否符合预期?
2. 假设是否正确?
3. 5.2 中的待确认问题如何处理?

确认后我将开始实现。
