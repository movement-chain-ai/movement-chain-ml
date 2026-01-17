"""
Movement Chain - Vision Analyzer (视频分析器)

基于 MediaPipe Pose Landmarker 的视频分析模块。
支持 Python 3.14 + MediaPipe 0.10.x (仅 Tasks API)

用法:
    from vision_analyzer import VisionAnalyzer

    analyzer = VisionAnalyzer()
    result = analyzer.analyze_video("path/to/video.mp4")
    print(f"检测到 {len(result.frames)} 帧, Impact 帧: {result.impact_frame_idx}")

Author: Movement Chain AI Team
Date: 2025-01-15
"""

import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from .schemas import (
    PoseFrame,
    VisionMetrics,
    VisionResult,
)

# 模型路径配置 (relative to project root: project_root/models/)
# __file__ = src/movement_chain/vision_analyzer.py
# .parent.parent.parent = project root
MODELS_DIR = Path(__file__).parent.parent.parent / "models"

# 可用模型: heavy (最准确), full (平衡), lite (最快)
MODEL_FILES = {
    "heavy": "pose_landmarker_heavy.task",
    "full": "pose_landmarker_full.task",
    "lite": "pose_landmarker_lite.task",
}

DEFAULT_MODEL_TYPE = "heavy"  # Heavy 作为默认，准确度最高


def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    计算三点夹角 (p2 为顶点)

    Args:
        p1, p2, p3: 三个点的坐标 (numpy array)

    Returns:
        夹角 (度数)
    """
    v1 = p1 - p2
    v2 = p3 - p2
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1, 1))))


def calculate_vector_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    计算两个向量之间的夹角

    Args:
        v1, v2: 两个向量

    Returns:
        夹角 (度数)
    """
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1, 1))))


class VisionAnalyzer:
    """视频姿态分析器"""

    def __init__(
        self,
        model_type: str = DEFAULT_MODEL_TYPE,
        model_path: Path | None = None,
    ):
        """
        初始化分析器

        Args:
            model_type: 模型类型 ("heavy", "full", "lite")
                - heavy: 最高精度，速度较慢 (默认)
                - full: 平衡精度和速度
                - lite: 最快速度，精度较低
            model_path: 自定义模型文件路径 (覆盖 model_type)
        """
        self.model_type = model_type

        if model_path:
            self.model_path = model_path
        else:
            if model_type not in MODEL_FILES:
                raise ValueError(
                    f"未知模型类型: {model_type}，可选: {list(MODEL_FILES.keys())}"
                )
            self.model_path = MODELS_DIR / MODEL_FILES[model_type]

        if not self.model_path.exists():
            # 生成下载命令
            model_name = MODEL_FILES.get(model_type, "pose_landmarker_heavy.task")
            base_name = model_name.replace(".task", "")
            raise FileNotFoundError(
                f"模型文件不存在: {self.model_path}\n"
                f"请运行以下命令下载模型:\n"
                f'  curl -L -o models/{model_name} '
                f'"https://storage.googleapis.com/mediapipe-models/pose_landmarker/'
                f'{base_name}/float16/latest/{model_name}"'
            )

    def analyze_video(
        self,
        video_path: str,
        show_progress: bool = True,
    ) -> VisionResult:
        """
        分析视频，提取所有帧的姿态数据

        Args:
            video_path: 视频文件路径
            show_progress: 是否显示进度

        Returns:
            VisionResult 包含所有帧数据和汇总指标
        """
        video_path_obj = Path(video_path)
        if not video_path_obj.exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        # 打开视频
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if show_progress:
            print(f"[VisionAnalyzer] 视频: {video_path_obj.name}")
            print(
                f"[VisionAnalyzer] 尺寸: {width}x{height}, FPS: {fps:.1f}, 总帧数: {total_frames}"
            )

        # 初始化 PoseLandmarker
        base_options = python.BaseOptions(model_asset_path=str(self.model_path))
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        frames: list[PoseFrame] = []
        prev_wrist_pos: tuple[float, float] | None = None

        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # 转换为 MediaPipe Image
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                # 计算时间戳
                timestamp_ms = frame_idx * (1000.0 / fps)

                # 姿态检测
                try:
                    detection_result = landmarker.detect_for_video(mp_image, int(timestamp_ms))
                except Exception as e:
                    if show_progress and frame_idx % 100 == 0:
                        print(f"[WARN] 帧 {frame_idx} 检测失败: {e}")
                    frame_idx += 1
                    continue

                if detection_result.pose_landmarks:
                    landmarks = detection_result.pose_landmarks[0]

                    # 转换为列表格式
                    landmarks_list = [
                        [lm.x, lm.y, lm.z, getattr(lm, "visibility", 1.0)] for lm in landmarks
                    ]

                    # 计算角度指标
                    metrics = self._extract_frame_metrics(landmarks)

                    # 计算手腕速度 (用于 Impact 检测)
                    wrist_speed = None
                    right_wrist = landmarks[16]
                    current_wrist_pos = (right_wrist.x * width, right_wrist.y * height)
                    if prev_wrist_pos is not None:
                        dx = current_wrist_pos[0] - prev_wrist_pos[0]
                        dy = current_wrist_pos[1] - prev_wrist_pos[1]
                        wrist_speed = np.sqrt(dx * dx + dy * dy)
                    prev_wrist_pos = current_wrist_pos

                    pose_frame = PoseFrame(
                        frame_idx=frame_idx,
                        timestamp_ms=timestamp_ms,
                        landmarks=landmarks_list,
                        left_arm_angle=metrics.get("left_arm_angle"),
                        right_arm_angle=metrics.get("right_arm_angle"),
                        x_factor=metrics.get("x_factor"),
                        wrist_speed=wrist_speed,
                    )
                    frames.append(pose_frame)

                # 进度显示
                if show_progress and frame_idx % 100 == 0:
                    print(
                        f"[VisionAnalyzer] 处理进度: {frame_idx}/{total_frames} ({100*frame_idx/total_frames:.1f}%)"
                    )

                frame_idx += 1

        cap.release()

        if show_progress:
            print(f"[VisionAnalyzer] 完成! 检测到 {len(frames)} 帧姿态数据")

        # 检测 Impact 帧
        impact_frame_idx, impact_confidence = self._detect_impact_frame(frames)

        # 计算汇总指标
        vision_metrics = self._calculate_vision_metrics(frames, impact_frame_idx)

        return VisionResult(
            video_file=str(video_path),
            fps=fps,
            total_frames=total_frames,
            width=width,
            height=height,
            frames=frames,
            metrics=vision_metrics,
            impact_frame_idx=impact_frame_idx,
            impact_confidence=impact_confidence,
        )

    def _extract_frame_metrics(self, landmarks) -> dict:
        """从单帧关键点提取角度指标"""
        metrics = {}

        def get_point(idx):
            lm = landmarks[idx]
            return np.array([lm.x, lm.y, lm.z])

        # 左臂角度 (肩11-肘13-腕15)
        try:
            left_arm = calculate_angle(get_point(11), get_point(13), get_point(15))
            metrics["left_arm_angle"] = round(left_arm, 1)
        except Exception:
            metrics["left_arm_angle"] = None

        # 右臂角度 (肩12-肘14-腕16)
        try:
            right_arm = calculate_angle(get_point(12), get_point(14), get_point(16))
            metrics["right_arm_angle"] = round(right_arm, 1)
        except Exception:
            metrics["right_arm_angle"] = None

        # X-Factor (肩线与髋线夹角)
        try:
            shoulder_vec = np.array(
                [landmarks[12].x - landmarks[11].x, landmarks[12].y - landmarks[11].y]
            )
            hip_vec = np.array(
                [landmarks[24].x - landmarks[23].x, landmarks[24].y - landmarks[23].y]
            )
            x_factor = calculate_vector_angle(shoulder_vec, hip_vec)
            metrics["x_factor"] = round(x_factor, 1)
        except Exception:
            metrics["x_factor"] = None

        return metrics

    def _detect_impact_frame(self, frames: list[PoseFrame]) -> tuple[int | None, float | None]:
        """
        检测 Impact 帧 (基于手腕速度峰值)

        Returns:
            (impact_frame_idx, confidence)
        """
        if len(frames) < 10:
            return None, None

        # 提取手腕速度序列
        speeds = []
        for f in frames:
            speeds.append(f.wrist_speed if f.wrist_speed is not None else 0)

        speeds = np.array(speeds)

        # 平滑处理
        if len(speeds) > 5:
            kernel = np.ones(5) / 5
            speeds_smooth = np.convolve(speeds, kernel, mode="same")
        else:
            speeds_smooth = speeds

        # 找到速度峰值
        peak_idx = int(np.argmax(speeds_smooth))
        peak_value = speeds_smooth[peak_idx]

        # 计算置信度 (峰值与平均值的比率)
        mean_speed = np.mean(speeds_smooth)
        confidence = min(1.0, (peak_value / (mean_speed + 1e-6)) / 5.0) if mean_speed > 0 else 0

        # 返回对应的帧索引
        if peak_idx < len(frames):
            return frames[peak_idx].frame_idx, round(confidence, 2)

        return None, None

    def _calculate_vision_metrics(
        self, frames: list[PoseFrame], impact_frame_idx: int | None
    ) -> VisionMetrics:
        """计算视觉汇总指标"""

        if not frames:
            return VisionMetrics()

        # 收集所有帧的 X-Factor
        x_factors = [f.x_factor for f in frames if f.x_factor is not None]
        x_factor_max = max(x_factors) if x_factors else None

        # 找到 Top 位置 (X-Factor 最大的帧)
        top_frame_idx = None
        if x_factors:
            max_xf_idx = np.argmax(x_factors)
            # 找到对应的原始帧索引
            xf_frames = [f for f in frames if f.x_factor is not None]
            if max_xf_idx < len(xf_frames):
                top_frame_idx = xf_frames[max_xf_idx].frame_idx

        # 计算头部移动距离
        head_movement = self._calculate_head_movement(frames)

        # 计算前臂伸展度 (在 Top 位置)
        lead_arm_extension = None
        if top_frame_idx is not None:
            top_frames = [f for f in frames if f.frame_idx == top_frame_idx]
            if top_frames:
                # 假设右手打者，前臂是左臂
                left_arm = top_frames[0].left_arm_angle
                if left_arm is not None:
                    # 完全伸直是 180 度
                    lead_arm_extension = round((left_arm / 180.0) * 100, 1)

        # 计算脊柱角度 (准备姿势)
        spine_angle = self._calculate_spine_angle(frames[0] if frames else None)

        # 计算髋部旋转
        hip_rotation = self._calculate_hip_rotation(frames, impact_frame_idx)

        # 计算准备姿势评分
        address_score = self._calculate_address_score(frames[0] if frames else None)

        return VisionMetrics(
            address_posture_score=address_score,
            spine_angle_deg=spine_angle,
            head_movement_cm=head_movement,
            x_factor_max_deg=x_factor_max,
            hip_rotation_deg=hip_rotation,
            lead_arm_extension_pct=lead_arm_extension,
            top_position_frame=top_frame_idx,
            impact_frame=impact_frame_idx,
        )

    def _calculate_head_movement(self, frames: list[PoseFrame]) -> float | None:
        """计算头部移动距离 (像素转厘米，假设肩宽 = 40cm)"""
        if len(frames) < 2:
            return None

        # 获取第一帧的参考数据
        first_frame = frames[0]
        if not first_frame.landmarks or len(first_frame.landmarks) < 1:
            return None

        # 鼻尖位置 (索引 0)
        nose_positions = []
        shoulder_widths = []

        for f in frames:
            if f.landmarks and len(f.landmarks) > 24:
                nose_positions.append((f.landmarks[0][0], f.landmarks[0][1]))
                # 肩宽用于像素到厘米转换
                left_shoulder = f.landmarks[11]
                right_shoulder = f.landmarks[12]
                sw = np.sqrt(
                    (right_shoulder[0] - left_shoulder[0]) ** 2
                    + (right_shoulder[1] - left_shoulder[1]) ** 2
                )
                shoulder_widths.append(sw)

        if len(nose_positions) < 2:
            return None

        # 计算最大移动距离
        nose_positions = np.array(nose_positions)
        start_pos = nose_positions[0]
        distances = np.sqrt(
            (nose_positions[:, 0] - start_pos[0]) ** 2 + (nose_positions[:, 1] - start_pos[1]) ** 2
        )
        max_distance_pixels = float(np.max(distances))

        # 转换为厘米 (假设肩宽 40cm)
        avg_shoulder_width = np.mean(shoulder_widths) if shoulder_widths else 0.1
        cm_per_pixel = 40.0 / avg_shoulder_width if avg_shoulder_width > 0 else 1.0
        head_movement_cm = max_distance_pixels * cm_per_pixel

        return round(head_movement_cm, 1)

    def _calculate_spine_angle(self, frame: PoseFrame | None) -> float | None:
        """计算脊柱倾斜角度"""
        if frame is None or not frame.landmarks or len(frame.landmarks) < 25:
            return None

        try:
            # 肩膀中点
            left_shoulder = np.array(frame.landmarks[11][:2])
            right_shoulder = np.array(frame.landmarks[12][:2])
            shoulder_mid = (left_shoulder + right_shoulder) / 2

            # 髋部中点
            left_hip = np.array(frame.landmarks[23][:2])
            right_hip = np.array(frame.landmarks[24][:2])
            hip_mid = (left_hip + right_hip) / 2

            # 脊柱向量
            spine_vec = shoulder_mid - hip_mid

            # 与垂直方向的夹角 (假设 y 轴向下)
            vertical = np.array([0, -1])
            angle = calculate_vector_angle(spine_vec, vertical)

            return round(angle, 1)
        except Exception:
            return None

    def _calculate_hip_rotation(
        self, frames: list[PoseFrame], impact_frame_idx: int | None
    ) -> float | None:
        """计算髋部旋转角度 (从准备到 Impact)"""
        if len(frames) < 2 or impact_frame_idx is None:
            return None

        # 找到准备帧和 Impact 帧
        address_frame = frames[0]
        impact_frames = [f for f in frames if f.frame_idx == impact_frame_idx]

        if not impact_frames or not address_frame.landmarks:
            return None

        impact_frame = impact_frames[0]
        if not impact_frame.landmarks:
            return None

        try:
            # 准备姿势的髋线方向
            addr_hip_vec = np.array(
                [
                    address_frame.landmarks[24][0] - address_frame.landmarks[23][0],
                    address_frame.landmarks[24][1] - address_frame.landmarks[23][1],
                ]
            )

            # Impact 时的髋线方向
            impact_hip_vec = np.array(
                [
                    impact_frame.landmarks[24][0] - impact_frame.landmarks[23][0],
                    impact_frame.landmarks[24][1] - impact_frame.landmarks[23][1],
                ]
            )

            rotation = calculate_vector_angle(addr_hip_vec, impact_hip_vec)
            return round(rotation, 1)
        except Exception:
            return None

    def _calculate_address_score(self, frame: PoseFrame | None) -> float | None:
        """计算准备姿势评分 (0-100)"""
        if frame is None or not frame.landmarks:
            return None

        score = 100.0
        penalties = []

        try:
            # 脊柱角度评估 (理想: 30-45 度)
            spine_angle = self._calculate_spine_angle(frame)
            if spine_angle is not None:
                if spine_angle < 20 or spine_angle > 60:
                    penalties.append(20)
                elif spine_angle < 30 or spine_angle > 45:
                    penalties.append(10)

            # 膝盖弯曲评估
            # 左腿: 髋23-膝25-踝27
            left_knee_angle = None
            right_knee_angle = None

            if len(frame.landmarks) > 28:
                try:
                    left_knee_angle = calculate_angle(
                        np.array(frame.landmarks[23][:3]),
                        np.array(frame.landmarks[25][:3]),
                        np.array(frame.landmarks[27][:3]),
                    )
                    right_knee_angle = calculate_angle(
                        np.array(frame.landmarks[24][:3]),
                        np.array(frame.landmarks[26][:3]),
                        np.array(frame.landmarks[28][:3]),
                    )
                except Exception:
                    pass

            # 理想膝盖角度: 160-175 度 (微弯)
            for knee_angle in [left_knee_angle, right_knee_angle]:
                if knee_angle is not None:
                    if knee_angle < 140 or knee_angle > 180:
                        penalties.append(15)
                    elif knee_angle < 155 or knee_angle > 175:
                        penalties.append(5)

            score -= sum(penalties)
            return max(0, round(score, 1))

        except Exception:
            return None


def main():
    """命令行测试入口"""
    print("=" * 60)
    print("  Movement Chain - Vision Analyzer")
    print(f"  Python {sys.version_info.major}.{sys.version_info.minor}")
    print(f"  MediaPipe {mp.__version__}")
    print("=" * 60)

    if len(sys.argv) < 2:
        print("\n用法: python vision_analyzer.py <video_path>")
        print("示例: python vision_analyzer.py data/swing.mp4")
        return

    video_path = sys.argv[1]

    try:
        analyzer = VisionAnalyzer()
        result = analyzer.analyze_video(video_path)

        print("\n" + "=" * 60)
        print("分析结果")
        print("=" * 60)
        print(f"视频: {result.video_file}")
        print(f"帧数: {len(result.frames)}/{result.total_frames}")
        print(f"FPS: {result.fps:.1f}")
        print()

        print("Impact 检测:")
        print(f"  帧索引: {result.impact_frame_idx}")
        print(f"  置信度: {result.impact_confidence}")
        print()

        print("视觉指标:")
        m = result.metrics
        print(f"  准备姿势评分: {m.address_posture_score}")
        print(f"  脊柱角度: {m.spine_angle_deg}°")
        print(f"  头部移动: {m.head_movement_cm} cm")
        print(f"  X-Factor 最大值: {m.x_factor_max_deg}°")
        print(f"  髋部旋转: {m.hip_rotation_deg}°")
        print(f"  前臂伸展度: {m.lead_arm_extension_pct}%")
        print("=" * 60)

    except Exception as e:
        print(f"\n[ERROR] 分析失败: {e}")
        raise


if __name__ == "__main__":
    main()
