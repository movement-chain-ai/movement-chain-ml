"""
Movement Chain - Rerun Visualizer (3D 可视化)

使用 Rerun SDK 可视化融合后的挥杆数据。
支持 3D 骨架动画、时间序列图、阶段标注。

用法:
    from rerun_visualizer import RerunVisualizer

    visualizer = RerunVisualizer()
    visualizer.visualize_swing(fused_data, output_path="output/swing.rrd")

    # 直接打开 Viewer
    visualizer.visualize_swing(fused_data, spawn_viewer=True)

Author: Movement Chain AI Team
Date: 2025-01-15
"""

from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import rerun as rr

    RERUN_AVAILABLE = True
except ImportError:
    RERUN_AVAILABLE = False
    print("[WARN] rerun-sdk 未安装，可视化功能不可用")
    print("       运行: pip install rerun-sdk>=0.20.0")

from .schemas import (
    PHASE_COLORS,
    POSE_CONNECTIONS,
    FusedFrame,
    FusedSwingData,
)


class RerunVisualizer:
    """Rerun 可视化器"""

    def __init__(self, app_name: str = "Movement Chain Swing Analyzer"):
        """
        初始化可视化器

        Args:
            app_name: Rerun 应用名称
        """
        self.app_name = app_name

        if not RERUN_AVAILABLE:
            raise ImportError("rerun-sdk 未安装。请运行: pip install rerun-sdk>=0.20.0")

    def visualize_swing(
        self,
        fused_data: FusedSwingData,
        output_path: str | None = None,
        spawn_viewer: bool = False,
        downsample_factor: int = 10,
        video_path: str | None = None,
    ) -> str | None:
        """
        可视化融合后的挥杆数据

        Args:
            fused_data: 融合后的挥杆数据
            output_path: 输出 .rrd 文件路径 (None = 不保存)
            spawn_viewer: 是否启动 Rerun Viewer
            downsample_factor: 降采样因子 (IMU 数据量大，需要降采样)
            video_path: 原始视频路径 (用于显示视频帧)

        Returns:
            保存的文件路径 (如果有)
        """
        print(f"[RerunVisualizer] 开始可视化 ({len(fused_data.frames)} 帧)...")

        # 初始化 Rerun
        rr.init(self.app_name, spawn=spawn_viewer)

        if output_path:
            output_path_obj = Path(output_path)
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)
            rr.save(str(output_path_obj))
            print(f"[RerunVisualizer] 输出文件: {output_path}")

        # 记录元数据
        self._log_metadata(fused_data)

        # 记录阶段信息
        self._log_phases(fused_data.imu_phases)

        # 预加载视频帧 (如果提供)
        video_frames: dict[int, np.ndarray] = {}
        if video_path and Path(video_path).exists():
            video_frames = self._preload_video_frames(video_path, downsample_factor)
            print(f"[RerunVisualizer] 预加载 {len(video_frames)} 个视频帧")

        # 记录帧数据
        frames_logged = 0
        for i, frame in enumerate(fused_data.frames):
            # 降采样
            if i % downsample_factor != 0:
                continue

            # 设置时间 (在记录任何数据之前)
            rr.set_time("time", duration=frame.timestamp_ms / 1000.0)
            rr.set_time("frame", sequence=frame.frame_idx)

            # 记录视频帧 (如果有)
            if video_frames and frame.has_vision:
                # 计算对应的视频帧索引
                video_time_ms = frame.timestamp_ms - fused_data.alignment_offset_ms
                video_frame_idx = int(video_time_ms * fused_data.vision_fps / 1000.0)

                video_frame = self._find_closest_frame(video_frames, video_frame_idx)
                if video_frame is not None:
                    rr.log("video", rr.Image(video_frame))

            self._log_frame(frame, fused_data.vision_fps)
            frames_logged += 1

            # 进度显示
            if frames_logged % 50 == 0:
                print(f"[RerunVisualizer] 进度: {frames_logged} 帧")

        print(f"[RerunVisualizer] 完成! 共记录 {frames_logged} 帧")

        if output_path:
            print(f"[RerunVisualizer] 文件保存: {output_path}")
            print(f"[RerunVisualizer] 运行 'rerun {output_path}' 打开可视化")

        return output_path

    def _preload_video_frames(
        self, video_path: str, downsample_factor: int
    ) -> dict[int, np.ndarray]:
        """预加载视频帧到内存"""
        frames: dict[int, np.ndarray] = {}
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"[RerunVisualizer] 警告: 无法打开视频 {video_path}")
            return frames

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 保留所有帧 (内存允许的情况下)
            # 对于 252 帧的 1080p 视频约需 1.5GB 内存
            # 如果内存不足，可以改为每隔几帧保留
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames[frame_idx] = frame_rgb
            frame_idx += 1

        cap.release()
        return frames

    def _find_closest_frame(
        self, frames: dict[int, np.ndarray], target_idx: int
    ) -> np.ndarray | None:
        """找到最接近目标索引的预加载帧"""
        if not frames:
            return None
        if target_idx in frames:
            return frames[target_idx]
        # 找最近的帧
        closest = min(frames.keys(), key=lambda x: abs(x - target_idx))
        return frames[closest]

    def _log_metadata(self, fused_data: FusedSwingData) -> None:
        """记录元数据"""
        rr.log(
            "metadata",
            rr.TextDocument(
                f"""# Movement Chain Swing Analysis

**Session ID**: {fused_data.session_id}
**Video**: {fused_data.video_file}
**IMU**: {fused_data.imu_file}
**Analysis Time**: {fused_data.analysis_time}
**Time Offset**: {fused_data.alignment_offset_ms:.1f} ms

## Vision Metrics
- X-Factor Max: {fused_data.vision_metrics.x_factor_max_deg}°
- Head Movement: {fused_data.vision_metrics.head_movement_cm} cm
- Lead Arm Extension: {fused_data.vision_metrics.lead_arm_extension_pct}%

## IMU Metrics
- Peak Velocity: {fused_data.imu_metrics.get('peak_angular_velocity_dps', 'N/A')} °/s
- Tempo Ratio: {fused_data.imu_metrics.get('tempo_ratio', 'N/A')}
- Total Swing Time: {fused_data.imu_metrics.get('total_swing_time_ms', 'N/A')} ms
""",
                media_type=rr.MediaType.MARKDOWN,
            ),
            static=True,
        )

    def _log_phases(self, imu_phases: list[dict[str, Any]]) -> None:
        """记录阶段信息到 Rerun"""
        for phase in imu_phases:
            phase_name = phase.get("name", "unknown")
            phase_cn = phase.get("name_cn", phase_name)
            start_ms = phase.get("start_time_ms", 0)
            end_ms = phase.get("end_time_ms", 0)
            duration = phase.get("duration_ms", 0)

            # 记录阶段标记
            rr.log(
                f"phases/{phase_name}",
                rr.TextDocument(
                    f"**{phase_cn}** ({phase_name})\n"
                    f"Time: {start_ms:.0f} - {end_ms:.0f} ms\n"
                    f"Duration: {duration:.0f} ms",
                    media_type=rr.MediaType.MARKDOWN,
                ),
                static=True,
            )

    def _log_frame(self, frame: FusedFrame, fps: float) -> None:
        """记录单帧数据"""
        # 设置时间 (rerun 0.28+ API)
        rr.set_time("time", duration=frame.timestamp_ms / 1000.0)
        rr.set_time("frame", sequence=frame.frame_idx)

        # 获取阶段颜色
        phase_color = (0.5, 0.5, 0.5)  # 默认灰色
        if frame.phase_name:
            phase_color = PHASE_COLORS.get(frame.phase_name.lower(), (0.5, 0.5, 0.5))

        # 记录 3D 骨架 (如果有 Vision 数据)
        if frame.has_vision and frame.pose_landmarks:
            self._log_skeleton(frame.pose_landmarks, phase_color)

        # 记录 IMU 时间序列
        if frame.gyro_dps:
            rr.log("sensors/gyro/x", rr.Scalars(frame.gyro_dps[0]))
            rr.log("sensors/gyro/y", rr.Scalars(frame.gyro_dps[1]))
            rr.log("sensors/gyro/z", rr.Scalars(frame.gyro_dps[2]))

        if frame.gyro_magnitude:
            rr.log("sensors/gyro_magnitude", rr.Scalars(frame.gyro_magnitude))

        if frame.accel_g:
            rr.log("sensors/accel/x", rr.Scalars(frame.accel_g[0]))
            rr.log("sensors/accel/y", rr.Scalars(frame.accel_g[1]))
            rr.log("sensors/accel/z", rr.Scalars(frame.accel_g[2]))

        # 记录 Vision 指标
        if frame.left_arm_angle:
            rr.log("metrics/left_arm_angle", rr.Scalars(frame.left_arm_angle))
        if frame.right_arm_angle:
            rr.log("metrics/right_arm_angle", rr.Scalars(frame.right_arm_angle))
        if frame.x_factor:
            rr.log("metrics/x_factor", rr.Scalars(frame.x_factor))

        # 记录阶段名称
        if frame.phase_name:
            rr.log(
                "phase/current",
                rr.TextDocument(
                    f"**{frame.phase_name_cn or frame.phase_name}**",
                    media_type=rr.MediaType.MARKDOWN,
                ),
            )

    def _log_skeleton(
        self, landmarks: list[list[float]], color: tuple[float, float, float]
    ) -> None:
        """记录 3D 骨架"""
        if not landmarks or len(landmarks) < 33:
            return

        # 转换为 3D 点
        points_3d = []
        for lm in landmarks:
            if len(lm) >= 3:
                # 坐标转换: MediaPipe 归一化坐标 -> 3D 空间
                # x: 左右 (0-1 -> -0.5-0.5)
                # y: 上下 (0-1 -> 0.5--0.5, 翻转 y 轴)
                # z: 前后 (深度)
                x = (lm[0] - 0.5) * 2
                y = -(lm[1] - 0.5) * 2  # 翻转 y
                z = -lm[2] * 2  # 深度
                points_3d.append([x, y, z])
            else:
                points_3d.append([0, 0, 0])

        points_3d = np.array(points_3d)

        # 记录关键点
        color_rgb = [int(c * 255) for c in color]
        rr.log(
            "skeleton/points",
            rr.Points3D(
                positions=points_3d,
                colors=[color_rgb] * len(points_3d),
                radii=[0.01] * len(points_3d),
            ),
        )

        # 记录连接线
        lines = []
        for start_idx, end_idx in POSE_CONNECTIONS:
            if start_idx < len(points_3d) and end_idx < len(points_3d):
                lines.append([points_3d[start_idx], points_3d[end_idx]])

        if lines:
            rr.log(
                "skeleton/bones",
                rr.LineStrips3D(
                    strips=lines,
                    colors=[color_rgb] * len(lines),
                    radii=[0.005] * len(lines),
                ),
            )

        # 高亮显示关键关节
        key_indices = [11, 12, 15, 16, 23, 24]  # 肩、腕、髋
        key_points = [points_3d[i] for i in key_indices if i < len(points_3d)]

        if key_points:
            rr.log(
                "skeleton/key_joints",
                rr.Points3D(
                    positions=np.array(key_points),
                    colors=[(255, 255, 0)] * len(key_points),  # 黄色
                    radii=[0.02] * len(key_points),
                ),
            )


def main():
    """测试入口"""
    print("=" * 60)
    print("  Movement Chain - Rerun Visualizer")
    print("=" * 60)

    if not RERUN_AVAILABLE:
        print("\n[ERROR] rerun-sdk 未安装")
        print("运行: pip install rerun-sdk>=0.20.0")
        return

    print(f"\nRerun SDK 版本: {rr.__version__}")
    print("\n此模块需要通过 pipeline.py 使用。")
    print("用法: python pipeline.py --video <video.mp4> --imu <imu.csv>")


if __name__ == "__main__":
    main()
