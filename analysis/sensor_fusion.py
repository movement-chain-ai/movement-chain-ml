"""
Movement Chain - Sensor Fusion (传感器融合)

将 Vision (MediaPipe) 和 IMU 数据进行时间对齐和融合。
使用 Impact 时刻作为同步锚点。

用法:
    from sensor_fusion import SensorFusion

    fusion = SensorFusion()
    fused_data = fusion.fuse(vision_result, imu_result)

Author: Movement Chain AI Team
Date: 2025-01-15
"""

import uuid
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from schemas import (
    FusedFrame,
    FusedSwingData,
    PoseFrame,
    VisionResult,
)


class SensorFusion:
    """传感器融合处理器"""

    def __init__(self):
        """初始化融合器"""
        pass

    def fuse(
        self,
        vision_result: VisionResult,
        imu_df: pd.DataFrame,
        imu_phases: list[dict[str, Any]],
        imu_metrics: dict[str, Any],
        imu_report: dict[str, Any],
        manual_impact_frame: int | None = None,
    ) -> FusedSwingData:
        """
        融合 Vision 和 IMU 数据

        Args:
            vision_result: 视频分析结果
            imu_df: IMU 数据 DataFrame (来自 imu_swing_analyzer)
            imu_phases: 阶段列表 (SwingPhase 转为 dict)
            imu_metrics: 指标 (SwingMetrics 转为 dict)
            imu_report: 完整报告
            manual_impact_frame: 手动指定的 Impact 帧 (可选)

        Returns:
            FusedSwingData 融合后的数据
        """
        print("[SensorFusion] 开始融合...")

        # 1. 确定 Impact 时刻
        vision_impact_frame = manual_impact_frame or vision_result.impact_frame_idx
        if vision_impact_frame is None:
            print("[WARN] 未检测到 Vision Impact 帧，使用中点作为估计")
            vision_impact_frame = len(vision_result.frames) // 2

        imu_impact_idx = self._find_imu_impact(imu_df, imu_phases)

        print(f"[SensorFusion] Vision Impact 帧: {vision_impact_frame}")
        print(f"[SensorFusion] IMU Impact 索引: {imu_impact_idx}")

        # 2. 计算时间偏移
        vision_impact_time_ms = vision_impact_frame * (1000.0 / vision_result.fps)
        imu_impact_time_ms = self._get_imu_timestamp(imu_df, imu_impact_idx)

        time_offset_ms = imu_impact_time_ms - vision_impact_time_ms
        print(f"[SensorFusion] 时间偏移: {time_offset_ms:.1f} ms")

        # 3. 构建时间对齐的融合帧
        fused_frames = self._create_fused_frames(
            vision_result=vision_result,
            imu_df=imu_df,
            imu_phases=imu_phases,
            time_offset_ms=time_offset_ms,
        )

        print(f"[SensorFusion] 生成 {len(fused_frames)} 个融合帧")

        # 4. 创建融合数据对象
        session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        fused_data = FusedSwingData(
            session_id=session_id,
            video_file=vision_result.video_file,
            imu_file=imu_report.get("source_file", "unknown"),
            analysis_time=datetime.now().isoformat(),
            alignment_offset_ms=time_offset_ms,
            frames=fused_frames,
            vision_fps=vision_result.fps,
            vision_metrics=vision_result.metrics,
            imu_phases=imu_phases,
            imu_metrics=imu_metrics,
            imu_report=imu_report,
        )

        print("[SensorFusion] 融合完成!")
        return fused_data

    def _find_imu_impact(self, imu_df: pd.DataFrame, imu_phases: list[dict[str, Any]]) -> int:
        """从 IMU 数据找到 Impact 时刻索引"""
        # 方法 1: 从阶段中查找
        for phase in imu_phases:
            if phase.get("phase", "").lower() == "impact":
                return phase.get("start_idx", 0)

        # 方法 2: 找陀螺仪峰值
        if "gyro_magnitude" in imu_df.columns:
            return int(imu_df["gyro_magnitude"].idxmax())

        # 方法 3: 计算合成陀螺仪并找峰值
        if all(col in imu_df.columns for col in ["GyX_dps", "GyY_dps", "GyZ_dps"]):
            gyro_mag = np.sqrt(
                imu_df["GyX_dps"] ** 2 + imu_df["GyY_dps"] ** 2 + imu_df["GyZ_dps"] ** 2
            )
            return int(gyro_mag.idxmax())

        # 默认: 使用中点
        return len(imu_df) // 2

    def _get_imu_timestamp(self, imu_df: pd.DataFrame, idx: int) -> float:
        """获取 IMU 数据在指定索引的时间戳 (毫秒)"""
        if idx >= len(imu_df):
            idx = len(imu_df) - 1

        if "time_ms" in imu_df.columns:
            return float(imu_df.iloc[idx]["time_ms"])

        if "timestamp" in imu_df.columns:
            timestamps = pd.to_datetime(imu_df["timestamp"])
            first_ts = timestamps.iloc[0]
            current_ts = timestamps.iloc[idx]
            return (current_ts - first_ts).total_seconds() * 1000

        # 假设 1000Hz 采样率
        return float(idx)

    def _create_fused_frames(
        self,
        vision_result: VisionResult,
        imu_df: pd.DataFrame,
        imu_phases: list[dict[str, Any]],
        time_offset_ms: float,
    ) -> list[FusedFrame]:
        """
        创建时间对齐的融合帧

        策略:
        1. 以 IMU 时间为基准
        2. 将 Vision 帧插值到 IMU 采样时刻
        """
        fused_frames = []

        # 预处理 Vision 帧为时间索引
        vision_frames_by_time = {}
        for vf in vision_result.frames:
            # 应用时间偏移
            aligned_time = vf.timestamp_ms + time_offset_ms
            vision_frames_by_time[aligned_time] = vf

        vision_times = sorted(vision_frames_by_time.keys())

        # 预处理阶段信息
        phase_map = self._build_phase_map(imu_phases)

        # 遍历 IMU 数据
        for idx, row in imu_df.iterrows():
            imu_time_ms = self._get_imu_timestamp(imu_df, idx)

            # 查找最近的 Vision 帧
            vision_frame = self._find_nearest_vision_frame(
                imu_time_ms, vision_times, vision_frames_by_time
            )

            # 获取 IMU 数据
            gyro_dps = self._extract_gyro(row)
            accel_g = self._extract_accel(row)
            gyro_magnitude = None
            if gyro_dps:
                gyro_magnitude = np.sqrt(sum(g**2 for g in gyro_dps))

            # 获取阶段信息
            phase_name, phase_name_cn = self._get_phase_at_idx(idx, phase_map)

            fused_frame = FusedFrame(
                timestamp_ms=imu_time_ms,
                frame_idx=int(idx),
                has_vision=vision_frame is not None,
                pose_landmarks=vision_frame.landmarks if vision_frame else None,
                left_arm_angle=vision_frame.left_arm_angle if vision_frame else None,
                right_arm_angle=vision_frame.right_arm_angle if vision_frame else None,
                x_factor=vision_frame.x_factor if vision_frame else None,
                gyro_dps=gyro_dps,
                accel_g=accel_g,
                gyro_magnitude=gyro_magnitude,
                phase_name=phase_name,
                phase_name_cn=phase_name_cn,
            )
            fused_frames.append(fused_frame)

        return fused_frames

    def _build_phase_map(self, imu_phases: list[dict[str, Any]]) -> dict[int, tuple[str, str]]:
        """构建索引到阶段的映射"""
        phase_map = {}
        for phase in imu_phases:
            start_idx = phase.get("start_idx", 0)
            end_idx = phase.get("end_idx", start_idx)
            name = phase.get("phase", "unknown")
            name_cn = phase.get("phase_cn", name)
            for i in range(start_idx, end_idx + 1):
                phase_map[i] = (name, name_cn)
        return phase_map

    def _get_phase_at_idx(
        self, idx: int, phase_map: dict[int, tuple[str, str]]
    ) -> tuple[str | None, str | None]:
        """获取指定索引的阶段信息"""
        if idx in phase_map:
            return phase_map[idx]
        return None, None

    def _find_nearest_vision_frame(
        self,
        target_time: float,
        vision_times: list[float],
        vision_frames: dict[float, PoseFrame],
        max_gap_ms: float = 100.0,
    ) -> PoseFrame | None:
        """查找最近的 Vision 帧"""
        if not vision_times:
            return None

        # 二分查找最近时间
        import bisect

        pos = bisect.bisect_left(vision_times, target_time)

        candidates = []
        if pos > 0:
            candidates.append(vision_times[pos - 1])
        if pos < len(vision_times):
            candidates.append(vision_times[pos])

        if not candidates:
            return None

        # 找最近的
        nearest_time = min(candidates, key=lambda t: abs(t - target_time))
        gap = abs(nearest_time - target_time)

        if gap > max_gap_ms:
            return None

        return vision_frames.get(nearest_time)

    def _extract_gyro(self, row) -> tuple[float, float, float] | None:
        """从行数据提取陀螺仪值"""
        # 尝试不同的列名
        for x_col in ["GyX_dps", "GyX", "gyro_x"]:
            if x_col in row:
                y_col = x_col.replace("X", "Y").replace("x", "y")
                z_col = x_col.replace("X", "Z").replace("x", "z")
                if y_col in row and z_col in row:
                    return (float(row[x_col]), float(row[y_col]), float(row[z_col]))
        return None

    def _extract_accel(self, row) -> tuple[float, float, float] | None:
        """从行数据提取加速度值"""
        for x_col in ["AcX", "accel_x", "ax"]:
            if x_col in row:
                y_col = x_col.replace("X", "Y").replace("x", "y")
                z_col = x_col.replace("X", "Z").replace("x", "z")
                if y_col in row and z_col in row:
                    return (float(row[x_col]), float(row[y_col]), float(row[z_col]))
        return None


def main():
    """测试入口"""
    print("=" * 60)
    print("  Movement Chain - Sensor Fusion")
    print("=" * 60)
    print("\n此模块需要通过 pipeline.py 使用，不支持独立运行。")
    print("用法: python pipeline.py --video <video.mp4> --imu <imu.csv>")


if __name__ == "__main__":
    main()
