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
from .schemas import (
    AddressPhaseMetrics,
    BackswingPhaseMetrics,
    DownswingPhaseMetrics,
    FinishPhaseMetrics,
    FollowThroughPhaseMetrics,
    FusedFrame,
    FusedSwingData,
    ImpactPhaseMetrics,
    PerPhaseMetrics,
    PhaseIMUMetrics,
    PhaseTimingMetrics,
    PhaseVisionMetrics,
    PoseFrame,
    TakeawayPhaseMetrics,
    TopPhaseMetrics,
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

        # 4. 聚合 Per-Phase Metrics (V2)
        phase_metrics = self.aggregate_phase_metrics(fused_frames, imu_phases)

        # 5. 创建融合数据对象
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
            phase_metrics=phase_metrics,  # V2
        )

        print("[SensorFusion] 融合完成!")
        return fused_data

    def _find_imu_impact(self, imu_df: pd.DataFrame, imu_phases: list[dict[str, Any]]) -> int:
        """从 IMU 数据找到 Impact 时刻索引"""
        # 方法 1: 从阶段中查找
        for phase in imu_phases:
            if phase.get("name", "").lower() == "impact":
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
            # 优先使用预计算的 gyro_magnitude
            gyro_magnitude = None
            if "gyro_mag_dps" in row:
                gyro_magnitude = float(row["gyro_mag_dps"])
            elif "gyro_magnitude" in row:
                gyro_magnitude = float(row["gyro_magnitude"])
            elif gyro_dps:
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
            name = phase.get("name", "unknown")
            name_cn = phase.get("name_cn", name)
            for i in range(int(start_idx), int(end_idx) + 1):
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
        # 尝试不同的列名格式
        # 格式1: gyro_x_dps (来自 imu_swing_analyzer 处理后)
        if "gyro_x_dps" in row:
            return (
                float(row["gyro_x_dps"]),
                float(row["gyro_y_dps"]),
                float(row["gyro_z_dps"]),
            )
        # 格式2: GyX_dps
        if "GyX_dps" in row:
            return (float(row["GyX_dps"]), float(row["GyY_dps"]), float(row["GyZ_dps"]))
        # 格式3: GyX (原始值，需要外部转换)
        if "GyX" in row:
            return (float(row["GyX"]), float(row["GyY"]), float(row["GyZ"]))
        # 格式4: gyro_x
        if "gyro_x" in row:
            return (float(row["gyro_x"]), float(row["gyro_y"]), float(row["gyro_z"]))
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

    # ========================================
    # V2: Per-Phase Metrics Aggregation
    # ========================================

    def aggregate_phase_metrics(
        self,
        fused_frames: list[FusedFrame],
        imu_phases: list[dict[str, Any]],
    ) -> list[PerPhaseMetrics]:
        """
        聚合每个阶段的指标

        Args:
            fused_frames: 融合帧列表
            imu_phases: IMU 阶段列表 (dict 格式)

        Returns:
            list[PerPhaseMetrics] 每阶段的完整指标
        """
        print("[SensorFusion] 聚合 Per-Phase Metrics...")
        phase_metrics_list = []

        for phase_info in imu_phases:
            phase_name = phase_info.get("name", "unknown")
            phase_name_cn = phase_info.get("name_cn", phase_name)
            start_idx = phase_info.get("start_idx", 0)
            end_idx = phase_info.get("end_idx", start_idx)

            # 过滤该阶段的帧
            phase_frames = [
                f for f in fused_frames
                if start_idx <= f.frame_idx <= end_idx
            ]

            if not phase_frames:
                continue

            # 计算各类指标
            timing = self._compute_timing_metrics(phase_frames, phase_info)
            imu_metrics = self._compute_phase_imu_metrics(phase_frames)
            vision_metrics = self._compute_phase_vision_metrics(phase_frames)
            phase_specific = self._compute_phase_specific_metrics(
                phase_name, phase_frames, imu_metrics, vision_metrics
            )

            # 构建 PerPhaseMetrics
            per_phase = PerPhaseMetrics(
                phase_name=phase_name,
                phase_name_cn=phase_name_cn,
                timing=timing,
                imu=imu_metrics,
                vision=vision_metrics,
                emg=None,  # EMG 预留
                **phase_specific,
            )

            phase_metrics_list.append(per_phase)

        print(f"[SensorFusion] 聚合完成: {len(phase_metrics_list)} 个阶段")
        return phase_metrics_list

    def _compute_timing_metrics(
        self,
        frames: list[FusedFrame],
        phase_info: dict[str, Any],
    ) -> PhaseTimingMetrics:
        """计算阶段时间指标"""
        if not frames:
            return PhaseTimingMetrics(
                start_ms=0, end_ms=0, duration_ms=0, frame_count=0
            )

        start_ms = frames[0].timestamp_ms
        end_ms = frames[-1].timestamp_ms
        duration_ms = phase_info.get("duration_ms", end_ms - start_ms)

        return PhaseTimingMetrics(
            start_ms=start_ms,
            end_ms=end_ms,
            duration_ms=duration_ms,
            frame_count=len(frames),
        )

    def _compute_phase_imu_metrics(
        self,
        frames: list[FusedFrame],
    ) -> PhaseIMUMetrics | None:
        """计算阶段 IMU 指标"""
        gyro_mags = [f.gyro_magnitude for f in frames if f.gyro_magnitude is not None]
        gyro_xs = [f.gyro_dps[0] for f in frames if f.gyro_dps is not None]
        gyro_ys = [f.gyro_dps[1] for f in frames if f.gyro_dps is not None]
        gyro_zs = [f.gyro_dps[2] for f in frames if f.gyro_dps is not None]

        if not gyro_mags:
            return None

        gyro_array = np.array(gyro_mags)
        std = np.std(gyro_array)
        # 稳定性评分: 标准差越小越稳定，归一化到 0-100
        stability_score = max(0, 100 - std / 10) if std > 0 else 100.0

        return PhaseIMUMetrics(
            gyro_magnitude_max=float(np.max(gyro_array)),
            gyro_magnitude_avg=float(np.mean(gyro_array)),
            gyro_magnitude_min=float(np.min(gyro_array)),
            gyro_stability_score=float(stability_score),
            gyro_x_max=float(max(gyro_xs)) if gyro_xs else None,
            gyro_y_max=float(max(gyro_ys)) if gyro_ys else None,
            gyro_z_max=float(max(gyro_zs)) if gyro_zs else None,
        )

    def _compute_phase_vision_metrics(
        self,
        frames: list[FusedFrame],
    ) -> PhaseVisionMetrics | None:
        """计算阶段视觉指标"""
        x_factors = [f.x_factor for f in frames if f.x_factor is not None]
        left_arms = [f.left_arm_angle for f in frames if f.left_arm_angle is not None]
        right_arms = [f.right_arm_angle for f in frames if f.right_arm_angle is not None]

        if not x_factors and not left_arms:
            return None

        # X-Factor 统计
        x_factor_start = x_factors[0] if x_factors else None
        x_factor_end = x_factors[-1] if x_factors else None
        x_factor_max = max(x_factors) if x_factors else None
        x_factor_min = min(x_factors) if x_factors else None
        x_factor_delta = None
        if x_factor_start is not None and x_factor_end is not None:
            x_factor_delta = x_factor_end - x_factor_start

        # 手臂角度平均
        left_arm_avg = float(np.mean(left_arms)) if left_arms else None
        right_arm_avg = float(np.mean(right_arms)) if right_arms else None

        # 头部位移 (需要 landmarks 数据，这里简化处理)
        head_displacement = self._compute_head_displacement(frames)

        return PhaseVisionMetrics(
            x_factor_start=x_factor_start,
            x_factor_end=x_factor_end,
            x_factor_max=x_factor_max,
            x_factor_min=x_factor_min,
            x_factor_delta=x_factor_delta,
            left_arm_angle_avg=left_arm_avg,
            right_arm_angle_avg=right_arm_avg,
            head_displacement_cm=head_displacement,
        )

    def _compute_head_displacement(self, frames: list[FusedFrame]) -> float | None:
        """计算头部位移 (基于 landmarks)"""
        head_positions = []
        for f in frames:
            if f.pose_landmarks and len(f.pose_landmarks) > 0:
                # 鼻子位置 (landmark 0)
                nose = f.pose_landmarks[0]
                if len(nose) >= 2:
                    head_positions.append((nose[0], nose[1]))

        if len(head_positions) < 2:
            return None

        # 计算最大位移 (像素)
        first_pos = np.array(head_positions[0])
        max_displacement = 0.0
        for pos in head_positions[1:]:
            dist = np.linalg.norm(np.array(pos) - first_pos)
            max_displacement = max(max_displacement, dist)

        # 转换为厘米 (假设 1 像素 ≈ 0.1 cm，需根据实际校准)
        return float(max_displacement * 0.1)

    def _compute_phase_specific_metrics(
        self,
        phase_name: str,
        frames: list[FusedFrame],
        imu_metrics: PhaseIMUMetrics | None,
        vision_metrics: PhaseVisionMetrics | None,
    ) -> dict[str, Any]:
        """计算阶段特定指标"""
        phase_lower = phase_name.lower()
        result = {}

        if phase_lower == "address":
            result["address"] = self._compute_address_metrics(frames, imu_metrics)
        elif phase_lower == "takeaway":
            result["takeaway"] = self._compute_takeaway_metrics(frames, imu_metrics)
        elif phase_lower == "backswing":
            result["backswing"] = self._compute_backswing_metrics(
                frames, imu_metrics, vision_metrics
            )
        elif phase_lower == "top":
            result["top"] = self._compute_top_metrics(frames, vision_metrics)
        elif phase_lower == "downswing":
            result["downswing"] = self._compute_downswing_metrics(frames, imu_metrics)
        elif phase_lower == "impact":
            result["impact"] = self._compute_impact_metrics(
                frames, imu_metrics, vision_metrics
            )
        elif phase_lower in ["follow_through", "follow-through", "followthrough"]:
            result["follow_through"] = self._compute_follow_through_metrics(
                frames, imu_metrics
            )
        elif phase_lower == "finish":
            result["finish"] = self._compute_finish_metrics(frames, imu_metrics)

        return result

    def _compute_address_metrics(
        self,
        frames: list[FusedFrame],
        imu_metrics: PhaseIMUMetrics | None,
    ) -> AddressPhaseMetrics:
        """计算 Address 阶段特定指标"""
        stability = imu_metrics.gyro_stability_score if imu_metrics else None
        return AddressPhaseMetrics(
            stability_score=stability,
            spine_angle_deg=None,  # 需要更复杂的姿态计算
            stance_width_ratio=None,  # 需要更复杂的姿态计算
        )

    def _compute_takeaway_metrics(
        self,
        frames: list[FusedFrame],
        imu_metrics: PhaseIMUMetrics | None,
    ) -> TakeawayPhaseMetrics:
        """计算 Takeaway 阶段特定指标"""
        # 计算初始加速度 (陀螺仪变化率)
        initial_accel = None
        if len(frames) >= 2:
            gyro_mags = [f.gyro_magnitude for f in frames[:5] if f.gyro_magnitude]
            if len(gyro_mags) >= 2:
                initial_accel = (gyro_mags[-1] - gyro_mags[0]) / (len(gyro_mags) - 1)

        return TakeawayPhaseMetrics(
            initial_acceleration_dps2=initial_accel,
            rotation_start_ms=frames[0].timestamp_ms if frames else None,
        )

    def _compute_backswing_metrics(
        self,
        frames: list[FusedFrame],
        imu_metrics: PhaseIMUMetrics | None,
        vision_metrics: PhaseVisionMetrics | None,
    ) -> BackswingPhaseMetrics:
        """计算 Backswing 阶段特定指标"""
        # X-Factor 增长速率
        buildup_rate = None
        if vision_metrics and vision_metrics.x_factor_delta is not None:
            duration_s = (frames[-1].timestamp_ms - frames[0].timestamp_ms) / 1000
            if duration_s > 0:
                buildup_rate = vision_metrics.x_factor_delta / duration_s

        return BackswingPhaseMetrics(
            x_factor_buildup_rate=buildup_rate,
            shoulder_turn_deg=None,  # 需要复杂姿态计算
            hip_turn_deg=None,  # 需要复杂姿态计算
            sway_cm=vision_metrics.head_displacement_cm if vision_metrics else None,
        )

    def _compute_top_metrics(
        self,
        frames: list[FusedFrame],
        vision_metrics: PhaseVisionMetrics | None,
    ) -> TopPhaseMetrics:
        """计算 Top 阶段特定指标"""
        x_factor_max = vision_metrics.x_factor_max if vision_metrics else None

        # 前臂伸展百分比 (基于手臂角度，180度为完全伸直)
        lead_arm_ext = None
        if vision_metrics and vision_metrics.left_arm_angle_avg is not None:
            lead_arm_ext = min(100, vision_metrics.left_arm_angle_avg / 180 * 100)

        # 停顿时间
        pause_duration = None
        if len(frames) >= 2:
            pause_duration = frames[-1].timestamp_ms - frames[0].timestamp_ms

        return TopPhaseMetrics(
            x_factor_max_deg=x_factor_max,
            lead_arm_extension_pct=lead_arm_ext,
            pause_duration_ms=pause_duration,
        )

    def _compute_downswing_metrics(
        self,
        frames: list[FusedFrame],
        imu_metrics: PhaseIMUMetrics | None,
    ) -> DownswingPhaseMetrics:
        """计算 Downswing 阶段特定指标"""
        peak_velocity = imu_metrics.gyro_magnitude_max if imu_metrics else None

        # 加速率
        accel_rate = None
        gyro_mags = [f.gyro_magnitude for f in frames if f.gyro_magnitude]
        if len(gyro_mags) >= 2:
            duration_s = (frames[-1].timestamp_ms - frames[0].timestamp_ms) / 1000
            if duration_s > 0:
                accel_rate = (max(gyro_mags) - gyro_mags[0]) / duration_s

        # 手腕释放点 (找到加速度最大的位置)
        wrist_release = None
        if gyro_mags:
            peak_idx = gyro_mags.index(max(gyro_mags))
            wrist_release = (peak_idx / len(gyro_mags)) * 100

        return DownswingPhaseMetrics(
            peak_velocity_dps=peak_velocity,
            acceleration_rate_dps2=accel_rate,
            hip_lead_ms=None,  # 需要复杂姿态计算
            wrist_release_point_pct=wrist_release,
        )

    def _compute_impact_metrics(
        self,
        frames: list[FusedFrame],
        imu_metrics: PhaseIMUMetrics | None,
        vision_metrics: PhaseVisionMetrics | None,
    ) -> ImpactPhaseMetrics:
        """计算 Impact 阶段特定指标"""
        velocity_at_impact = imu_metrics.gyro_magnitude_max if imu_metrics else None

        # 头部稳定性检查
        head_stable = None
        if vision_metrics and vision_metrics.head_displacement_cm is not None:
            head_stable = vision_metrics.head_displacement_cm < 5.0  # 5cm 阈值

        return ImpactPhaseMetrics(
            velocity_at_impact_dps=velocity_at_impact,
            head_stable=head_stable,
            weight_shift_pct=None,  # 需要复杂姿态计算
        )

    def _compute_follow_through_metrics(
        self,
        frames: list[FusedFrame],
        imu_metrics: PhaseIMUMetrics | None,
    ) -> FollowThroughPhaseMetrics:
        """计算 Follow-through 阶段特定指标"""
        # 减速率
        decel_rate = None
        gyro_mags = [f.gyro_magnitude for f in frames if f.gyro_magnitude]
        if len(gyro_mags) >= 2:
            duration_s = (frames[-1].timestamp_ms - frames[0].timestamp_ms) / 1000
            if duration_s > 0:
                decel_rate = (gyro_mags[0] - gyro_mags[-1]) / duration_s

        return FollowThroughPhaseMetrics(
            deceleration_rate_dps2=decel_rate,
            rotation_completion_pct=None,  # 需要复杂姿态计算
        )

    def _compute_finish_metrics(
        self,
        frames: list[FusedFrame],
        imu_metrics: PhaseIMUMetrics | None,
    ) -> FinishPhaseMetrics:
        """计算 Finish 阶段特定指标"""
        stability = imu_metrics.gyro_stability_score if imu_metrics else None

        # 平衡评分 (基于陀螺仪稳定性)
        balance_score = stability

        return FinishPhaseMetrics(
            final_stability_score=stability,
            balance_score=balance_score,
        )


def main():
    """测试入口"""
    print("=" * 60)
    print("  Movement Chain - Sensor Fusion")
    print("=" * 60)
    print("\n此模块需要通过 pipeline.py 使用，不支持独立运行。")
    print("用法: python pipeline.py --video <video.mp4> --imu <imu.csv>")


if __name__ == "__main__":
    main()
