#!/usr/bin/env python3
"""
Movement Chain - IMU Swing Analyzer
MVP 验证脚本：分析手部 IMU 数据，检测挥杆阶段，计算核心指标

Author: Movement Chain AI Team
Date: 2026-01-14
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Tuple, Optional, ClassVar
from pathlib import Path
import json
from datetime import datetime

# ============================================================
# 配置参数
# ============================================================


@dataclass
class IMUConfig:
    """IMU 配置参数"""

    # MPU6050 灵敏度设置 (LSB per °/s)
    GYRO_SENSITIVITY: ClassVar[Dict[int, float]] = {
        250: 131.0,  # ±250°/s
        500: 65.5,  # ±500°/s
        1000: 32.8,  # ±1000°/s
        2000: 16.4,  # ±2000°/s
    }

    # 默认使用 ±2000°/s，但会提供选项
    gyro_range: int = 2000

    # 阶段检测阈值
    address_threshold_dps: float = 30.0  # 静止判定阈值
    movement_start_threshold_dps: float = 50.0  # 运动开始阈值

    # 消抖参数
    debounce_samples: int = 5  # 需要连续 N 个采样点才确认状态变化


@dataclass
class SwingPhase:
    """挥杆阶段数据"""

    name: str
    name_cn: str
    start_idx: int
    end_idx: int
    start_time_ms: float
    end_time_ms: float
    duration_ms: float
    peak_gyro_dps: float

    def __post_init__(self) -> None:
        """Validate invariants at construction time."""
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

    # 核心 5 项 (原有)
    peak_angular_velocity_dps: float
    backswing_duration_ms: float
    downswing_duration_ms: float
    total_swing_time_ms: float
    tempo_ratio: Optional[float]  # Can be None if downswing_duration is 0

    # 新增 2 项
    wrist_release_point_pct: Optional[float]  # 手腕释放点 (下杆完成百分比)
    acceleration_time_ms: Optional[float]  # 加速时段 (毫秒)

    # 评估等级
    velocity_level: str
    tempo_level: str
    wrist_release_level: str
    acceleration_level: str
    overall_level: str

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
# 数据加载与预处理
# ============================================================


def load_imu_data(filepath: str) -> pd.DataFrame:
    """
    加载 IMU CSV 数据

    Args:
        filepath: CSV 文件路径

    Returns:
        DataFrame with columns: timestamp, AcX, AcY, AcZ, GyX, GyY, GyZ, Tmp

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file cannot be parsed or contains no valid data
    """
    # Validate file exists
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"IMU data file not found: {filepath}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {filepath}")

    # Read file with error handling
    try:
        # Count rows before skipping bad lines
        with open(filepath, "r", encoding="utf-8") as f:
            total_lines = sum(1 for line in f if not line.startswith("#"))

        df = pd.read_csv(
            filepath,
            comment="#",
            names=["timestamp", "AcX", "AcY", "AcZ", "GyX", "GyY", "GyZ", "Tmp"],
            on_bad_lines="skip",  # 跳过格式错误的行
        )

        # Log if rows were skipped
        skipped_lines = total_lines - len(df)
        if skipped_lines > 0:
            print(f"⚠️ 警告: 跳过 {skipped_lines} 行格式错误的数据")

    except pd.errors.ParserError as e:
        raise ValueError(f"Failed to parse CSV file {filepath}: {e}") from e
    except UnicodeDecodeError as e:
        raise ValueError(f"Encoding error reading {filepath}. Expected UTF-8: {e}") from e

    # 转换时间戳为 datetime
    original_count = len(df)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # 删除时间戳无效的行
    df = df.dropna(subset=["timestamp"])
    timestamp_dropped = original_count - len(df)
    if timestamp_dropped > 0:
        print(f"⚠️ 警告: 删除 {timestamp_dropped} 行无效时间戳数据")

    # 确保数值列是数值类型
    numeric_cols = ["AcX", "AcY", "AcZ", "GyX", "GyY", "GyZ", "Tmp"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 删除包含无效数值的行 (例如 -1 表示传感器错误)
    before_filter = len(df)
    df = df[(df["GyX"] != -1) & (df["GyY"] != -1) & (df["GyZ"] != -1)]
    sensor_errors = before_filter - len(df)
    if sensor_errors > 0:
        print(f"⚠️ 警告: 删除 {sensor_errors} 行传感器错误数据 (值为 -1)")

    # Validate we have enough data
    if len(df) == 0:
        raise ValueError(
            f"No valid IMU data found in {filepath}. "
            "Check that the file contains valid timestamps and sensor readings."
        )

    if len(df) < 10:
        raise ValueError(
            f"Insufficient data: only {len(df)} valid rows found in {filepath}. "
            "Need at least 10 samples for analysis."
        )

    # 计算相对时间 (毫秒)
    df["time_ms"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds() * 1000

    # 重置索引
    df = df.reset_index(drop=True)

    print(f"✅ 加载数据: {len(df)} 行有效数据")
    print(f"   时间范围: {df['time_ms'].iloc[0]:.1f}ms - {df['time_ms'].iloc[-1]:.1f}ms")

    # Safe sampling rate calculation (avoid division by zero)
    time_diff_median = df["time_ms"].diff().median()
    if time_diff_median > 0:
        print(f"   采样率估计: {1000 / time_diff_median:.1f} Hz")
    else:
        print("   采样率估计: 无法计算 (时间差为0)")

    return df


def convert_to_dps(df: pd.DataFrame, gyro_range: int = 2000) -> pd.DataFrame:
    """
    将原始陀螺仪值转换为 °/s

    Args:
        df: 原始数据 DataFrame
        gyro_range: 陀螺仪量程设置 (250, 500, 1000, 2000)

    Returns:
        添加了 gyro_x_dps, gyro_y_dps, gyro_z_dps, gyro_mag_dps 列的 DataFrame

    Raises:
        ValueError: If gyro_range is not a valid value
    """
    # Validate gyro_range parameter
    if gyro_range not in IMUConfig.GYRO_SENSITIVITY:
        valid_ranges = list(IMUConfig.GYRO_SENSITIVITY.keys())
        raise ValueError(f"Invalid gyro_range: {gyro_range}. Valid options: {valid_ranges}")

    sensitivity = IMUConfig.GYRO_SENSITIVITY[gyro_range]

    df["gyro_x_dps"] = df["GyX"] / sensitivity
    df["gyro_y_dps"] = df["GyY"] / sensitivity
    df["gyro_z_dps"] = df["GyZ"] / sensitivity

    # 计算合成角速度 (magnitude)
    df["gyro_mag_dps"] = np.sqrt(
        df["gyro_x_dps"] ** 2 + df["gyro_y_dps"] ** 2 + df["gyro_z_dps"] ** 2
    )

    print(f"✅ 单位转换完成 (量程: ±{gyro_range}°/s, 灵敏度: {sensitivity} LSB/(°/s))")
    print(f"   角速度范围: {df['gyro_mag_dps'].min():.1f} - {df['gyro_mag_dps'].max():.1f} °/s")

    return df


# ============================================================
# 挥杆隔离 (从多次动作中提取单次挥杆)
# ============================================================


def isolate_swing(
    df: pd.DataFrame,
    window_before_ms: float = 2000,
    window_after_ms: float = 1500,
    min_peak_velocity: float = 300,
) -> Tuple[pd.DataFrame, Dict]:
    """
    从数据中自动隔离单次挥杆

    通过找到最大角速度点(Impact)，然后截取前后一定时间窗口的数据。

    Args:
        df: 处理后的数据 DataFrame (需要有 gyro_mag_dps 和 time_ms)
        window_before_ms: Impact 前保留的时间 (毫秒)，默认 2000ms
        window_after_ms: Impact 后保留的时间 (毫秒)，默认 1500ms
        min_peak_velocity: 最小峰值速度阈值，低于此值认为没有有效挥杆

    Returns:
        (隔离后的 DataFrame, 隔离信息字典)

    Raises:
        ValueError: If required columns are missing from DataFrame
    """
    # Validate required columns exist
    required_cols = ["gyro_mag_dps", "time_ms"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"DataFrame missing required columns: {missing}. "
            "Did you call convert_to_dps() first?"
        )

    # 找到最大角速度点 (Impact)
    impact_idx = df["gyro_mag_dps"].idxmax()
    peak_velocity = df.loc[impact_idx, "gyro_mag_dps"]
    impact_time = df.loc[impact_idx, "time_ms"]

    # 检查是否有有效挥杆
    if peak_velocity < min_peak_velocity:
        print(f"⚠️ 警告: 峰值速度 {peak_velocity:.1f}°/s 低于阈值 {min_peak_velocity}°/s")
        print("   可能没有有效的挥杆动作")

    # 计算截取窗口
    original_start = df["time_ms"].iloc[0]
    original_end = df["time_ms"].iloc[-1]
    original_duration = original_end - original_start

    start_time = max(original_start, impact_time - window_before_ms)
    end_time = min(original_end, impact_time + window_after_ms)

    # 截取数据
    mask = (df["time_ms"] >= start_time) & (df["time_ms"] <= end_time)
    segment = df[mask].copy()

    # 重置时间为从 0 开始
    time_offset = segment["time_ms"].iloc[0]
    segment["time_ms"] = segment["time_ms"] - time_offset
    segment = segment.reset_index(drop=True)

    # 记录隔离信息
    isolation_info = {
        "original_duration_ms": round(original_duration, 1),
        "isolated_duration_ms": round(end_time - start_time, 1),
        "impact_time_original_ms": round(impact_time, 1),
        "impact_time_isolated_ms": round(impact_time - time_offset, 1),
        "window_before_ms": window_before_ms,
        "window_after_ms": window_after_ms,
        "rows_before": len(df),
        "rows_after": len(segment),
        "data_reduction_pct": round((1 - len(segment) / len(df)) * 100, 1),
    }

    print(f"✅ 挥杆隔离完成:")
    print(f"   原始时长: {original_duration:.0f}ms → 隔离后: {end_time - start_time:.0f}ms")
    print(
        f"   数据量: {len(df)} 行 → {len(segment)} 行 (减少 {isolation_info['data_reduction_pct']:.0f}%)"
    )
    print(f"   Impact 位置: {impact_time:.0f}ms → {impact_time - time_offset:.0f}ms (相对时间)")

    return segment, isolation_info


# ============================================================
# 阶段检测算法
# ============================================================


def detect_swing_phases(df: pd.DataFrame, config: Optional[IMUConfig] = None) -> List[SwingPhase]:
    """
    检测挥杆的 8 个阶段

    基于角速度特征检测：
    1. Address: 静止期 (gyro < threshold)
    2. Takeaway: 开始运动 (gyro 开始增大)
    3. Backswing: 上杆期 (gyro 持续增大，方向为正)
    4. Top: 顶点 (gyro 零交叉，从正变负)
    5. Transition: 转换期 (零交叉后的短暂期间)
    6. Downswing: 下杆期 (gyro 负值快速增大)
    7. Impact: 击球 (gyro 达到负峰值)
    8. Follow-through: 收杆 (gyro 减速回归)

    Args:
        df: 处理后的数据 DataFrame (需要有 gyro_mag_dps 和 gyro_y_dps)
        config: IMU 配置参数

    Returns:
        检测到的阶段列表
    """
    if config is None:
        config = IMUConfig()

    phases = []
    gyro_mag = df["gyro_mag_dps"].values
    gyro_y = df["gyro_y_dps"].values  # 主要旋转轴
    time_ms = df["time_ms"].values
    n = len(df)

    # ============================================================
    # 1. 检测 Address (静止期)
    # ============================================================
    address_end = 0
    for i in range(n):
        # 找到第一个超过阈值的点
        if gyro_mag[i] > config.movement_start_threshold_dps:
            # 往前找稳定的静止期
            address_end = max(0, i - config.debounce_samples)
            break

    if address_end > 0:
        phases.append(
            SwingPhase(
                name="Address",
                name_cn="准备",
                start_idx=0,
                end_idx=address_end,
                start_time_ms=time_ms[0],
                end_time_ms=time_ms[address_end],
                duration_ms=time_ms[address_end] - time_ms[0],
                peak_gyro_dps=gyro_mag[0:address_end].max() if address_end > 0 else 0,
            )
        )

    # ============================================================
    # 2. 检测 Top (顶点) - 零交叉检测
    # ============================================================
    # 寻找 gyro_y 从正变负的零交叉点
    top_idx = None
    for i in range(address_end + 10, n - 10):
        # 检查是否是从正变负的零交叉
        if gyro_y[i - 1] > 0 and gyro_y[i] < 0:
            # 验证这是真正的顶点（之前一段时间都是正值）
            prev_window = gyro_y[max(0, i - 20) : i]
            if np.mean(prev_window) > 50:  # 之前确实在上杆
                top_idx = i
                break

    # ============================================================
    # 3. 检测 Impact (峰值) - 在 Top 之后找最大值
    # ============================================================
    if top_idx is not None:
        # 在 Top 之后寻找峰值
        search_start = top_idx
        search_end = min(n, top_idx + int(n * 0.4))  # Top 后 40% 的范围内

        impact_idx = search_start + np.argmax(gyro_mag[search_start:search_end])
        peak_velocity = gyro_mag[impact_idx]
    else:
        # 如果没找到 Top，直接找全局峰值
        print("⚠️ 警告: 无法通过零交叉检测 'Top' 阶段。" "使用峰值速度估算，结果可能不准确。")
        impact_idx = np.argmax(gyro_mag)
        peak_velocity = gyro_mag[impact_idx]
        # 估计 Top 位置（峰值前的某个点）
        top_idx = max(0, impact_idx - int((impact_idx - address_end) * 0.3))

    # ============================================================
    # 4. 构建其他阶段
    # ============================================================

    # Takeaway: Address 结束到 Backswing 开始
    takeaway_end = address_end + int((top_idx - address_end) * 0.2)
    if takeaway_end > address_end:
        phases.append(
            SwingPhase(
                name="Takeaway",
                name_cn="起杆",
                start_idx=address_end,
                end_idx=takeaway_end,
                start_time_ms=time_ms[address_end],
                end_time_ms=time_ms[takeaway_end],
                duration_ms=time_ms[takeaway_end] - time_ms[address_end],
                peak_gyro_dps=gyro_mag[address_end:takeaway_end].max(),
            )
        )

    # Backswing: Takeaway 到 Top
    backswing_start = takeaway_end
    if backswing_start < top_idx:
        phases.append(
            SwingPhase(
                name="Backswing",
                name_cn="上杆",
                start_idx=backswing_start,
                end_idx=top_idx,
                start_time_ms=time_ms[backswing_start],
                end_time_ms=time_ms[top_idx],
                duration_ms=time_ms[top_idx] - time_ms[backswing_start],
                peak_gyro_dps=gyro_mag[backswing_start:top_idx].max(),
            )
        )

    # Top: 瞬时点
    phases.append(
        SwingPhase(
            name="Top",
            name_cn="顶点",
            start_idx=top_idx,
            end_idx=top_idx + 1,
            start_time_ms=time_ms[top_idx],
            end_time_ms=time_ms[min(top_idx + 1, n - 1)],
            duration_ms=0,  # 瞬时
            peak_gyro_dps=gyro_mag[top_idx],
        )
    )

    # Transition: Top 后的短暂期间
    transition_end = top_idx + max(1, int((impact_idx - top_idx) * 0.15))
    transition_end = min(transition_end, impact_idx - 1, n - 1)  # 确保不超过 impact

    if transition_end > top_idx:
        phases.append(
            SwingPhase(
                name="Transition",
                name_cn="转换",
                start_idx=top_idx,
                end_idx=transition_end,
                start_time_ms=time_ms[top_idx],
                end_time_ms=time_ms[transition_end],
                duration_ms=time_ms[transition_end] - time_ms[top_idx],
                peak_gyro_dps=gyro_mag[top_idx : transition_end + 1].max(),
            )
        )

    # Downswing: Transition 到 Impact
    if transition_end < impact_idx:
        phases.append(
            SwingPhase(
                name="Downswing",
                name_cn="下杆",
                start_idx=transition_end,
                end_idx=impact_idx,
                start_time_ms=time_ms[transition_end],
                end_time_ms=time_ms[impact_idx],
                duration_ms=time_ms[impact_idx] - time_ms[transition_end],
                peak_gyro_dps=gyro_mag[transition_end : impact_idx + 1].max(),
            )
        )

    # Impact: 瞬时点
    phases.append(
        SwingPhase(
            name="Impact",
            name_cn="击球",
            start_idx=impact_idx,
            end_idx=impact_idx + 1,
            start_time_ms=time_ms[impact_idx],
            end_time_ms=time_ms[min(impact_idx + 1, n - 1)],
            duration_ms=0,  # 瞬时
            peak_gyro_dps=peak_velocity,
        )
    )

    # Follow-through: Impact 之后
    follow_end = min(n - 1, impact_idx + int((n - impact_idx) * 0.8))
    if impact_idx < follow_end:
        phases.append(
            SwingPhase(
                name="Follow-through",
                name_cn="收杆",
                start_idx=impact_idx,
                end_idx=follow_end,
                start_time_ms=time_ms[impact_idx],
                end_time_ms=time_ms[follow_end],
                duration_ms=time_ms[follow_end] - time_ms[impact_idx],
                peak_gyro_dps=gyro_mag[impact_idx:follow_end].max(),
            )
        )

    print(f"✅ 阶段检测完成: {len(phases)} 个阶段")
    return phases


# ============================================================
# 指标计算
# ============================================================


def calculate_wrist_release_point(df: pd.DataFrame, phases: List[SwingPhase]) -> Optional[float]:
    """
    计算手腕释放点 (Wrist Cock Release Point)

    手腕释放点是下杆过程中手腕开始释放的位置，通过检测角加速度的峰值来确定。
    职业选手通常在下杆完成 85-95% 时才释放手腕。

    Args:
        df: 数据 DataFrame (需要有 gyro_mag_dps 和 time_ms)
        phases: 阶段列表

    Returns:
        释放点位置 (下杆完成百分比)，如 85.0 表示在下杆 85% 处释放
    """
    # 获取下杆阶段
    downswing = next((p for p in phases if p.name == "Downswing"), None)
    if not downswing or downswing.duration_ms <= 0:
        return None

    # 获取下杆期间的数据
    mask = (df["time_ms"] >= downswing.start_time_ms) & (df["time_ms"] <= downswing.end_time_ms)
    downswing_data = df[mask].copy()

    if len(downswing_data) < 3:
        return None

    # 计算角加速度 (角速度的一阶导数)
    # 加速度 = Δ速度 / Δ时间
    time_diff = downswing_data["time_ms"].diff()
    velocity_diff = downswing_data["gyro_mag_dps"].diff()

    # 避免除以零
    time_diff = time_diff.replace(0, np.nan)
    downswing_data["gyro_accel"] = velocity_diff / time_diff * 1000  # 转换为 °/s²

    # 找到最大加速度点 (手腕释放点)
    max_accel_idx = downswing_data["gyro_accel"].idxmax()
    if pd.isna(max_accel_idx):
        return None

    release_time = df.loc[max_accel_idx, "time_ms"]

    # 计算释放点在下杆中的位置 (百分比)
    progress = (release_time - downswing.start_time_ms) / downswing.duration_ms * 100

    return round(progress, 1)


def calculate_acceleration_time(df: pd.DataFrame, phases: List[SwingPhase]) -> Optional[float]:
    """
    计算加速时段 (Acceleration Time)

    加速时段是从下杆开始加速到达到峰值速度(Impact)的时间。
    职业选手的加速时段通常在 230-280ms。

    Args:
        df: 数据 DataFrame
        phases: 阶段列表

    Returns:
        加速时段 (毫秒)
    """
    # 获取关键阶段
    transition = next((p for p in phases if p.name == "Transition"), None)
    downswing = next((p for p in phases if p.name == "Downswing"), None)
    impact = next((p for p in phases if p.name == "Impact"), None)

    if not impact:
        return None

    # 加速开始点: Transition 结束 或 Downswing 开始
    if downswing:
        accel_start_time = downswing.start_time_ms
    elif transition:
        accel_start_time = transition.end_time_ms
    else:
        return None

    # 加速结束点: Impact
    accel_end_time = impact.start_time_ms

    acceleration_time = accel_end_time - accel_start_time

    return round(acceleration_time, 1) if acceleration_time > 0 else None


def calculate_metrics(df: pd.DataFrame, phases: List[SwingPhase]) -> SwingMetrics:
    """
    计算挥杆核心指标 (完整 7 项)

    Args:
        df: 数据 DataFrame
        phases: 检测到的阶段列表

    Returns:
        SwingMetrics 指标对象
    """
    # 获取关键阶段
    phase_dict = {p.name: p for p in phases}

    # 峰值角速度
    peak_velocity = df["gyro_mag_dps"].max()

    # 计算时间指标
    address_phase = phase_dict.get("Address")
    top_phase = phase_dict.get("Top")
    impact_phase = phase_dict.get("Impact")

    address_time = address_phase.end_time_ms if address_phase else 0
    top_time = top_phase.start_time_ms if top_phase else 0
    impact_time = impact_phase.start_time_ms if impact_phase else 0

    backswing_duration = top_time - address_time
    downswing_duration = impact_time - top_time
    total_swing_time = impact_time - address_time

    # 节奏比 (None if downswing_duration is 0 or negative)
    if downswing_duration <= 0:
        print(
            f"⚠️ 警告: 下杆时长无效 ({downswing_duration}ms)。" "阶段检测可能失败，节奏比无法计算。"
        )
        tempo_ratio: Optional[float] = None
    else:
        tempo_ratio = backswing_duration / downswing_duration

    # ============================================================
    # 新增指标计算
    # ============================================================
    wrist_release_point = calculate_wrist_release_point(df, phases)
    acceleration_time = calculate_acceleration_time(df, phases)

    # ============================================================
    # 水平评估 (基于 biomechanics-benchmarks)
    # ============================================================

    # 峰值速度评估
    if peak_velocity < 600:
        velocity_level = "初学者"
    elif peak_velocity < 1000:
        velocity_level = "业余"
    elif peak_velocity < 1500:
        velocity_level = "进阶"
    else:
        velocity_level = "职业"

    # 节奏比评估
    if tempo_ratio is None:
        tempo_level = "无数据"
    elif tempo_ratio < 2.0 or tempo_ratio > 5.0:
        tempo_level = "初学者"
    elif 2.0 <= tempo_ratio < 2.5:
        tempo_level = "业余"
    elif 2.5 <= tempo_ratio <= 3.5:
        tempo_level = "进阶/职业"
    else:
        tempo_level = "业余"

    # 手腕释放点评估
    if wrist_release_point is None:
        wrist_release_level = "无数据"
    elif wrist_release_point < 50:
        wrist_release_level = "过早 (<50%)"
    elif wrist_release_point < 70:
        wrist_release_level = "初学者"
    elif wrist_release_point < 85:
        wrist_release_level = "业余"
    else:
        wrist_release_level = "职业 (85-95%)"

    # 加速时段评估
    if acceleration_time is None:
        acceleration_level = "无数据"
    elif acceleration_time < 150:
        acceleration_level = "过短"
    elif acceleration_time < 200:
        acceleration_level = "进阶"
    elif acceleration_time <= 280:
        acceleration_level = "职业"
    elif acceleration_time <= 350:
        acceleration_level = "业余"
    else:
        acceleration_level = "初学者"

    # 综合评估
    levels = [velocity_level, tempo_level, wrist_release_level, acceleration_level]
    pro_keywords = ["职业", "进阶/职业", "职业 (85-95%)"]
    if any(kw in levels for kw in pro_keywords):
        overall_level = "进阶"
    elif "业余" in levels:
        overall_level = "业余"
    else:
        overall_level = "初学者"

    metrics = SwingMetrics(
        peak_angular_velocity_dps=round(peak_velocity, 1),
        backswing_duration_ms=round(backswing_duration, 1),
        downswing_duration_ms=round(downswing_duration, 1),
        total_swing_time_ms=round(total_swing_time, 1),
        tempo_ratio=round(tempo_ratio, 2) if tempo_ratio is not None else None,
        wrist_release_point_pct=wrist_release_point,
        acceleration_time_ms=acceleration_time,
        velocity_level=velocity_level,
        tempo_level=tempo_level,
        wrist_release_level=wrist_release_level,
        acceleration_level=acceleration_level,
        overall_level=overall_level,
    )

    return metrics


# ============================================================
# 可视化
# ============================================================


def plot_swing_analysis(
    df: pd.DataFrame,
    phases: List[SwingPhase],
    metrics: SwingMetrics,
    output_path: Optional[str] = None,
    show_plot: bool = True,
) -> None:
    """
    绘制挥杆分析图表

    Args:
        df: 数据 DataFrame
        phases: 阶段列表
        metrics: 指标对象
        output_path: 输出文件路径
        show_plot: 是否显示图表
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Movement Chain - IMU Swing Analysis (MVP)", fontsize=14, fontweight="bold")

    time_ms = df["time_ms"].values

    # 阶段颜色映射
    phase_colors = {
        "Address": "#E8E8E8",
        "Takeaway": "#FFE4B5",
        "Backswing": "#98FB98",
        "Top": "#FF6B6B",
        "Transition": "#FFA07A",
        "Downswing": "#87CEEB",
        "Impact": "#FF4500",
        "Follow-through": "#DDA0DD",
    }

    # ============================================================
    # 图1: 三轴角速度
    # ============================================================
    ax1 = axes[0]
    ax1.plot(time_ms, df["gyro_x_dps"], label="GyroX (俯仰)", alpha=0.8, linewidth=1)
    ax1.plot(time_ms, df["gyro_y_dps"], label="GyroY (偏航)", alpha=0.8, linewidth=1)
    ax1.plot(time_ms, df["gyro_z_dps"], label="GyroZ (翻滚)", alpha=0.8, linewidth=1)
    ax1.set_ylabel("Angular Velocity (°/s)")
    ax1.set_title("Three-Axis Angular Velocity")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

    # ============================================================
    # 图2: 合成角速度 + 阶段标注
    # ============================================================
    ax2 = axes[1]
    ax2.plot(time_ms, df["gyro_mag_dps"], "b-", linewidth=1.5, label="Magnitude")
    ax2.set_ylabel("Angular Velocity (°/s)")
    ax2.set_title("Composite Angular Velocity with Phase Detection")
    ax2.grid(True, alpha=0.3)

    # 添加阶段背景色
    for phase in phases:
        if phase.name in phase_colors:
            ax2.axvspan(
                phase.start_time_ms,
                (
                    phase.end_time_ms
                    if phase.end_time_ms > phase.start_time_ms
                    else phase.start_time_ms + 10
                ),
                alpha=0.3,
                color=phase_colors[phase.name],
                label=f"{phase.name} ({phase.name_cn})",
            )

    # 标记关键点
    phase_dict = {p.name: p for p in phases}
    if "Top" in phase_dict:
        top = phase_dict["Top"]
        ax2.axvline(x=top.start_time_ms, color="red", linestyle="--", linewidth=2)
        ax2.annotate(
            "Top\n顶点",
            xy=(top.start_time_ms, df["gyro_mag_dps"].max() * 0.3),
            fontsize=10,
            ha="center",
            color="red",
        )

    if "Impact" in phase_dict:
        impact = phase_dict["Impact"]
        ax2.axvline(x=impact.start_time_ms, color="orange", linestyle="--", linewidth=2)
        ax2.annotate(
            f"Impact\n击球\n{metrics.peak_angular_velocity_dps:.0f}°/s",
            xy=(impact.start_time_ms, metrics.peak_angular_velocity_dps),
            fontsize=10,
            ha="center",
            color="orange",
            xytext=(impact.start_time_ms + 50, metrics.peak_angular_velocity_dps * 0.9),
        )

    # 图例 (去重)
    handles, labels = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax2.legend(by_label.values(), by_label.keys(), loc="upper left", fontsize=8)

    # ============================================================
    # 图3: 指标汇总
    # ============================================================
    ax3 = axes[2]
    ax3.axis("off")

    # 创建指标表格
    metrics_text = f"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                              挥杆分析报告 - MVP 验证                               ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║  📊 核心指标                                                                      ║
║  ──────────────────────────────────────────────────────────────────────────────  ║
║  Peak Angular Velocity:  {metrics.peak_angular_velocity_dps:>7.0f}°/s    → {metrics.velocity_level:<8} (基准: 600-1500°/s)       ║
║  Backswing Duration:     {metrics.backswing_duration_ms:>7.0f} ms    → {'业余' if metrics.backswing_duration_ms > 850 else '进阶':<8} (基准: 700-850ms)        ║
║  Downswing Duration:     {metrics.downswing_duration_ms:>7.0f} ms    → {'进阶' if 230 <= metrics.downswing_duration_ms <= 300 else '待改进':<8} (基准: 230-300ms)        ║
║  Total Swing Time:       {metrics.total_swing_time_ms:>7.0f} ms    → (基准: 950-1100ms)                                ║
║  Tempo Ratio:            {metrics.tempo_ratio:>7.2f}       → {metrics.tempo_level:<8} (理想: 3:1)                    ║
║                                                                                  ║
║  🎯 综合评估: {metrics.overall_level}                                                             ║
║                                                                                  ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""
    ax3.text(
        0.05,
        0.5,
        metrics_text,
        transform=ax3.transAxes,
        fontsize=10,
        fontfamily="monospace",
        verticalalignment="center",
    )

    plt.tight_layout()

    if output_path:
        try:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"✅ 图表已保存: {output_path}")
        except PermissionError:
            print(f"❌ 错误: 无法写入 {output_path}，权限被拒绝")
            raise
        except OSError as e:
            print(f"❌ 错误: 无法保存图表到 {output_path}: {e}")
            raise

    if show_plot:
        plt.show()


# ============================================================
# 报告生成
# ============================================================


def generate_report(
    phases: List[SwingPhase],
    metrics: SwingMetrics,
    output_path: Optional[str] = None,
    isolation_info: Optional[Dict] = None,
) -> Dict:
    """
    生成 JSON 格式的分析报告

    Args:
        phases: 阶段列表
        metrics: 指标对象
        output_path: 输出文件路径
        isolation_info: 挥杆隔离信息 (可选)

    Returns:
        报告字典
    """
    report = {
        "analysis_time": datetime.now().isoformat(),
        "version": "MVP-2.0",
        "phases": [
            {
                "phase": p.name,
                "phase_cn": p.name_cn,
                "start_time_ms": round(p.start_time_ms, 1),
                "end_time_ms": round(p.end_time_ms, 1),
                "duration_ms": round(p.duration_ms, 1),
                "peak_gyro_dps": (
                    round(p.peak_gyro_dps, 1) if not np.isnan(p.peak_gyro_dps) else None
                ),
            }
            for p in phases
        ],
        "metrics": asdict(metrics),
        "benchmarks": {
            "peak_velocity": {
                "beginner": "<600°/s",
                "amateur": "600-1000°/s",
                "advanced": "1000-1500°/s",
                "professional": ">1500°/s",
            },
            "tempo_ratio": {"ideal": "3:1 (2.5-3.5)", "amateur": "2.0-2.5"},
            "backswing_duration": {
                "beginner": ">1000ms",
                "amateur": "850-1000ms",
                "advanced": "700-850ms",
                "professional": "700-800ms",
            },
            "downswing_duration": {
                "beginner": ">350ms",
                "amateur": "300-350ms",
                "advanced": "250-300ms",
                "professional": "230-280ms",
            },
            "total_swing_time": {
                "beginner": ">1400ms",
                "amateur": "1200-1400ms",
                "advanced": "1000-1200ms",
                "professional": "950-1100ms",
            },
            "wrist_release_point": {
                "early": "<50%",
                "beginner": "50-70%",
                "amateur": "70-85%",
                "professional": "85-95%",
            },
            "acceleration_time": {
                "beginner": ">350ms",
                "amateur": "280-350ms",
                "advanced": "200-280ms",
                "professional": "230-280ms",
            },
        },
    }

    # 添加隔离信息 (如果有)
    if isolation_info:
        report["isolation"] = isolation_info

    if output_path:
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"✅ 报告已保存: {output_path}")
        except PermissionError:
            print(f"❌ 错误: 无法写入 {output_path}，权限被拒绝")
            raise
        except OSError as e:
            print(f"❌ 错误: 无法保存报告到 {output_path}: {e}")
            raise

    return report


# ============================================================
# 主函数
# ============================================================


def analyze_swing(
    filepath: str,
    gyro_range: int = 2000,
    output_dir: Optional[str] = None,
    show_plot: bool = True,
    auto_isolate: bool = True,
    window_before_ms: float = 2000,
    window_after_ms: float = 1500,
) -> Tuple[pd.DataFrame, List[SwingPhase], SwingMetrics, Dict]:
    """
    执行完整的挥杆分析

    Args:
        filepath: IMU CSV 文件路径
        gyro_range: 陀螺仪量程 (250, 500, 1000, 2000)
        output_dir: 输出目录
        show_plot: 是否显示图表
        auto_isolate: 是否自动隔离单次挥杆 (默认 True)
        window_before_ms: Impact 前保留时间 (毫秒)
        window_after_ms: Impact 后保留时间 (毫秒)

    Returns:
        (处理后的数据, 阶段列表, 指标, 报告字典)
    """
    print("=" * 70)
    print("Movement Chain - IMU Swing Analyzer (v2.0)")
    print("=" * 70)
    print(f"输入文件: {filepath}")
    print(f"陀螺仪量程: ±{gyro_range}°/s")
    print(f"自动隔离: {'开启' if auto_isolate else '关闭'}")
    print()

    # 1. 加载数据
    df = load_imu_data(filepath)

    # 2. 单位转换
    df = convert_to_dps(df, gyro_range)

    # 3. 自动隔离单次挥杆 (新增)
    isolation_info = None
    if auto_isolate:
        df, isolation_info = isolate_swing(
            df, window_before_ms=window_before_ms, window_after_ms=window_after_ms
        )

    # 4. 阶段检测
    phases = detect_swing_phases(df)

    # 5. 指标计算
    metrics = calculate_metrics(df, phases)

    # 6. 输出
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 保存图表
        plot_file = output_path / "swing_analysis.png"
        plot_swing_analysis(df, phases, metrics, str(plot_file), show_plot=False)

        # 保存报告
        report_file = output_path / "swing_report.json"
        report = generate_report(phases, metrics, str(report_file), isolation_info)
    else:
        plot_swing_analysis(df, phases, metrics, show_plot=show_plot)
        report = generate_report(phases, metrics, isolation_info=isolation_info)

    # 打印摘要
    print()
    print("=" * 70)
    print("分析结果摘要 (完整 7 项 IMU 指标)")
    print("=" * 70)
    print(f"峰值角速度:   {metrics.peak_angular_velocity_dps:>7.0f}°/s  ({metrics.velocity_level})")
    print(
        f"上杆时长:     {metrics.backswing_duration_ms:>7.0f}ms   ({_evaluate_duration(metrics.backswing_duration_ms, 700, 850)})"
    )
    print(
        f"下杆时长:     {metrics.downswing_duration_ms:>7.0f}ms   ({_evaluate_duration(metrics.downswing_duration_ms, 230, 300)})"
    )
    print(
        f"总挥杆时间:   {metrics.total_swing_time_ms:>7.0f}ms   ({_evaluate_duration(metrics.total_swing_time_ms, 950, 1100)})"
    )
    print(f"节奏比:       {metrics.tempo_ratio:>7.2f}     ({metrics.tempo_level})")
    print(
        f"手腕释放点:   {_format_optional(metrics.wrist_release_point_pct, '%')}  ({metrics.wrist_release_level})"
    )
    print(
        f"加速时段:     {_format_optional(metrics.acceleration_time_ms, 'ms')}  ({metrics.acceleration_level})"
    )
    print("-" * 70)
    print(f"综合评估: {metrics.overall_level}")
    print("=" * 70)

    return df, phases, metrics, report


def _evaluate_duration(value: float, min_pro: float, max_pro: float) -> str:
    """评估时间指标"""
    if min_pro <= value <= max_pro:
        return "职业范围"
    elif value < min_pro:
        return "偏快"
    else:
        return "偏慢"


def _format_optional(value: Optional[float], unit: str) -> str:
    """格式化可选值"""
    if value is None:
        return "   N/A  "
    return f"{value:>7.1f}{unit}"


# ============================================================
# 入口点
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Movement Chain IMU Swing Analyzer v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python imu_swing_analyzer.py data.csv
  python imu_swing_analyzer.py data.csv --gyro-range 2000 --output-dir ./output
  python imu_swing_analyzer.py data.csv --no-isolate  # 不自动隔离
        """,
    )
    parser.add_argument("filepath", help="IMU CSV 文件路径")
    parser.add_argument(
        "--gyro-range",
        type=int,
        default=2000,
        choices=[250, 500, 1000, 2000],
        help="陀螺仪量程设置 (默认: 2000)",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录")
    parser.add_argument("--no-plot", action="store_true", help="不显示图表")
    parser.add_argument("--no-isolate", action="store_true", help="不自动隔离单次挥杆")
    parser.add_argument(
        "--window-before", type=float, default=2000, help="Impact 前保留时间 (ms, 默认: 2000)"
    )
    parser.add_argument(
        "--window-after", type=float, default=1500, help="Impact 后保留时间 (ms, 默认: 1500)"
    )

    args = parser.parse_args()

    analyze_swing(
        filepath=args.filepath,
        gyro_range=args.gyro_range,
        output_dir=args.output_dir,
        show_plot=not args.no_plot,
        auto_isolate=not args.no_isolate,
        window_before_ms=args.window_before,
        window_after_ms=args.window_after,
    )
