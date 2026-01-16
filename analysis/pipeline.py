#!/usr/bin/env python3
"""
Movement Chain - Fusion Pipeline (融合 Pipeline)

完整的 视频 + IMU → 融合 → Rerun → AI 反馈 流水线。

用法:
    # 基本用法
    python pipeline.py --video swing.mp4 --imu swing.csv

    # 指定输出目录
    python pipeline.py --video swing.mp4 --imu swing.csv --output output/session1

    # 启动 Rerun Viewer
    python pipeline.py --video swing.mp4 --imu swing.csv --spawn-viewer

    # 只生成 Rerun 文件，不启动 Viewer
    python pipeline.py --video swing.mp4 --imu swing.csv --no-spawn

Author: Movement Chain AI Team
Date: 2025-01-15
"""

import argparse
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

# 添加当前目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

from ai_coach import AICoach

# 导入 IMU 分析器
from imu_swing_analyzer import analyze_swing
from rerun_visualizer import RERUN_AVAILABLE, RerunVisualizer
from schemas import KinematicPrompt
from sensor_fusion import SensorFusion
from vision_analyzer import VisionAnalyzer


def run_pipeline(
    video_path: str,
    imu_path: str,
    output_dir: str = "output",
    spawn_viewer: bool = False,
    gyro_range: int = 2000,
    downsample_factor: int = 10,
) -> KinematicPrompt:
    """
    运行完整的融合 Pipeline

    Args:
        video_path: 视频文件路径
        imu_path: IMU CSV 文件路径
        output_dir: 输出目录
        spawn_viewer: 是否启动 Rerun Viewer
        gyro_range: IMU 陀螺仪量程 (250, 500, 1000, 2000)
        downsample_factor: Rerun 降采样因子

    Returns:
        KinematicPrompt 结构化 AI 输入
    """
    print("=" * 70)
    print("  Movement Chain - Fusion Pipeline")
    print("=" * 70)
    print()

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ========================================
    # Step 1: 视频分析 (MediaPipe)
    # ========================================
    print("[Step 1/5] 视频分析 (MediaPipe)")
    print("-" * 70)

    vision_analyzer = VisionAnalyzer()
    vision_result = vision_analyzer.analyze_video(video_path)

    print(f"  ✅ 检测到 {len(vision_result.frames)} 帧")
    print(f"  ✅ Impact 帧: {vision_result.impact_frame_idx}")
    print()

    # ========================================
    # Step 2: IMU 分析
    # ========================================
    print("[Step 2/5] IMU 分析")
    print("-" * 70)

    imu_df, imu_phases, imu_metrics, imu_report = analyze_swing(
        filepath=imu_path,
        gyro_range=gyro_range,
        output_dir=None,  # 不单独保存
        show_plot=False,
        auto_isolate=True,
    )

    # 转换为字典格式
    imu_phases_dict = [asdict(p) for p in imu_phases]
    imu_metrics_dict = asdict(imu_metrics)

    print(f"  ✅ 检测到 {len(imu_phases)} 个阶段")
    print(f"  ✅ 峰值角速度: {imu_metrics.peak_angular_velocity_dps:.0f}°/s")
    print()

    # ========================================
    # Step 3: 传感器融合
    # ========================================
    print("[Step 3/5] 传感器融合")
    print("-" * 70)

    fusion = SensorFusion()
    fused_data = fusion.fuse(
        vision_result=vision_result,
        imu_df=imu_df,
        imu_phases=imu_phases_dict,
        imu_metrics=imu_metrics_dict,
        imu_report=imu_report,
    )

    print(f"  ✅ 融合 {len(fused_data.frames)} 帧")
    print(f"  ✅ 时间偏移: {fused_data.alignment_offset_ms:.1f}ms")
    print()

    # ========================================
    # Step 4: Rerun 可视化
    # ========================================
    print("[Step 4/5] Rerun 可视化")
    print("-" * 70)

    rerun_file = None
    if RERUN_AVAILABLE:
        visualizer = RerunVisualizer()
        rerun_file = str(output_path / f"{session_id}_swing.rrd")

        visualizer.visualize_swing(
            fused_data=fused_data,
            output_path=rerun_file,
            spawn_viewer=spawn_viewer,
            downsample_factor=downsample_factor,
        )

        print(f"  ✅ Rerun 文件: {rerun_file}")
        if spawn_viewer:
            print("  ✅ Rerun Viewer 已启动")
    else:
        print("  ⚠️ Rerun SDK 未安装，跳过可视化")
        print("     运行: pip install rerun-sdk>=0.20.0")

    print()

    # ========================================
    # Step 5: 生成 AI 反馈
    # ========================================
    print("[Step 5/5] 生成 AI Kinematic Prompt")
    print("-" * 70)

    coach = AICoach()
    kinematic_prompt = coach.generate_kinematic_prompt(
        fused_data=fused_data,
        visualization_file=rerun_file,
    )

    # 保存 Kinematic Prompt
    prompt_file = output_path / f"{session_id}_kinematic_prompt.json"
    kinematic_prompt.save(str(prompt_file))

    print(f"  ✅ Kinematic Prompt: {prompt_file}")
    print()

    # ========================================
    # 输出总结
    # ========================================
    print("=" * 70)
    print("  Pipeline 完成!")
    print("=" * 70)
    print()
    print(f"  输出目录: {output_path}")
    print(f"  Session ID: {session_id}")
    print()
    print("  生成文件:")
    print(f"    - {prompt_file.name} (AI 输入)")
    if rerun_file:
        print(f"    - {Path(rerun_file).name} (Rerun 可视化)")
    print()
    print("  整体评估:")
    print(f"    - 水平: {kinematic_prompt.overall_level}")
    if kinematic_prompt.key_issues:
        print("    - 主要问题:")
        for issue in kinematic_prompt.key_issues[:3]:
            print(f"      • {issue}")
    print()
    print("  下一步:")
    if rerun_file:
        print(f"    1. 运行 'rerun {rerun_file}' 查看 3D 可视化")
    print(f"    2. 将 {prompt_file.name} 内容粘贴到 Claude/ChatGPT 获取详细反馈")
    print("=" * 70)

    return kinematic_prompt


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="Movement Chain Fusion Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python pipeline.py --video swing.mp4 --imu swing.csv
  python pipeline.py --video swing.mp4 --imu swing.csv --spawn-viewer
  python pipeline.py --video swing.mp4 --imu swing.csv --output output/my_session
        """,
    )

    parser.add_argument(
        "--video",
        "-v",
        required=True,
        help="视频文件路径 (.mp4, .mov, .avi)",
    )

    parser.add_argument(
        "--imu",
        "-i",
        required=True,
        help="IMU CSV 数据文件路径",
    )

    parser.add_argument(
        "--output",
        "-o",
        default="output",
        help="输出目录 (默认: output)",
    )

    parser.add_argument(
        "--spawn-viewer",
        action="store_true",
        help="启动 Rerun Viewer",
    )

    parser.add_argument(
        "--gyro-range",
        type=int,
        default=2000,
        choices=[250, 500, 1000, 2000],
        help="IMU 陀螺仪量程 (默认: 2000)",
    )

    parser.add_argument(
        "--downsample",
        type=int,
        default=10,
        help="Rerun 降采样因子 (默认: 10)",
    )

    args = parser.parse_args()

    # 检查文件存在
    if not Path(args.video).exists():
        print(f"[ERROR] 视频文件不存在: {args.video}")
        sys.exit(1)

    if not Path(args.imu).exists():
        print(f"[ERROR] IMU 文件不存在: {args.imu}")
        sys.exit(1)

    try:
        run_pipeline(
            video_path=args.video,
            imu_path=args.imu,
            output_dir=args.output,
            spawn_viewer=args.spawn_viewer,
            gyro_range=args.gyro_range,
            downsample_factor=args.downsample,
        )
    except Exception as e:
        print(f"\n[ERROR] Pipeline 失败: {e}")
        raise


if __name__ == "__main__":
    main()
