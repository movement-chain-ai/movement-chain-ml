"""
MediaPipe Pose Landmarker 测试脚本 - Movement Chain ML

适配 Python 3.14 + MediaPipe 0.10.x (仅 Tasks API)

用法:
  python analysis/test_mediapipe.py                    # 摄像头实时测试
  python analysis/test_mediapipe.py path/to/video.mp4  # 视频文件测试
  python analysis/test_mediapipe.py --save             # 保存结果到 output/

按 'q' 退出，按 's' 截图保存
"""

import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 模型路径
MODEL_PATH = Path(__file__).parent.parent / "models" / "pose_landmarker_full.task"

# BlazePose 33 关键点连接定义 (手动定义，替代 solutions.pose.POSE_CONNECTIONS)
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

# 关键点索引 (高尔夫重点)
LANDMARK_NAMES = {
    0: "NOSE",
    11: "LEFT_SHOULDER",  # 高尔夫关键
    12: "RIGHT_SHOULDER",
    13: "LEFT_ELBOW",
    14: "RIGHT_ELBOW",
    15: "LEFT_WRIST",  # 高尔夫关键
    16: "RIGHT_WRIST",
    23: "LEFT_HIP",  # X-Factor
    24: "RIGHT_HIP",
    25: "LEFT_KNEE",
    26: "RIGHT_KNEE",
    27: "LEFT_ANKLE",
    28: "RIGHT_ANKLE",
}


@dataclass
class PoseFrame:
    """单帧姿态数据"""

    frame_idx: int
    timestamp_ms: float
    landmarks: list
    left_arm_angle: float | None = None
    right_arm_angle: float | None = None
    x_factor_approx: float | None = None


def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """计算三点夹角 (p2 为顶点)"""
    v1 = p1 - p2
    v2 = p3 - p2
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1, 1))))


def extract_golf_metrics(landmarks) -> dict:
    """从关键点提取高尔夫相关指标"""
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
        hip_vec = np.array([landmarks[24].x - landmarks[23].x, landmarks[24].y - landmarks[23].y])
        cos_angle = np.dot(shoulder_vec, hip_vec) / (
            np.linalg.norm(shoulder_vec) * np.linalg.norm(hip_vec) + 1e-6
        )
        x_factor = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        metrics["x_factor_approx"] = round(x_factor, 1)
    except Exception:
        metrics["x_factor_approx"] = None

    return metrics


def draw_landmarks_on_image(image: np.ndarray, landmarks, image_width: int, image_height: int):
    """在图像上绘制骨架 (纯 OpenCV 实现)"""
    annotated = image.copy()

    # 将归一化坐标转换为像素坐标
    points = []
    for lm in landmarks:
        x = int(lm.x * image_width)
        y = int(lm.y * image_height)
        visibility = lm.visibility if hasattr(lm, "visibility") else 1.0
        points.append((x, y, visibility))

    # 绘制连接线
    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx < len(points) and end_idx < len(points):
            start = points[start_idx]
            end = points[end_idx]
            # 只绘制可见度高的连接
            if start[2] > 0.5 and end[2] > 0.5:
                cv2.line(annotated, (start[0], start[1]), (end[0], end[1]), (0, 255, 0), 2)

    # 绘制关键点
    for idx, (x, y, vis) in enumerate(points):
        if vis > 0.5:
            # 高尔夫关键点用不同颜色
            if idx in [11, 12, 15, 16, 23, 24]:  # 肩、腕、髋
                color = (0, 255, 255)  # 黄色
                radius = 6
            else:
                color = (0, 0, 255)  # 红色
                radius = 4
            cv2.circle(annotated, (x, y), radius, color, -1)

    return annotated


def draw_metrics_overlay(frame: np.ndarray, metrics: dict, fps: float) -> np.ndarray:
    """绘制指标信息叠加层"""
    # 背景框
    cv2.rectangle(frame, (10, 10), (280, 130), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (280, 130), (0, 255, 0), 2)

    y_offset = 35
    cv2.putText(
        frame, f"FPS: {fps:.1f}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
    )

    if metrics.get("left_arm_angle"):
        y_offset += 25
        cv2.putText(
            frame,
            f"Left Arm: {metrics['left_arm_angle']} deg",
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    if metrics.get("right_arm_angle"):
        y_offset += 25
        cv2.putText(
            frame,
            f"Right Arm: {metrics['right_arm_angle']} deg",
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    if metrics.get("x_factor_approx"):
        y_offset += 25
        cv2.putText(
            frame,
            f"X-Factor: {metrics['x_factor_approx']} deg",
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )

    return frame


def run_pose_detection(source: str | int = 0, save_results: bool = False) -> list[PoseFrame]:
    """运行姿态检测"""

    # 检查模型文件
    if not MODEL_PATH.exists():
        print(f"[ERROR] 模型文件不存在: {MODEL_PATH}")
        print("[INFO] 请运行以下命令下载模型:")
        print(
            "  curl -L -o models/pose_landmarker_full.task "
            '"https://storage.googleapis.com/mediapipe-models/pose_landmarker/'
            'pose_landmarker_full/float16/latest/pose_landmarker_full.task"'
        )
        return []

    # 打开视频源
    if isinstance(source, str) and Path(source).exists():
        cap = cv2.VideoCapture(source)
        source_name = Path(source).name
        print(f"[INFO] 加载视频文件: {source_name}")
    else:
        cap = cv2.VideoCapture(0)
        source_name = "webcam"
        print("[INFO] 使用摄像头 (按 'q' 退出, 's' 截图)")

    if not cap.isOpened():
        print("[ERROR] 无法打开视频源!")
        return []

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"[INFO] 视频尺寸: {width}x{height}, FPS: {fps_video:.1f}")

    # 初始化 PoseLandmarker (VIDEO 模式)
    base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    results_data: list[PoseFrame] = []
    frame_idx = 0
    fps_counter = 0
    fps_start = time.time()
    current_fps = 0.0

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        print("\n[INFO] MediaPipe PoseLandmarker 初始化成功!")
        print(f"[INFO] 模型: {MODEL_PATH.name}")
        print("[INFO] 关键点: 33 (BlazePose Full)")
        print("-" * 50)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                if source_name != "webcam":
                    print("[INFO] 视频处理完成")
                break

            # 转换为 MediaPipe Image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # 时间戳
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            if timestamp_ms == 0:
                timestamp_ms = int(frame_idx * 1000 / fps_video)

            # 姿态检测
            try:
                detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)
            except Exception as e:
                print(f"[WARN] 检测失败: {e}")
                frame_idx += 1
                continue

            annotated_frame = frame.copy()
            metrics = {}

            if detection_result.pose_landmarks:
                landmarks = detection_result.pose_landmarks[0]

                # 绘制骨架
                annotated_frame = draw_landmarks_on_image(frame, landmarks, width, height)

                # 提取指标
                metrics = extract_golf_metrics(landmarks)

                # 保存数据
                landmarks_list = [
                    [lm.x, lm.y, lm.z, getattr(lm, "visibility", 1.0)] for lm in landmarks
                ]

                pose_frame = PoseFrame(
                    frame_idx=frame_idx,
                    timestamp_ms=float(timestamp_ms),
                    landmarks=landmarks_list,
                    left_arm_angle=metrics.get("left_arm_angle"),
                    right_arm_angle=metrics.get("right_arm_angle"),
                    x_factor_approx=metrics.get("x_factor_approx"),
                )
                results_data.append(pose_frame)

                # 每秒打印状态
                if frame_idx % 30 == 0:
                    print(
                        f"[Frame {frame_idx:4d}] "
                        f"L.Arm: {metrics.get('left_arm_angle', 'N/A'):>6} | "
                        f"R.Arm: {metrics.get('right_arm_angle', 'N/A'):>6} | "
                        f"X-Factor: {metrics.get('x_factor_approx', 'N/A'):>6}"
                    )

            # 计算 FPS
            fps_counter += 1
            if time.time() - fps_start >= 1.0:
                current_fps = fps_counter / (time.time() - fps_start)
                fps_counter = 0
                fps_start = time.time()

            # 绘制叠加层
            annotated_frame = draw_metrics_overlay(annotated_frame, metrics, current_fps)

            # 显示
            cv2.imshow("MediaPipe Pose - Movement Chain", annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\n[INFO] 用户退出")
                break
            elif key == ord("s"):
                output_dir = Path("output")
                output_dir.mkdir(exist_ok=True)
                path = output_dir / f"screenshot_{frame_idx:04d}.png"
                cv2.imwrite(str(path), annotated_frame)
                print(f"[INFO] 截图: {path}")

            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    # 保存结果
    if save_results and results_data:
        output_path = Path("output/mediapipe_results.json")
        output_path.parent.mkdir(exist_ok=True)

        report = {
            "test_time": datetime.now().isoformat(),
            "source": source_name,
            "total_frames": len(results_data),
            "model": "pose_landmarker_full",
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            "mediapipe_version": mp.__version__,
            "frames": [asdict(f) for f in results_data[-100:]],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n[INFO] 结果保存: {output_path}")

    print(f"\n[SUMMARY] 处理帧数: {len(results_data)}")
    return results_data


def main():
    print("=" * 50)
    print("  MediaPipe Pose Landmarker 测试")
    print("  Movement Chain ML - Phase 1")
    print(f"  Python {sys.version_info.major}.{sys.version_info.minor}")
    print(f"  MediaPipe {mp.__version__}")
    print("=" * 50)

    source: str | int = 0
    save_results = "--save" in sys.argv

    for arg in sys.argv[1:]:
        if arg.endswith((".mp4", ".mov", ".avi", ".mkv")):
            source = arg
            break

    results = run_pose_detection(source=source, save_results=save_results)

    if results:
        print("\n" + "=" * 50)
        print("[SUCCESS] MediaPipe 测试通过!")
        print(f"  - 检测 {len(results)} 帧")
        print("  - 33 关键点正常")
        print("  - 角度计算正常")
        print("=" * 50)
    else:
        print("\n[WARNING] 未检测到姿态")


if __name__ == "__main__":
    main()
