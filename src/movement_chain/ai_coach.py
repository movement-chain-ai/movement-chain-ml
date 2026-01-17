"""
Movement Chain - AI Coach (Kinematic Prompt Generator)

将融合后的挥杆分析数据转换为结构化的 Kinematic Prompt，
用于喂给 LLM 生成个性化反馈。

用法:
    from ai_coach import AICoach

    coach = AICoach()
    prompt = coach.generate_kinematic_prompt(fused_data)
    prompt.save("output/kinematic_prompt.json")

    # 获取 AI Prompt 文本
    print(prompt.ai_prompt)

Author: Movement Chain AI Team
Date: 2025-01-15
"""

from dataclasses import asdict
from typing import Any

from .schemas import (
    BENCHMARKS,
    DiagnosticRuleTriggers,
    FusedSwingData,
    Issue,
    KinematicPrompt,
    KinematicPromptV2,
    PerPhaseMetrics,
    PhaseAnalysis,
    SensorAvailability,
)


class AICoach:
    """AI 教练 - 生成结构化反馈"""

    def __init__(self):
        """初始化 AI 教练"""
        self.benchmarks = BENCHMARKS

    def generate_kinematic_prompt(
        self,
        fused_data: FusedSwingData,
        visualization_file: str | None = None,
    ) -> KinematicPrompt:
        """
        生成 Kinematic Prompt

        Args:
            fused_data: 融合后的挥杆数据
            visualization_file: 可视化文件路径 (可选)

        Returns:
            KinematicPrompt 结构化 AI 输入
        """
        print("[AICoach] 生成 Kinematic Prompt...")

        # 1. 检测问题
        issues = self._analyze_issues(fused_data)

        # 2. 分析各阶段
        phase_analysis = self._analyze_phases(fused_data)

        # 3. 确定整体水平
        overall_level = self._determine_level(fused_data)

        # 4. 提取关键问题
        key_issues = self._extract_key_issues(issues)

        # 5. 生成 AI Prompt 文本
        ai_prompt = self._generate_prompt_text(fused_data, issues, phase_analysis, overall_level)

        prompt = KinematicPrompt(
            session_id=fused_data.session_id,
            analysis_time=fused_data.analysis_time,
            overall_level=overall_level,
            key_issues=key_issues,
            imu_metrics=fused_data.imu_metrics,
            vision_metrics=asdict(fused_data.vision_metrics),
            phase_analysis=phase_analysis,
            issues=issues,
            visualization_file=visualization_file,
            ai_prompt=ai_prompt,
        )

        print(f"[AICoach] 检测到 {len(issues)} 个问题")
        print(f"[AICoach] 整体水平: {overall_level}")

        return prompt

    def _analyze_issues(self, fused_data: FusedSwingData) -> list[Issue]:
        """分析指标，检测问题"""
        issues = []

        # IMU 指标分析
        imu = fused_data.imu_metrics

        # 峰值角速度
        peak_velocity = imu.get("peak_angular_velocity_dps")
        if peak_velocity is not None:
            issue = self._check_metric(
                metric_name="peak_angular_velocity_dps",
                value=peak_velocity,
                category="power",
                description_low="挥杆速度较慢，力量输出不足",
                description_high="挥杆速度过快，可能影响控制",
                suggestion_low="增加核心力量训练，注意髋部旋转发力",
                suggestion_high="适当减速，确保击球稳定性",
            )
            if issue:
                issues.append(issue)

        # 节奏比
        tempo_ratio = imu.get("tempo_ratio")
        if tempo_ratio is not None:
            benchmark = self.benchmarks.get("tempo_ratio", {})
            ideal = benchmark.get("ideal", (2.5, 3.5))

            if tempo_ratio < ideal[0]:
                issues.append(
                    Issue(
                        category="tempo",
                        severity="warning",
                        metric_name="tempo_ratio",
                        actual_value=tempo_ratio,
                        benchmark_range=f"{ideal[0]:.1f}-{ideal[1]:.1f}",
                        description="下杆过慢，节奏不理想",
                        suggestion="尝试加快下杆速度，保持上杆平稳",
                    )
                )
            elif tempo_ratio > ideal[1]:
                issues.append(
                    Issue(
                        category="tempo",
                        severity="warning",
                        metric_name="tempo_ratio",
                        actual_value=tempo_ratio,
                        benchmark_range=f"{ideal[0]:.1f}-{ideal[1]:.1f}",
                        description="上杆过快，节奏不理想",
                        suggestion="放慢上杆速度，理想的节奏比是 3:1",
                    )
                )

        # 上杆时间
        backswing = imu.get("backswing_duration_ms")
        if backswing is not None:
            issue = self._check_metric(
                metric_name="backswing_duration_ms",
                value=backswing,
                category="timing",
                description_low="上杆过快，可能影响蓄力",
                description_high="上杆过慢，浪费动能",
                suggestion_low="放慢上杆节奏，充分蓄力",
                suggestion_high="适当加快上杆，保持流畅",
            )
            if issue:
                issues.append(issue)

        # 下杆时间
        downswing = imu.get("downswing_duration_ms")
        if downswing is not None:
            issue = self._check_metric(
                metric_name="downswing_duration_ms",
                value=downswing,
                category="timing",
                description_low="下杆爆发力强",  # 实际上这可能是好事
                description_high="下杆过慢，缺乏爆发力",
                suggestion_low="保持当前的爆发力",
                suggestion_high="增加髋部和核心发力，提升下杆速度",
            )
            if issue:
                issues.append(issue)

        # Vision 指标分析
        vision = asdict(fused_data.vision_metrics)

        # 头部移动
        head_movement = vision.get("head_movement_cm")
        if head_movement is not None:
            benchmark = self.benchmarks.get("head_movement_cm", {})
            good = benchmark.get("good", (5, 10))

            if head_movement > good[1]:
                issues.append(
                    Issue(
                        category="stability",
                        severity="warning" if head_movement < 15 else "critical",
                        metric_name="head_movement_cm",
                        actual_value=head_movement,
                        benchmark_range=f"<{good[1]} cm",
                        description=f"头部移动过大 ({head_movement:.1f} cm)",
                        suggestion="保持头部稳定，眼睛始终盯着球",
                    )
                )

        # X-Factor
        x_factor = vision.get("x_factor_max_deg")
        if x_factor is not None:
            issue = self._check_metric(
                metric_name="x_factor_max_deg",
                value=x_factor,
                category="rotation",
                description_low="肩髋分离角度不足，旋转不够充分",
                description_high="肩髋分离角度过大，可能导致失控",
                suggestion_low="增加肩部旋转，保持髋部稳定以增大 X-Factor",
                suggestion_high="适当控制旋转幅度",
            )
            if issue:
                issues.append(issue)

        # 前臂伸展
        lead_arm = vision.get("lead_arm_extension_pct")
        if lead_arm is not None:
            benchmark = self.benchmarks.get("lead_arm_extension_pct", {})
            amateur = benchmark.get("amateur", (70, 85))

            if lead_arm < amateur[0]:
                issues.append(
                    Issue(
                        category="posture",
                        severity="warning",
                        metric_name="lead_arm_extension_pct",
                        actual_value=lead_arm,
                        benchmark_range=f">{amateur[0]}%",
                        description="前臂弯曲过多",
                        suggestion="在顶点保持前臂伸直，增加挥杆弧度",
                    )
                )

        return issues

    def _check_metric(
        self,
        metric_name: str,
        value: float,
        category: str,
        description_low: str,
        description_high: str,
        suggestion_low: str,
        suggestion_high: str,
    ) -> Issue | None:
        """检查单个指标是否在基准范围内"""
        benchmark = self.benchmarks.get(metric_name, {})

        if not benchmark:
            return None

        # 查找值所在的等级
        for level, (low, high) in benchmark.items():
            if low <= value <= high:
                # 在某个范围内
                if level in ["beginner", "poor"]:
                    return Issue(
                        category=category,
                        severity="warning",
                        metric_name=metric_name,
                        actual_value=value,
                        benchmark_range=f"{low}-{high}",
                        description=description_low,
                        suggestion=suggestion_low,
                    )
                return None

        # 不在任何范围内，检查是太低还是太高
        all_values = [v for level_range in benchmark.values() for v in level_range]
        min_val, max_val = min(all_values), max(all_values)

        if value < min_val:
            return Issue(
                category=category,
                severity="info",
                metric_name=metric_name,
                actual_value=value,
                benchmark_range=f">{min_val}",
                description=description_low,
                suggestion=suggestion_low,
            )
        elif value > max_val:
            return Issue(
                category=category,
                severity="info",
                metric_name=metric_name,
                actual_value=value,
                benchmark_range=f"<{max_val}",
                description=description_high,
                suggestion=suggestion_high,
            )

        return None

    def _analyze_phases(self, fused_data: FusedSwingData) -> list[PhaseAnalysis]:
        """分析各阶段表现"""
        phase_analysis = []

        for phase in fused_data.imu_phases:
            name = phase.get("name", "unknown")
            name_cn = phase.get("name_cn", name)
            duration = phase.get("duration_ms", 0)

            # 根据阶段检查问题
            issues = []
            benchmark_comp = "在正常范围内"

            if name.lower() == "backswing":
                benchmark = self.benchmarks.get("backswing_duration_ms", {})
                pro = benchmark.get("professional", (700, 800))
                if duration < pro[0]:
                    issues.append("上杆过快")
                    benchmark_comp = f"比职业选手快 {pro[0] - duration:.0f}ms"
                elif duration > pro[1]:
                    issues.append("上杆较慢")
                    benchmark_comp = f"比职业选手慢 {duration - pro[1]:.0f}ms"
                else:
                    benchmark_comp = "职业水平"

            elif name.lower() == "downswing":
                benchmark = self.benchmarks.get("downswing_duration_ms", {})
                pro = benchmark.get("professional", (230, 280))
                if duration < pro[0]:
                    benchmark_comp = f"比职业选手快 {pro[0] - duration:.0f}ms (爆发力强)"
                elif duration > pro[1]:
                    issues.append("下杆较慢")
                    benchmark_comp = f"比职业选手慢 {duration - pro[1]:.0f}ms"
                else:
                    benchmark_comp = "职业水平"

            phase_analysis.append(
                PhaseAnalysis(
                    phase=name,
                    phase_cn=name_cn,
                    duration_ms=duration,
                    issues=issues,
                    benchmark_comparison=benchmark_comp,
                )
            )

        return phase_analysis

    def _determine_level(self, fused_data: FusedSwingData) -> str:
        """确定整体水平"""
        imu = fused_data.imu_metrics

        # 使用峰值角速度作为主要指标
        peak_velocity = imu.get("peak_angular_velocity_dps", 0)

        benchmark = self.benchmarks.get("peak_angular_velocity_dps", {})

        for level in ["professional", "advanced", "amateur", "beginner"]:
            if level in benchmark:
                low, high = benchmark[level]
                if low <= peak_velocity <= high:
                    level_cn = {
                        "beginner": "初级",
                        "amateur": "中级",
                        "advanced": "高级",
                        "professional": "职业",
                    }
                    return level_cn.get(level, level)

        # 默认
        if peak_velocity > 1500:
            return "职业"
        elif peak_velocity > 1000:
            return "高级"
        elif peak_velocity > 600:
            return "中级"
        else:
            return "初级"

    def _extract_key_issues(self, issues: list[Issue]) -> list[str]:
        """提取关键问题列表"""
        # 按严重程度排序
        severity_order = {"critical": 0, "warning": 1, "info": 2}
        sorted_issues = sorted(issues, key=lambda x: severity_order.get(x.severity, 3))

        # 取前 3 个关键问题
        return [issue.description for issue in sorted_issues[:3]]

    def _generate_prompt_text(
        self,
        fused_data: FusedSwingData,
        issues: list[Issue],
        phase_analysis: list[PhaseAnalysis],
        overall_level: str,
    ) -> str:
        """生成 AI Prompt 文本"""
        imu = fused_data.imu_metrics
        vision = asdict(fused_data.vision_metrics)

        prompt = f"""# 高尔夫挥杆分析报告

## 基本信息
- 分析时间: {fused_data.analysis_time}
- 整体水平评估: {overall_level}

## IMU 传感器数据 (手部运动)
- 峰值角速度: {imu.get('peak_angular_velocity_dps', 'N/A')} °/s
- 上杆时长: {imu.get('backswing_duration_ms', 'N/A')} ms
- 下杆时长: {imu.get('downswing_duration_ms', 'N/A')} ms
- 总挥杆时间: {imu.get('total_swing_time_ms', 'N/A')} ms
- 节奏比 (上杆:下杆): {imu.get('tempo_ratio', 'N/A')}

## 视觉分析数据 (姿态)
- X-Factor 最大值: {vision.get('x_factor_max_deg', 'N/A')}°
- 头部移动距离: {vision.get('head_movement_cm', 'N/A')} cm
- 前臂伸展度: {vision.get('lead_arm_extension_pct', 'N/A')}%
- 脊柱倾斜角: {vision.get('spine_angle_deg', 'N/A')}°
- 髋部旋转角: {vision.get('hip_rotation_deg', 'N/A')}°

## 阶段分析
"""

        for pa in phase_analysis:
            prompt += f"### {pa.phase_cn} ({pa.phase})\n"
            prompt += f"- 时长: {pa.duration_ms:.0f} ms\n"
            prompt += f"- 对比基准: {pa.benchmark_comparison}\n"
            if pa.issues:
                prompt += f"- 问题: {', '.join(pa.issues)}\n"
            prompt += "\n"

        prompt += "## 检测到的问题\n"
        for issue in issues:
            severity_cn = {"critical": "严重", "warning": "警告", "info": "提示"}
            prompt += f"- [{severity_cn.get(issue.severity, issue.severity)}] {issue.description}\n"
            prompt += f"  建议: {issue.suggestion}\n"

        prompt += """
## 请根据以上数据提供:
1. 对整体挥杆技术的评价
2. 最需要优先改进的 2-3 个方面
3. 具体的练习建议和训练方法
4. 鼓励和正面反馈

请用专业但易懂的语言，像一位经验丰富的高尔夫教练一样给出建议。
"""

        return prompt


# ============================================================
# V2 AI Coach - 纯数据 + 布尔规则触发
# ============================================================


class AICoachV2:
    """V2 AI 教练 - 生成纯数据输出，无硬编码文本建议"""

    def __init__(self):
        """初始化 V2 AI 教练"""
        self.benchmarks = BENCHMARKS

    def generate_kinematic_prompt(
        self,
        fused_data: FusedSwingData,
        phase_metrics: list[PerPhaseMetrics] | None = None,
        visualization_file: str | None = None,
    ) -> KinematicPromptV2:
        """
        生成 V2 Kinematic Prompt (纯数据 + 布尔规则)

        Args:
            fused_data: 融合后的挥杆数据
            phase_metrics: 逐阶段指标 (可选，优先使用 fused_data.phase_metrics)
            visualization_file: 可视化文件路径 (可选)

        Returns:
            KinematicPromptV2 纯数据结构
        """
        print("[AICoachV2] 生成 Kinematic Prompt V2...")

        # 使用 fused_data 中的 phase_metrics，或传入的参数
        phases = phase_metrics or fused_data.phase_metrics or []

        # 1. 检测传感器可用性
        sensors = self._detect_sensor_availability(fused_data)

        # 2. 评估规则 (返回布尔触发器)
        rule_triggers = self._evaluate_rules(fused_data)

        # 3. 确定整体水平
        overall_level = self._determine_level(fused_data)

        # 4. 构建 KinematicPromptV2
        prompt = KinematicPromptV2(
            session_id=fused_data.session_id,
            analysis_time=fused_data.analysis_time,
            schema_version="2.0",
            sensors=sensors,
            overall_level=overall_level,
            imu_global_metrics=fused_data.imu_metrics,
            vision_global_metrics=asdict(fused_data.vision_metrics),
            phases=phases,
            rule_triggers=rule_triggers,
            visualization_file=visualization_file,
        )

        print(f"[AICoachV2] 传感器: Vision={sensors.vision}, IMU={sensors.imu}")
        print(f"[AICoachV2] 规则触发: {rule_triggers.rules_triggered}/{rule_triggers.rules_evaluated}")
        print(f"[AICoachV2] 整体水平: {overall_level}")

        return prompt

    def _detect_sensor_availability(
        self,
        fused_data: FusedSwingData,
    ) -> SensorAvailability:
        """检测各传感器数据可用性"""
        # 统计有 Vision 数据的帧
        vision_frames = sum(1 for f in fused_data.frames if f.has_vision)

        # 统计有 IMU 数据的帧
        imu_frames = sum(1 for f in fused_data.frames if f.gyro_magnitude is not None)

        return SensorAvailability(
            vision=vision_frames > 0,
            imu=imu_frames > 0,
            emg=False,  # EMG 预留
            vision_frame_count=vision_frames,
            imu_frame_count=imu_frames,
            emg_frame_count=0,
        )

    def _evaluate_rules(
        self,
        fused_data: FusedSwingData,
    ) -> DiagnosticRuleTriggers:
        """
        评估诊断规则，返回布尔触发器

        所有规则仅返回 True/False，不生成文本建议
        """
        triggers = DiagnosticRuleTriggers()
        rules_evaluated = 0
        rules_triggered = 0

        imu = fused_data.imu_metrics
        vision = asdict(fused_data.vision_metrics)

        # ========================================
        # P0 - 关键问题
        # ========================================

        # 节奏比检查 (理想范围 2.5-3.5)
        tempo_ratio = imu.get("tempo_ratio")
        if tempo_ratio is not None:
            rules_evaluated += 1
            if tempo_ratio < 2.0 or tempo_ratio > 4.0:
                triggers.tempo_ratio_outside_ideal = True
                rules_triggered += 1

        # 头部移动检查 (超过 10cm 为过量)
        head_movement = vision.get("head_movement_cm")
        if head_movement is not None:
            rules_evaluated += 1
            if head_movement > 10.0:
                triggers.head_movement_excessive = True
                rules_triggered += 1

        # X-Factor 检查 (低于 35 度为不足)
        x_factor = vision.get("x_factor_max_deg")
        if x_factor is not None:
            rules_evaluated += 1
            if x_factor < 35.0:
                triggers.x_factor_insufficient = True
                rules_triggered += 1

        # ========================================
        # P1 - 警告
        # ========================================

        # 上杆过快检查 (低于 700ms)
        backswing_duration = imu.get("backswing_duration_ms")
        if backswing_duration is not None:
            rules_evaluated += 1
            if backswing_duration < 700:
                triggers.backswing_too_fast = True
                rules_triggered += 1

        # 下杆过慢检查 (超过 350ms)
        downswing_duration = imu.get("downswing_duration_ms")
        if downswing_duration is not None:
            rules_evaluated += 1
            if downswing_duration > 350:
                triggers.downswing_too_slow = True
                rules_triggered += 1

        # 前臂弯曲检查 (低于 70%)
        lead_arm = vision.get("lead_arm_extension_pct")
        if lead_arm is not None:
            rules_evaluated += 1
            if lead_arm < 70.0:
                triggers.lead_arm_bent = True
                rules_triggered += 1

        # 手腕过早释放检查 (释放点低于 50%)
        wrist_release = imu.get("wrist_release_point")
        if wrist_release is not None:
            rules_evaluated += 1
            if wrist_release < 50.0:
                triggers.early_wrist_release = True
                rules_triggered += 1

        # ========================================
        # P2 - 信息
        # ========================================

        # 峰值速度低于业余水平 (低于 600 °/s)
        peak_velocity = imu.get("peak_angular_velocity_dps")
        if peak_velocity is not None:
            rules_evaluated += 1
            if peak_velocity < 600:
                triggers.velocity_below_amateur = True
                rules_triggered += 1

        # 更新统计
        triggers.rules_evaluated = rules_evaluated
        triggers.rules_triggered = rules_triggered

        return triggers

    def _determine_level(self, fused_data: FusedSwingData) -> str:
        """确定整体水平 (复用 AICoach 逻辑)"""
        imu = fused_data.imu_metrics

        # 使用峰值角速度作为主要指标
        peak_velocity = imu.get("peak_angular_velocity_dps", 0)

        benchmark = self.benchmarks.get("peak_angular_velocity_dps", {})

        for level in ["professional", "advanced", "amateur", "beginner"]:
            if level in benchmark:
                low, high = benchmark[level]
                if low <= peak_velocity <= high:
                    level_cn = {
                        "beginner": "初级",
                        "amateur": "中级",
                        "advanced": "高级",
                        "professional": "职业",
                    }
                    return level_cn.get(level, level)

        # 默认
        if peak_velocity > 1500:
            return "职业"
        elif peak_velocity > 1000:
            return "高级"
        elif peak_velocity > 600:
            return "中级"
        else:
            return "初级"


def main():
    """测试入口"""
    print("=" * 60)
    print("  Movement Chain - AI Coach")
    print("=" * 60)
    print("\n此模块需要通过 pipeline.py 使用。")
    print("用法: python pipeline.py --video <video.mp4> --imu <imu.csv>")


if __name__ == "__main__":
    main()
