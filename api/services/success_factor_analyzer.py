from typing import List, Dict, Optional, Any
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from datetime import datetime, timedelta
from data.models.funnel_stage import FunnelStage
from data.models.engagement_history import EngagementHistory
from data.models.student_profile import StudentProfile
import numpy as np
from scipy import stats
import logging

logger = logging.getLogger(__name__)

class SuccessFactorAnalyzer:
    def __init__(self, db: Session):
        self.db = db

    def get_success_factors_summary(self) -> Dict[str, Any]:
        """Get a summary of success factors across all stages."""
        stages = self.db.query(FunnelStage).order_by(FunnelStage.stage_order).all()
        summary = {
            "stages": [],
            "overall_success_rate": 0.0,
            "total_students": 0
        }

        total_successful = 0
        total_students = 0

        for stage in stages:
            stage_data = self.analyze_stage_success_factors(stage.id)
            summary["stages"].append(stage_data)
            total_successful += stage_data["successful_students"]
            total_students += stage_data["total_students"]

        if total_students > 0:
            summary["overall_success_rate"] = (total_successful / total_students) * 100
        summary["total_students"] = total_students

        return summary

    def analyze_stage_success_factors(self, stage_id: str) -> Dict[str, Any]:
        """Analyze success factors for a specific stage."""
        stage = self.db.query(FunnelStage).filter(FunnelStage.id == stage_id).first()
        if not stage:
            raise ValueError(f"Stage {stage_id} not found")

        # Get all students who reached this stage
        students = (
            self.db.query(StudentProfile)
            .filter(StudentProfile.funnel_stage == stage.stage_name)
            .all()
        )

        total_students = len(students)
        successful_students = sum(1 for s in students if s.is_successful)

        # Analyze engagement patterns
        engagement_metrics = self._analyze_engagement_patterns(stage_id)

        return {
            "stage_id": stage_id,
            "stage_name": stage.stage_name,
            "total_students": total_students,
            "successful_students": successful_students,
            "success_rate": (successful_students / total_students * 100) if total_students > 0 else 0,
            "engagement_metrics": engagement_metrics
        }

    def _analyze_engagement_patterns(self, stage_id: str) -> Dict[str, Any]:
        """Analyze engagement patterns for students in a specific stage."""
        # Get all engagements for students in this stage
        engagements = (
            self.db.query(EngagementHistory)
            .join(StudentProfile, StudentProfile.id == EngagementHistory.student_id)
            .filter(StudentProfile.current_stage_id == stage_id)
            .all()
        )

        # Calculate engagement metrics
        total_engagements = len(engagements)
        engagement_types = {}
        for engagement in engagements:
            engagement_type = engagement.engagement_type
            if engagement_type not in engagement_types:
                engagement_types[engagement_type] = 0
            engagement_types[engagement_type] += 1

        return {
            "total_engagements": total_engagements,
            "engagement_types": engagement_types
        }

    def analyze_success_factors(self, goal_stage: FunnelStage) -> Dict:
        """
        Perform comprehensive analysis of factors contributing to goal achievement.
        """
        try:
            # Get successful and unsuccessful students
            logger.info(f"Getting successful students for stage {goal_stage.stage_name}")
            successful_students = self._get_successful_students(goal_stage)
            logger.info(f"Found {len(successful_students)} successful students")
            
            logger.info(f"Getting unsuccessful students for stage {goal_stage.stage_name}")
            unsuccessful_students = self._get_unsuccessful_students(goal_stage)
            logger.info(f"Found {len(unsuccessful_students)} unsuccessful students")

            # Analyze various factors
            logger.info("Analyzing engagement patterns")
            engagement_analysis = self._analyze_engagement_patterns(
                successful_students, unsuccessful_students
            )
            logger.info(f"Found {len(engagement_analysis)} engagement factors")

            logger.info("Analyzing timing patterns")
            timing_analysis = self._analyze_timing_patterns(
                successful_students, unsuccessful_students
            )
            logger.info(f"Found {len(timing_analysis)} timing factors")

            logger.info("Analyzing stage transitions")
            stage_analysis = self._analyze_stage_transitions(
                successful_students, unsuccessful_students
            )
            logger.info(f"Found {len(stage_analysis)} stage factors")

            logger.info("Analyzing response patterns")
            response_analysis = self._analyze_response_patterns(
                successful_students, unsuccessful_students
            )
            logger.info(f"Found {len(response_analysis)} response factors")

            logger.info("Calculating overall impact")
            overall_impact = self._calculate_overall_impact(
                engagement_analysis,
                timing_analysis,
                stage_analysis,
                response_analysis
            )

            result = {
                "engagement_factors": engagement_analysis,
                "timing_factors": timing_analysis,
                "stage_factors": stage_analysis,
                "response_factors": response_analysis,
                "overall_impact": overall_impact
            }

            logger.info(f"Successfully completed analysis for stage {goal_stage.stage_name}")
            return result

        except Exception as e:
            logger.error(f"Error in analyze_success_factors for stage {goal_stage.stage_name}: {str(e)}", exc_info=True)
            raise

    def _get_successful_students(self, goal_stage: FunnelStage) -> List[StudentProfile]:
        """Get students who reached the goal stage."""
        return (
            self.db.query(StudentProfile)
            .filter(
                StudentProfile.current_stage_id == goal_stage.id,
                StudentProfile.is_successful == True
            )
            .all()
        )

    def _get_unsuccessful_students(self, goal_stage: FunnelStage) -> List[StudentProfile]:
        """Get students who haven't reached the goal stage."""
        return (
            self.db.query(StudentProfile)
            .filter(
                StudentProfile.current_stage_id == goal_stage.id,
                StudentProfile.is_successful == False
            )
            .all()
        )

    def _analyze_timing_patterns(
        self, successful: List[StudentProfile], unsuccessful: List[StudentProfile]
    ) -> List[Dict]:
        """Analyze timing patterns that contribute to success."""
        factors = []
        
        # Analyze time of day
        successful_times = self._get_engagement_times(successful)
        unsuccessful_times = self._get_engagement_times(unsuccessful)
        
        # Analyze day of week
        successful_days = self._get_engagement_days(successful)
        unsuccessful_days = self._get_engagement_days(unsuccessful)
        
        # Analyze time between engagements
        successful_intervals = self._get_engagement_intervals(successful)
        unsuccessful_intervals = self._get_engagement_intervals(unsuccessful)

        # Calculate impacts
        time_impact = self._calculate_time_impact(
            successful_times, unsuccessful_times
        )
        day_impact = self._calculate_day_impact(
            successful_days, unsuccessful_days
        )
        interval_impact = self._calculate_interval_impact(
            successful_intervals, unsuccessful_intervals
        )

        factors.extend([
            {
                "factor": "time_of_day",
                "impact": time_impact,
                "optimal_times": self._find_optimal_times(successful_times)
            },
            {
                "factor": "day_of_week",
                "impact": day_impact,
                "optimal_days": self._find_optimal_days(successful_days)
            },
            {
                "factor": "engagement_interval",
                "impact": interval_impact,
                "optimal_interval": self._find_optimal_interval(successful_intervals)
            }
        ])

        return sorted(factors, key=lambda x: x["impact"], reverse=True)

    def _analyze_stage_transitions(
        self, successful: List[StudentProfile], unsuccessful: List[StudentProfile]
    ) -> List[Dict]:
        """Analyze stage transition patterns that contribute to success."""
        factors = []
        
        # Get all stages
        stages = (
            self.db.query(FunnelStage)
            .filter(FunnelStage.is_active == True)
            .order_by(FunnelStage.stage_order)
            .all()
        )

        for i in range(len(stages) - 1):
            current_stage = stages[i]
            next_stage = stages[i + 1]

            # Calculate transition rates
            successful_rate = self._calculate_transition_rate(
                successful, current_stage, next_stage
            )
            unsuccessful_rate = self._calculate_transition_rate(
                unsuccessful, current_stage, next_stage
            )

            # Calculate time in stage
            successful_time = self._calculate_time_in_stage(
                successful, current_stage
            )
            unsuccessful_time = self._calculate_time_in_stage(
                unsuccessful, current_stage
            )

            # Calculate impact
            impact = self._calculate_transition_impact(
                successful_rate,
                unsuccessful_rate,
                successful_time,
                unsuccessful_time
            )

            if impact['significant']:
                factors.append({
                    "from_stage": current_stage.stage_name,
                    "to_stage": next_stage.stage_name,
                    "impact": impact,
                    "optimal_time": self._find_optimal_stage_time(successful_time)
                })

        return sorted(factors, key=lambda x: x["impact"]["value"], reverse=True)

    def _analyze_response_patterns(
        self, successful: List[StudentProfile], unsuccessful: List[StudentProfile]
    ) -> List[Dict]:
        """Analyze response patterns that contribute to success."""
        factors = []
        
        # Get all response types
        response_types = [
            "email_response",
            "call_response",
            "message_response",
            "application_response"
        ]

        for response_type in response_types:
            # Calculate response rates
            successful_rate = self._calculate_response_rate(
                successful, response_type
            )
            unsuccessful_rate = self._calculate_response_rate(
                unsuccessful, response_type
            )

            # Calculate response times
            successful_time = self._calculate_response_time(
                successful, response_type
            )
            unsuccessful_time = self._calculate_response_time(
                unsuccessful, response_type
            )

            # Calculate impact
            impact = self._calculate_response_impact(
                successful_rate,
                unsuccessful_rate,
                successful_time,
                unsuccessful_time
            )

            if impact['significant']:
                factors.append({
                    "response_type": response_type,
                    "impact": impact,
                    "optimal_time": self._find_optimal_response_time(successful_time)
                })

        return sorted(factors, key=lambda x: x["impact"]["value"], reverse=True)

    def _calculate_statistical_impact(
        self, successful_data: List[float], unsuccessful_data: List[float]
    ) -> Dict:
        """Calculate statistical impact between successful and unsuccessful groups."""
        if not successful_data or not unsuccessful_data:
            return {
                "significant": False,
                "impact": 0.0,
                "p_value": 1.0
            }

        # Perform t-test
        t_stat, p_value = stats.ttest_ind(successful_data, unsuccessful_data)
        
        # Calculate effect size (Cohen's d)
        mean_diff = np.mean(successful_data) - np.mean(unsuccessful_data)
        pooled_std = np.sqrt(
            (np.var(successful_data) + np.var(unsuccessful_data)) / 2
        )
        effect_size = mean_diff / pooled_std if pooled_std > 0 else 0

        return {
            "significant": p_value < 0.05,
            "impact": effect_size,
            "p_value": p_value
        }

    def _calculate_overall_impact(
        self,
        engagement_analysis: List[Dict],
        timing_analysis: List[Dict],
        stage_analysis: List[Dict],
        response_analysis: List[Dict]
    ) -> Dict:
        """Calculate overall impact across all factors."""
        total_impact = 0.0
        total_factors = 0

        for analysis in [engagement_analysis, timing_analysis, stage_analysis, response_analysis]:
            for factor in analysis:
                if factor.get("impact", {}).get("significant", False):
                    total_impact += abs(factor["impact"]["impact"])
                    total_factors += 1

        overall_impact = total_impact / total_factors if total_factors > 0 else 0.0

        return {
            "value": overall_impact,
            "total_factors": total_factors,
            "factor_weights": {
                "engagement": self._get_factor_weight("engagement"),
                "timing": self._get_factor_weight("timing"),
                "stage": self._get_factor_weight("stage"),
                "response": self._get_factor_weight("response")
            }
        }

    def _get_factor_weight(self, factor_type: str) -> float:
        """Get weight for a specific factor type."""
        weights = {
            "engagement": 0.4,
            "timing": 0.2,
            "stage": 0.2,
            "response": 0.2
        }
        return weights.get(factor_type, 0.0)

    def _calculate_engagement_frequency(
        self, students: List[StudentProfile], engagement_type: str
    ) -> List[float]:
        """Calculate engagement frequency for a group of students."""
        frequencies = []
        for student in students:
            count = (
                self.db.query(EngagementHistory)
                .filter(
                    EngagementHistory.student_id == student.id,
                    EngagementHistory.engagement_type == engagement_type
                )
                .count()
            )
            frequencies.append(count)
        return frequencies

    def _calculate_engagement_recency(
        self, students: List[StudentProfile], engagement_type: str
    ) -> List[float]:
        """Calculate engagement recency for a group of students."""
        recencies = []
        now = datetime.utcnow()
        for student in students:
            latest = (
                self.db.query(EngagementHistory)
                .filter(
                    EngagementHistory.student_id == student.id,
                    EngagementHistory.engagement_type == engagement_type
                )
                .order_by(EngagementHistory.created_at.desc())
                .first()
            )
            if latest:
                recency = (now - latest.created_at).total_seconds() / 3600  # hours
                recencies.append(recency)
        return recencies

    def _get_engagement_times(self, students: List[StudentProfile]) -> List[float]:
        """Get engagement times for a group of students."""
        times = []
        for student in students:
            engagements = (
                self.db.query(EngagementHistory)
                .filter(EngagementHistory.student_id == student.id)
                .all()
            )
            for engagement in engagements:
                hour = engagement.created_at.hour + engagement.created_at.minute / 60
                times.append(hour)
        return times

    def _get_engagement_days(self, students: List[StudentProfile]) -> List[int]:
        """Get engagement days for a group of students."""
        days = []
        for student in students:
            engagements = (
                self.db.query(EngagementHistory)
                .filter(EngagementHistory.student_id == student.id)
                .all()
            )
            for engagement in engagements:
                days.append(engagement.created_at.weekday())
        return days

    def _get_engagement_intervals(self, students: List[StudentProfile]) -> List[float]:
        """Get time intervals between engagements for a group of students."""
        intervals = []
        for student in students:
            engagements = (
                self.db.query(EngagementHistory)
                .filter(EngagementHistory.student_id == student.id)
                .order_by(EngagementHistory.created_at)
                .all()
            )
            for i in range(1, len(engagements)):
                interval = (engagements[i].created_at - engagements[i-1].created_at).total_seconds() / 3600
                intervals.append(interval)
        return intervals

    def _calculate_time_impact(
        self, successful_times: List[float], unsuccessful_times: List[float]
    ) -> Dict:
        """Calculate impact of engagement timing."""
        return self._calculate_statistical_impact(successful_times, unsuccessful_times)

    def _calculate_day_impact(
        self, successful_days: List[int], unsuccessful_days: List[int]
    ) -> Dict:
        """Calculate impact of engagement day."""
        return self._calculate_statistical_impact(successful_days, unsuccessful_days)

    def _calculate_interval_impact(
        self, successful_intervals: List[float], unsuccessful_intervals: List[float]
    ) -> Dict:
        """Calculate impact of engagement intervals."""
        return self._calculate_statistical_impact(successful_intervals, unsuccessful_intervals)

    def _find_optimal_times(self, times: List[float]) -> List[Dict]:
        """Find optimal times for engagement."""
        if not times:
            return []

        # Group times into 2-hour blocks
        blocks = {}
        for time in times:
            block = int(time / 2) * 2
            if block not in blocks:
                blocks[block] = 0
            blocks[block] += 1

        # Find top 3 blocks
        optimal_blocks = sorted(
            [{"hour": k, "count": v} for k, v in blocks.items()],
            key=lambda x: x["count"],
            reverse=True
        )[:3]

        return optimal_blocks

    def _find_optimal_days(self, days: List[int]) -> List[Dict]:
        """Find optimal days for engagement."""
        if not days:
            return []

        # Count occurrences of each day
        day_counts = {}
        for day in days:
            if day not in day_counts:
                day_counts[day] = 0
            day_counts[day] += 1

        # Find top 3 days
        optimal_days = sorted(
            [{"day": k, "count": v} for k, v in day_counts.items()],
            key=lambda x: x["count"],
            reverse=True
        )[:3]

        return optimal_days

    def _find_optimal_interval(self, intervals: List[float]) -> Dict:
        """Find optimal interval between engagements."""
        if not intervals:
            return {"value": 0, "confidence": 0}

        # Calculate mean and confidence interval
        mean = np.mean(intervals)
        std = np.std(intervals)
        confidence = 1.96 * std / np.sqrt(len(intervals))

        return {
            "value": mean,
            "confidence": confidence
        }

    def _calculate_transition_rate(
        self, students: List[StudentProfile], from_stage: FunnelStage, to_stage: FunnelStage
    ) -> float:
        """Calculate transition rate between stages."""
        if not students:
            return 0.0

        transitions = 0
        for student in students:
            # Check if student has moved from from_stage to to_stage
            history = (
                self.db.query(EngagementHistory)
                .filter(
                    EngagementHistory.student_id == student.id,
                    EngagementHistory.stage_id == from_stage.id
                )
                .order_by(EngagementHistory.created_at)
                .all()
            )
            if history:
                transitions += 1

        return transitions / len(students)

    def _calculate_time_in_stage(
        self, students: List[StudentProfile], stage: FunnelStage
    ) -> List[float]:
        """Calculate time spent in a stage for each student."""
        times = []
        for student in students:
            # Get first and last engagement in stage
            first = (
                self.db.query(EngagementHistory)
                .filter(
                    EngagementHistory.student_id == student.id,
                    EngagementHistory.stage_id == stage.id
                )
                .order_by(EngagementHistory.created_at)
                .first()
            )
            last = (
                self.db.query(EngagementHistory)
                .filter(
                    EngagementHistory.student_id == student.id,
                    EngagementHistory.stage_id == stage.id
                )
                .order_by(EngagementHistory.created_at.desc())
                .first()
            )
            if first and last:
                time = (last.created_at - first.created_at).total_seconds() / 3600
                times.append(time)
        return times

    def _calculate_transition_impact(
        self,
        successful_rate: float,
        unsuccessful_rate: float,
        successful_time: List[float],
        unsuccessful_time: List[float]
    ) -> Dict:
        """Calculate impact of stage transition."""
        rate_impact = abs(successful_rate - unsuccessful_rate)
        time_impact = self._calculate_statistical_impact(successful_time, unsuccessful_time)

        return {
            "significant": rate_impact > 0.1 or time_impact["significant"],
            "value": (rate_impact + time_impact["impact"]) / 2,
            "rate_impact": rate_impact,
            "time_impact": time_impact
        }

    def _find_optimal_stage_time(self, times: List[float]) -> Dict:
        """Find optimal time to spend in a stage."""
        if not times:
            return {"value": 0, "confidence": 0}

        mean = np.mean(times)
        std = np.std(times)
        confidence = 1.96 * std / np.sqrt(len(times))

        return {
            "value": mean,
            "confidence": confidence
        }

    def _calculate_response_rate(
        self, students: List[StudentProfile], response_type: str
    ) -> float:
        """Calculate response rate for a group of students."""
        if not students:
            return 0.0

        responses = 0
        for student in students:
            # Count responses of the specified type
            count = (
                self.db.query(EngagementHistory)
                .filter(
                    EngagementHistory.student_id == student.id,
                    EngagementHistory.engagement_type == response_type
                )
                .count()
            )
            responses += count

        return responses / len(students)

    def _calculate_response_time(
        self, students: List[StudentProfile], response_type: str
    ) -> List[float]:
        """Calculate response times for a group of students."""
        times = []
        for student in students:
            # Get response times
            responses = (
                self.db.query(EngagementHistory)
                .filter(
                    EngagementHistory.student_id == student.id,
                    EngagementHistory.engagement_type == response_type
                )
                .order_by(EngagementHistory.created_at)
                .all()
            )
            for i in range(1, len(responses)):
                time = (responses[i].created_at - responses[i-1].created_at).total_seconds() / 3600
                times.append(time)
        return times

    def _calculate_response_impact(
        self,
        successful_rate: float,
        unsuccessful_rate: float,
        successful_time: List[float],
        unsuccessful_time: List[float]
    ) -> Dict:
        """Calculate impact of response patterns."""
        rate_impact = abs(successful_rate - unsuccessful_rate)
        time_impact = self._calculate_statistical_impact(successful_time, unsuccessful_time)

        return {
            "significant": rate_impact > 0.1 or time_impact["significant"],
            "value": (rate_impact + time_impact["impact"]) / 2,
            "rate_impact": rate_impact,
            "time_impact": time_impact
        }

    def _find_optimal_response_time(self, times: List[float]) -> Dict:
        """Find optimal response time."""
        if not times:
            return {"value": 0, "confidence": 0}

        mean = np.mean(times)
        std = np.std(times)
        confidence = 1.96 * std / np.sqrt(len(times))

        return {
            "value": mean,
            "confidence": confidence
        } 