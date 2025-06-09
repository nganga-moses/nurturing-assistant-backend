from typing import List, Dict, Optional, Any
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from datetime import datetime, timedelta
from data.models.funnel_stage import FunnelStage
from data.models.engagement_history import EngagementHistory
from data.models.student_profile import StudentProfile
from functools import lru_cache
import hashlib
import json
import logging

logger = logging.getLogger(__name__)

class GoalAnalyticsService:
    def __init__(self, db: Session):
        self.db = db
        self._cache = {}
        self._cache_ttl = 3600  # 1 hour in seconds

    def _get_cache_key(self, method: str, params: Dict) -> str:
        """Generate a cache key from method name and parameters."""
        key_data = f"{method}:{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_cached(self, method: str, params: Dict) -> Optional[Dict]:
        """Get cached data if available and not expired."""
        cache_key = self._get_cache_key(method, params)
        if cache_key in self._cache:
            timestamp, data = self._cache[cache_key]
            if (datetime.now() - timestamp).total_seconds() < self._cache_ttl:
                return data
            del self._cache[cache_key]
        return None

    def _set_cached(self, method: str, params: Dict, data: Dict) -> None:
        """Cache data with current timestamp."""
        cache_key = self._get_cache_key(method, params)
        self._cache[cache_key] = (datetime.now(), data)

    def get_goal_progress(self, student_id: str) -> Dict:
        """Calculate progress towards the tracking goal for a student."""
        # Try to get from cache
        cached_data = self._get_cached('get_goal_progress', {'student_id': student_id})
        if cached_data:
            return cached_data

        # Get student and their current stage
        student = self.db.query(StudentProfile).filter(StudentProfile.id == student_id).first()
        if not student:
            raise ValueError("Student not found")

        # Get all active stages for the university
        stages = (
            self.db.query(FunnelStage)
            .filter(
                FunnelStage.is_active == True
            )
            .order_by(FunnelStage.stage_order)
            .all()
        )

        # Find tracking goal stage
        goal_stage = next((stage for stage in stages if stage.is_tracking_goal), None)
        if not goal_stage:
            return {
                "currentStage": student.funnel_stage,
                "goalStage": None,
                "progressPercentage": 0,
                "stagesToGoal": 0,
                "estimatedTimeToGoal": None
            }

        # Calculate progress
        current_stage_index = next(
            (i for i, stage in enumerate(stages) if stage.stage_name.lower() == student.funnel_stage.lower()),
            -1
        )
        goal_stage_index = stages.index(goal_stage)
        
        if current_stage_index == -1:
            return {
                "currentStage": student.funnel_stage,
                "goalStage": goal_stage.stage_name,
                "progressPercentage": 0,
                "stagesToGoal": goal_stage_index + 1,
                "estimatedTimeToGoal": None
            }

        progress_percentage = (current_stage_index / goal_stage_index) * 100 if goal_stage_index > 0 else 100
        stages_to_goal = goal_stage_index - current_stage_index

        # Calculate estimated time to goal based on historical data
        avg_time_per_stage = self._calculate_average_time_per_stage(
            current_stage_index,
            goal_stage_index
        )
        estimated_time = avg_time_per_stage * stages_to_goal if avg_time_per_stage else None

        progress_data = {
            "currentStage": student.funnel_stage,
            "goalStage": goal_stage.stage_name,
            "progressPercentage": progress_percentage,
            "stagesToGoal": stages_to_goal,
            "estimatedTimeToGoal": estimated_time
        }
        
        # Cache the result
        self._set_cached('get_goal_progress', {'student_id': student_id}, progress_data)
        
        return progress_data

    def get_goal_analytics(self) -> Dict:
        """Calculate analytics for the tracking goal."""
        # Try to get from cache
        cached_data = self._get_cached('get_goal_analytics')
        if cached_data:
            return cached_data

        # Get all stages for the university
        stages = (
            self.db.query(FunnelStage)
            .filter(
                FunnelStage.is_active == True
            )
            .order_by(FunnelStage.stage_order)
            .all()
        )

        # Find tracking goal stage
        goal_stage = next((stage for stage in stages if stage.is_tracking_goal), None)
        if not goal_stage:
            return {
                "conversionRate": 0,
                "averageTimeToGoal": 0,
                "bottleneckStages": [],
                "successFactors": []
            }

        # Calculate conversion rates between stages
        conversion_rates = self._calculate_conversion_rates(stages)

        # Calculate average time to goal
        avg_time = self._calculate_average_time_to_goal(goal_stage)

        # Identify bottleneck stages
        bottlenecks = self._identify_bottleneck_stages(conversion_rates)

        # Analyze success factors
        success_factors = self._analyze_success_factors(goal_stage)

        analytics_data = {
            "conversionRate": conversion_rates.get(goal_stage.stage_name, 0),
            "averageTimeToGoal": avg_time,
            "bottleneckStages": bottlenecks,
            "successFactors": success_factors
        }
        
        # Cache the result
        self._set_cached('get_goal_analytics', analytics_data)
        
        return analytics_data

    def _calculate_average_time_per_stage(self,  start_index: int, end_index: int) -> Optional[float]:
        """Calculate average time spent in each stage based on historical data."""
        # Get all students who reached the goal stage
        students = (
            self.db.query(StudentProfile)
            .all()
        )

        total_days = 0
        count = 0

        for student in students:
            # Get stage transition history
            transitions = self._get_stage_transitions(student.id)
            if not transitions:
                continue

            # Calculate time spent in relevant stages
            for i in range(start_index, min(end_index, len(transitions) - 1)):
                if i + 1 < len(transitions):
                    days = (transitions[i + 1]['timestamp'] - transitions[i]['timestamp']).days
                    if days > 0:
                        total_days += days
                        count += 1

        return total_days / count if count > 0 else None

    def _calculate_conversion_rates(self, stages: List[FunnelStage]) -> Dict[str, float]:
        """Calculate conversion rates between stages."""
        rates = {}
        for i in range(len(stages) - 1):
            current_stage = stages[i]
            next_stage = stages[i + 1]

            # Count students in current stage
            current_count = (
                self.db.query(StudentProfile)
                .filter(StudentProfile.funnel_stage == current_stage.stage_name)
                .count()
            )

            # Count students who moved to next stage
            next_count = (
                self.db.query(StudentProfile)
                .filter(StudentProfile.funnel_stage == next_stage.stage_name)
                .count()
            )

            rate = (next_count / current_count) * 100 if current_count > 0 else 0
            rates[next_stage.stage_name] = rate

        return rates

    def _identify_bottleneck_stages(self, conversion_rates: Dict[str, float]) -> List[str]:
        """Identify stages with low conversion rates."""
        avg_rate = sum(conversion_rates.values()) / len(conversion_rates) if conversion_rates else 0
        return [
            stage for stage, rate in conversion_rates.items()
            if rate < avg_rate * 0.7  # 30% below average
        ]

    def _analyze_success_factors(self,goal_stage: FunnelStage) -> List[Dict]:
        """Analyze factors contributing to goal achievement."""
        # Get successful students (reached goal stage)
        successful_students = (
            self.db.query(StudentProfile)
            .filter(
                StudentProfile.funnel_stage == goal_stage.stage_name
            )
            .all()
        )

        # Get all engagement types from successful students
        engagement_types = (
            self.db.query(EngagementHistory.engagement_type)
            .filter(EngagementHistory.student_id.in_([s.id for s in successful_students]))
            .distinct()
            .all()
        )

        factors = []
        for engagement_type in engagement_types:
            # Calculate success rate for each engagement type
            total_students = (
                self.db.query(StudentProfile)
                   .count()
            )

            students_with_engagement = (
                self.db.query(StudentProfile)
                .join(EngagementHistory)
                .filter(
                    EngagementHistory.engagement_type == engagement_type[0]
                )
                .distinct()
                .count()
            )

            if students_with_engagement > 0:
                success_rate = (
                    self.db.query(StudentProfile)
                    .join(EngagementHistory)
                    .filter(
                         StudentProfile.funnel_stage == goal_stage.stage_name,
                        EngagementHistory.engagement_type == engagement_type[0]
                    )
                    .distinct()
                    .count()
                ) / students_with_engagement * 100

                factors.append({
                    "stage": goal_stage.stage_name,
                    "factors": [engagement_type[0]],
                    "impact": success_rate
                })

        return sorted(factors, key=lambda x: x["impact"], reverse=True)

    def _get_stage_transitions(self, student_id: str) -> List[Dict]:
        """Get historical stage transitions for a student."""
        # This would typically come from a stage transition history table
        # For now, we'll infer it from engagement history
        engagements = (
            self.db.query(EngagementHistory)
            .filter(EngagementHistory.student_id == student_id)
            .order_by(EngagementHistory.timestamp)
            .all()
        )

        transitions = []
        for engagement in engagements:
            if engagement.funnel_stage:
                transitions.append({
                    "stage": engagement.funnel_stage,
                    "timestamp": engagement.timestamp
                })

        return transitions

    @lru_cache(maxsize=100)
    def get_stage_transition_rates(self) -> Dict[str, float]:
        """Get cached stage transition rates."""
        stages = (
            self.db.query(FunnelStage)
            .filter(FunnelStage.is_active == True)
            .order_by(FunnelStage.stage_order)
            .all()
        )
        
        return self._calculate_conversion_rates(stages)

    def _calculate_average_time_to_goal(self, goal_stage: FunnelStage) -> float:
        """Calculate average time to goal."""
        # This method needs to be implemented
        raise NotImplementedError("Method _calculate_average_time_to_goal needs to be implemented")

    def clear_cache(self, method: Optional[str] = None) -> None:
        """Clear cache for a specific method or all methods."""
        if method:
            # Clear specific method cache
            keys_to_remove = [
                k for k in self._cache.keys()
                if k.startswith(self._get_cache_key(method, {}))
            ]
            for k in keys_to_remove:
                del self._cache[k]
        else:
            # Clear all cache
            self._cache.clear()
            self.get_stage_transition_rates.cache_clear()

    def get_goal_metrics(self) -> Dict[str, Any]:
        """Get goal metrics for all stages."""
        stages = self.db.query(FunnelStage).order_by(FunnelStage.stage_order).all()
        metrics = {
            "stages": [],
            "overall_conversion_rate": 0.0,
            "total_students": 0
        }

        total_converted = 0
        total_students = 0

        for stage in stages:
            stage_metrics = self.get_stage_metrics(stage.id)
            metrics["stages"].append(stage_metrics)
            total_converted += stage_metrics["converted_students"]
            total_students += stage_metrics["total_students"]

        if total_students > 0:
            metrics["overall_conversion_rate"] = (total_converted / total_students) * 100
        metrics["total_students"] = total_students

        return metrics

    def get_stage_metrics(self, stage_id: str) -> Dict[str, Any]:
        """Get goal metrics for a specific stage."""
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
        converted_students = sum(1 for s in students if s.is_successful)

        # Get engagement metrics
        engagement_metrics = self._get_engagement_metrics(stage_id)

        return {
            "stage_id": stage_id,
            "stage_name": stage.stage_name,
            "total_students": total_students,
            "converted_students": converted_students,
            "conversion_rate": (converted_students / total_students * 100) if total_students > 0 else 0,
            "engagement_metrics": engagement_metrics
        }

    def _get_engagement_metrics(self, stage_id: str) -> Dict[str, Any]:
        """Get engagement metrics for students in a specific stage."""
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