import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, date
import json

class AdvancedFeatureEngineering:
    def __init__(self, student_data: pd.DataFrame, engagement_data: pd.DataFrame):
        self.student_data = student_data.copy()
        self.engagement_data = engagement_data
        # Add age and age_range columns from birthdate
        if 'birthdate' in self.student_data.columns:
            self.student_data['age'] = self.student_data['birthdate'].apply(self._calculate_age)
            self.student_data['age_range'] = self.student_data['age'].apply(self._calculate_age_range)
        
    def _calculate_age(self, birthdate):
        if pd.isnull(birthdate):
            return np.nan
        if isinstance(birthdate, str):
            try:
                birthdate = pd.to_datetime(birthdate).date()
            except Exception:
                return np.nan
        elif isinstance(birthdate, pd.Timestamp):
            birthdate = birthdate.date()
        today = date.today()
        return today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))

    def _calculate_age_range(self, age):
        if pd.isnull(age):
            return None
        bins = [(0, 17), (18, 24), (25, 34), (35, 44), (45, 54), (55, 64), (65, 120)]
        for start, end in bins:
            if start <= age <= end:
                return f"{start}-{end}"
        return None

    def _extract_metrics(self, metrics_str):
        try:
            return json.loads(metrics_str)
        except:
            return {}
        
    def create_engagement_sequence_features(self) -> pd.DataFrame:
        """Analyze the sequence of engagements to predict application likelihood"""
        features = {}
        
        for student_id, engagements in self.engagement_data.groupby('student_id'):
            # Calculate engagement velocity
            engagement_dates = pd.to_datetime(engagements['timestamp'])
            time_diffs = engagement_dates.diff()
            velocity = 1 / time_diffs.dt.days  # Engagements per day
            
            # Calculate funnel progression
            funnel_stages = ['awareness', 'interest', 'consideration', 'decision']
            stage_indices = {stage: i for i, stage in enumerate(funnel_stages)}
            
            # Track stage transitions
            stage_transitions = []
            for stage in engagements['funnel_stage_before']:
                if stage.lower() in stage_indices:
                    stage_transitions.append(stage_indices[stage.lower()])
            
            # Calculate progression metrics
            features[student_id] = {
                'engagement_velocity': velocity.mean(),
                'velocity_std': velocity.std(),
                'max_stage_reached': max(stage_transitions) if stage_transitions else 0,
                'stage_progression_speed': len(stage_transitions) / len(engagements),
                'regression_count': sum(1 for i in range(1, len(stage_transitions)) 
                                     if stage_transitions[i] < stage_transitions[i-1])
            }
        
        return pd.DataFrame.from_dict(features, orient='index')

    def create_interaction_quality_features(self) -> pd.DataFrame:
        """Analyze the quality of student interactions"""
        features = {}
        
        for student_id, engagements in self.engagement_data.groupby('student_id'):
            # Extract metrics from JSON
            metrics_list = engagements['engagement_metrics'].apply(self._extract_metrics)
            
            # Calculate engagement quality metrics
            features[student_id] = {
                'avg_time_spent': np.mean([m.get('time_spent', 0) for m in metrics_list]),
                'avg_scroll_depth': np.mean([m.get('scroll_depth', 0) for m in metrics_list]),
                'avg_interaction_count': np.mean([m.get('interaction_count', 0) for m in metrics_list]),
                'completion_rate': np.mean([1 if m.get('form_completion', 0) > 0 or m.get('download_count', 0) > 0 else 0 
                                          for m in metrics_list])
            }
        
        return pd.DataFrame.from_dict(features, orient='index')

    def create_academic_engagement_features(self) -> pd.DataFrame:
        """Analyze academic-related engagement patterns"""
        features = {}
        
        for student_id, engagements in self.engagement_data.groupby('student_id'):
            # Calculate academic engagement metrics
            academic_engagements = engagements[engagements['engagement_type'].str.contains('academic')]
            
            features[student_id] = {
                'academic_engagement_ratio': len(academic_engagements) / len(engagements),
                'faculty_interaction_count': len(engagements[engagements['engagement_type'] == 'faculty_meeting']),
                'department_visit_count': len(engagements[engagements['engagement_type'] == 'department_visit']),
                'avg_academic_time_spent': np.mean([self._extract_metrics(m).get('time_spent', 0) 
                                                  for m in academic_engagements['engagement_metrics']])
            }
        
        return pd.DataFrame.from_dict(features, orient='index')
    
    def create_all_features(self) -> pd.DataFrame:
        """Create all engineered features"""
        sequence_features = self.create_engagement_sequence_features()
        quality_features = self.create_interaction_quality_features()
        academic_features = self.create_academic_engagement_features()
        
        # Combine all features
        all_features = pd.concat([
            sequence_features,
            quality_features,
            academic_features
        ], axis=1)
        
        # Fill missing values
        all_features = all_features.fillna(0)
        
        return all_features 