import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class DataQualityMonitor:
    """Class for monitoring and ensuring data quality in the student engagement system."""
    
    def __init__(self):
        self.quality_metrics = {}
    
    def check_student_data(self, students_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check quality of student data.
        
        Args:
            students_df: DataFrame containing student data
            
        Returns:
            Dictionary containing quality metrics
        """
        metrics = {
            'total_students': len(students_df),
            'missing_values': students_df.isnull().sum().to_dict(),
            'duplicate_students': students_df['student_id'].duplicated().sum(),
            'valid_emails': students_df['email'].str.contains('@').sum() if 'email' in students_df.columns else 0,
            'valid_phone_numbers': students_df['phone'].str.match(r'^\+?1?\d{9,15}$').sum() if 'phone' in students_df.columns else 0
        }
        
        self.quality_metrics['students'] = metrics
        return metrics
    
    def check_engagement_data(self, engagements_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check quality of engagement data.
        
        Args:
            engagements_df: DataFrame containing engagement data
            
        Returns:
            Dictionary containing quality metrics
        """
        metrics = {
            'total_engagements': len(engagements_df),
            'missing_values': engagements_df.isnull().sum().to_dict(),
            'unique_students': engagements_df['student_id'].nunique(),
            'engagement_types': engagements_df['engagement_type'].value_counts().to_dict(),
            'date_range': {
                'start': engagements_df['timestamp'].min(),
                'end': engagements_df['timestamp'].max()
            }
        }
        
        self.quality_metrics['engagements'] = metrics
        return metrics
    
    def check_data_consistency(self, students_df: pd.DataFrame, engagements_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check consistency between student and engagement data.
        
        Args:
            students_df: DataFrame containing student data
            engagements_df: DataFrame containing engagement data
            
        Returns:
            Dictionary containing consistency metrics
        """
        student_ids = set(students_df['student_id'])
        engagement_student_ids = set(engagements_df['student_id'])
        
        metrics = {
            'students_without_engagements': len(student_ids - engagement_student_ids),
            'engagements_without_students': len(engagement_student_ids - student_ids),
            'total_unique_students': len(student_ids | engagement_student_ids)
        }
        
        self.quality_metrics['consistency'] = metrics
        return metrics
    
    def get_quality_report(self) -> Dict[str, Any]:
        """
        Get a comprehensive quality report.
        
        Returns:
            Dictionary containing all quality metrics
        """
        return self.quality_metrics
    
    def log_quality_metrics(self):
        """Log quality metrics to the logger."""
        for category, metrics in self.quality_metrics.items():
            logger.info(f"\n{category.upper()} Quality Metrics:")
            for metric, value in metrics.items():
                logger.info(f"{metric}: {value}") 