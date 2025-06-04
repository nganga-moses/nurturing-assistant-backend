import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging
from dataclasses import dataclass
import json
import os
from data.processing.data_quality import DataValidator

logger = logging.getLogger(__name__)

@dataclass
class QualityMetrics:
    """Data quality metrics for a dataset."""
    missing_values: Dict[str, float]  # Column -> percentage missing
    data_type_errors: Dict[str, int]  # Column -> number of type errors
    range_violations: Dict[str, int]  # Column -> number of range violations
    imputation_quality: Dict[str, float]  # Column -> imputation quality score
    validation_quality: float  # Overall validation quality score

class DataQualityMonitor:
    """Monitors and tracks data quality metrics."""
    
    def __init__(self, metrics_dir: str = "data/quality_metrics"):
        self.metrics_dir = metrics_dir
        self.metrics_history = []
        os.makedirs(metrics_dir, exist_ok=True)
        self.validator = DataValidator()
    
    def calculate_metrics(self, data: pd.DataFrame, original_data: Optional[pd.DataFrame] = None) -> QualityMetrics:
        """
        Calculate quality metrics for a dataset.
        
        Args:
            data: Processed DataFrame
            original_data: Optional original DataFrame for comparison
            
        Returns:
            QualityMetrics object
        """
        metrics = QualityMetrics(
            missing_values=self._calculate_missing_values(data),
            data_type_errors=self._calculate_type_errors(data),
            range_violations=self._calculate_range_violations(data),
            imputation_quality=self._calculate_imputation_quality(data, original_data) if original_data is not None else {},
            validation_quality=self._calculate_validation_quality(data)
        )
        
        # Store metrics
        self.metrics_history.append({
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics.__dict__
        })
        
        return metrics
    
    def _calculate_missing_values(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate percentage of missing values per column (0-100)."""
        missing = self.validator._validate_missing(data)
        total = len(data)
        return {col: (count / total * 100 if total > 0 else 0.0) for col, count in missing.items()}
    
    def _calculate_type_errors(self, data: pd.DataFrame) -> Dict[str, int]:
        """Calculate type errors for each column."""
        return self.validator._validate_types(data)
    
    def _calculate_range_violations(self, data: pd.DataFrame) -> Dict[str, int]:
        """Calculate number of range violations per column."""
        return self.validator._validate_ranges(data)
    
    def _calculate_imputation_quality(self, data: pd.DataFrame, original_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate imputation quality for each column."""
        imputation_quality = {}
        
        # Define columns that can be imputed
        imputable_columns = ['gpa', 'sat_score', 'act_score']
        
        for column in imputable_columns:
            if column not in data.columns or column not in original_data.columns:
                imputation_quality[column] = 0.0
                continue
            
            # Count original missing values
            original_missing = original_data[column].isna().sum()
            if original_missing == 0:
                imputation_quality[column] = 1.0
                continue
            
            # Count values that were imputed
            imputed = data[column].notna().sum() - original_data[column].notna().sum()
            
            # Calculate quality as ratio of successfully imputed values
            imputation_quality[column] = imputed / original_missing if original_missing > 0 else 1.0
        
        return imputation_quality
    
    def _calculate_validation_quality(self, data: pd.DataFrame) -> float:
        """Calculate overall validation quality score."""
        validation_results = self.validator.validate_data(data)
        total_cells = len(data) * len(self.validator.expected_types)
        if total_cells == 0:
            return 0.0

        error_cells = (
            sum(validation_results['type_errors'].values()) +
            sum(validation_results['range_violations'].values()) +
            sum(validation_results['missing_values'].values())
        )
        return 1.0 - (error_cells / total_cells)
    
    def save_metrics(self):
        """Save metrics history to disk."""
        metrics_file = os.path.join(self.metrics_dir, "metrics_history.json")
        
        with open(metrics_file, "w") as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def load_metrics(self):
        """Load metrics history from disk."""
        metrics_file = os.path.join(self.metrics_dir, "metrics_history.json")
        
        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as f:
                self.metrics_history = json.load(f)
    
    def get_quality_trends(self) -> Dict[str, List[float]]:
        """Get quality trends over time."""
        trends = {
            "validation_quality": [],
            "completeness": [],
            "consistency": [],
            "uniqueness": []
        }
        
        for entry in self.metrics_history:
            metrics = entry["metrics"]
            trends["validation_quality"].append(metrics["validation_quality"])
            
            # Calculate completeness from missing values
            missing_avg = np.mean(list(metrics["missing_values"].values()))
            trends["completeness"].append(1 - missing_avg/100)
            
            # Calculate consistency from type and range errors
            total_errors = sum(metrics["data_type_errors"].values()) + sum(metrics["range_violations"].values())
            trends["consistency"].append(1 - total_errors/len(metrics["missing_values"]))
            
            # Calculate uniqueness (assuming it's part of validation quality)
            trends["uniqueness"].append(metrics["validation_quality"])
        
        return trends 
    
    def check_data_quality(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Check data quality and return quality metrics.
        
        Args:
            data: DataFrame to check
            
        Returns:
            Dictionary of quality metrics
        """
        metrics = self.calculate_metrics(data)
        
        return {
            'completeness': 1 - np.mean(list(metrics.missing_values.values())) / 100,
            'consistency': 1 - (sum(metrics.data_type_errors.values()) + sum(metrics.range_violations.values())) / len(data),
            'timeliness': metrics.validation_quality,
            'accuracy': np.mean(list(metrics.imputation_quality.values())) if metrics.imputation_quality else 1.0
        }
    
    def generate_metrics(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Generate comprehensive quality metrics report.
        
        Args:
            data: DataFrame to analyze
            
        Returns:
            Dictionary containing daily and weekly metrics
        """
        # Calculate daily metrics
        daily_metrics = {
            'completeness': self._calculate_completeness(data),
            'consistency': self._calculate_consistency(data),
            'timeliness': self._calculate_timeliness(data),
            'accuracy': self._calculate_accuracy(data)
        }
        
        # Calculate weekly metrics (using same data for now)
        weekly_metrics = daily_metrics.copy()
        
        # Generate data quality report
        data_quality_report = {
            'current_metrics': daily_metrics,
            'trends': {
                'completeness': self._calculate_trend('completeness'),
                'consistency': self._calculate_trend('consistency'),
                'timeliness': self._calculate_trend('timeliness'),
                'accuracy': self._calculate_trend('accuracy')
            },
            'recommendations': self._generate_recommendations(daily_metrics)
        }
        
        return {
            'daily_metrics': daily_metrics,
            'weekly_metrics': weekly_metrics,
            'data_quality_report': data_quality_report
        }
    
    def _calculate_trend(self, metric_name: str, window: int = 7) -> float:
        """Calculate trend for a metric."""
        if len(self.metrics_history) < 2:
            return 0.0

        recent_metrics = self.metrics_history[-window:]
        if len(recent_metrics) < 2:
            return 0.0

        values = [m[metric_name] for m in recent_metrics if metric_name in m]
        if len(values) < 2:
            return 0.0

        return (values[-1] - values[0]) / len(values)
    
    def _generate_recommendations(self, metrics: Dict[str, float]) -> List[str]:
        """Generate recommendations based on metrics."""
        recommendations = []
        threshold = 0.9
        
        if metrics['completeness'] < threshold:
            recommendations.append("Improve data completeness by implementing better data collection processes")
        if metrics['consistency'] < threshold:
            recommendations.append("Enhance data consistency by standardizing data formats and validation rules")
        if metrics['timeliness'] < threshold:
            recommendations.append("Optimize data timeliness by reducing processing delays")
        if metrics['accuracy'] < threshold:
            recommendations.append("Increase data accuracy by implementing better validation and verification")
        
        return recommendations 

    def _calculate_completeness(self, data: pd.DataFrame) -> float:
        """Calculate data completeness."""
        return self.validator._validate_completeness(data)

    def _calculate_consistency(self, data: pd.DataFrame) -> float:
        """Stub: Return 1.0 for consistency."""
        return 1.0

    def _calculate_timeliness(self, data: pd.DataFrame) -> float:
        """Stub: Return 1.0 for timeliness."""
        return 1.0

    def _calculate_accuracy(self, data: pd.DataFrame) -> float:
        """Calculate data accuracy based on imputation quality."""
        # Use imputation quality as a proxy for accuracy
        imputation_quality = self._calculate_imputation_quality(data, data)
        return np.mean(list(imputation_quality.values())) if imputation_quality else 1.0

    def get_quality_report(self) -> Dict[str, Any]:
        """Generate a comprehensive data quality report."""
        if not self.metrics_history:
            return {"error": "No metrics available"}

        latest_metrics = self.metrics_history[-1]
        trends = {
            'completeness': self._calculate_trend('completeness'),
            'validation_quality': self._calculate_trend('validation_quality')
        }

        return {
            'current_metrics': latest_metrics,
            'trends': trends,
            'recommendations': self._generate_recommendations(latest_metrics)
        } 