import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass
import json
import os

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
        """Calculate percentage of missing values per column."""
        return {
            col: (data[col].isna().sum() / len(data)) * 100
            for col in data.columns
        }
    
    def _calculate_type_errors(self, data: pd.DataFrame) -> Dict[str, int]:
        """Calculate number of type errors per column."""
        type_errors = {}
        
        for col in data.columns:
            if col in ["student_id", "engagement_id"]:
                # Check for string type
                type_errors[col] = sum(~data[col].astype(str).str.match(r'^[A-Za-z0-9_-]+$'))
            elif col in ["gpa", "sat_score", "act_score"]:
                # Check for numeric type and range
                type_errors[col] = sum(pd.to_numeric(data[col], errors='coerce').isna())
            elif "timestamp" in col.lower():
                # Check for datetime type
                type_errors[col] = sum(pd.to_datetime(data[col], errors='coerce').isna())
        
        return type_errors
    
    def _calculate_range_violations(self, data: pd.DataFrame) -> Dict[str, int]:
        """Calculate number of range violations per column."""
        range_violations = {}
        
        # Define valid ranges
        ranges = {
            "gpa": (0, 4.0),
            "sat_score": (400, 1600),
            "act_score": (1, 36),
            "age": (15, 30)
        }
        
        for col, (min_val, max_val) in ranges.items():
            if col in data.columns:
                range_violations[col] = sum(
                    (data[col] < min_val) | (data[col] > max_val)
                )
        
        return range_violations
    
    def _calculate_imputation_quality(self, processed_data: pd.DataFrame, original_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate quality of imputation by comparing with original data."""
        imputation_quality = {}
        
        for col in processed_data.columns:
            if col in original_data.columns:
                # Calculate how many values were imputed
                imputed_mask = original_data[col].isna() & ~processed_data[col].isna()
                if imputed_mask.any():
                    # Calculate quality based on distribution similarity
                    original_dist = original_data[~original_data[col].isna()][col].value_counts(normalize=True)
                    imputed_dist = processed_data[imputed_mask][col].value_counts(normalize=True)
                    
                    # Calculate distribution similarity
                    common_categories = set(original_dist.index) & set(imputed_dist.index)
                    if common_categories:
                        similarity = sum(
                            min(original_dist[cat], imputed_dist[cat])
                            for cat in common_categories
                        )
                        imputation_quality[col] = similarity
                    else:
                        imputation_quality[col] = 0.0
        
        return imputation_quality
    
    def _calculate_validation_quality(self, data: pd.DataFrame) -> float:
        """Calculate overall validation quality score."""
        # Calculate various quality factors
        completeness = 1 - sum(data.isna().sum()) / (data.shape[0] * data.shape[1])
        
        # Calculate consistency (e.g., funnel stage progression)
        if "funnel_stage_before" in data.columns and "funnel_stage_after" in data.columns:
            funnel_stages = ["Awareness", "Interest", "Consideration", "Decision", "Application"]
            stage_indices = {stage: i for i, stage in enumerate(funnel_stages)}
            
            valid_progressions = sum(
                stage_indices[after] >= stage_indices[before]
                for before, after in zip(data["funnel_stage_before"], data["funnel_stage_after"])
                if before in stage_indices and after in stage_indices
            )
            consistency = valid_progressions / len(data)
        else:
            consistency = 1.0
        
        # Calculate uniqueness
        uniqueness = 1 - (data.duplicated().sum() / len(data))
        
        # Combine factors with weights
        quality_score = (
            0.4 * completeness +
            0.4 * consistency +
            0.2 * uniqueness
        )
        
        return quality_score
    
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