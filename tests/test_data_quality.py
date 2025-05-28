import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from ..data.quality_monitor import DataQualityMonitor, QualityMetrics

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        "student_id": ["S001", "S002", "S003", "S004", "S005"],
        "gpa": [3.5, 3.8, np.nan, 3.2, 3.9],
        "sat_score": [1200, 1350, 1100, np.nan, 1500],
        "act_score": [25, 28, 22, 24, np.nan],
        "age": [18, 19, 17, 20, 21],
        "funnel_stage_before": ["Awareness", "Interest", "Consideration", "Decision", "Application"],
        "funnel_stage_after": ["Interest", "Consideration", "Decision", "Application", "Enrolled"]
    })

@pytest.fixture
def original_data():
    """Create original data with missing values for imputation testing."""
    return pd.DataFrame({
        "student_id": ["S001", "S002", "S003", "S004", "S005"],
        "gpa": [3.5, 3.8, np.nan, 3.2, 3.9],
        "sat_score": [1200, np.nan, 1100, np.nan, 1500],
        "act_score": [25, 28, np.nan, 24, np.nan],
        "age": [18, 19, 17, 20, 21],
        "funnel_stage_before": ["Awareness", "Interest", "Consideration", "Decision", "Application"],
        "funnel_stage_after": ["Interest", "Consideration", "Decision", "Application", "Enrolled"]
    })

def test_calculate_missing_values(sample_data):
    """Test calculation of missing values."""
    monitor = DataQualityMonitor()
    missing_values = monitor._calculate_missing_values(sample_data)
    
    assert "gpa" in missing_values
    assert "sat_score" in missing_values
    assert "act_score" in missing_values
    
    # Check specific missing value percentages
    assert missing_values["gpa"] == 20.0  # 1 out of 5 values missing
    assert missing_values["sat_score"] == 20.0
    assert missing_values["act_score"] == 20.0

def test_calculate_type_errors(sample_data):
    """Test calculation of type errors."""
    monitor = DataQualityMonitor()
    type_errors = monitor._calculate_type_errors(sample_data)
    
    assert "student_id" in type_errors
    assert "gpa" in type_errors
    assert "sat_score" in type_errors
    assert "act_score" in type_errors
    
    # Check that there are no type errors in the sample data
    assert all(errors == 0 for errors in type_errors.values())

def test_calculate_range_violations(sample_data):
    """Test calculation of range violations."""
    monitor = DataQualityMonitor()
    range_violations = monitor._calculate_range_violations(sample_data)
    
    assert "gpa" in range_violations
    assert "sat_score" in range_violations
    assert "act_score" in range_violations
    assert "age" in range_violations
    
    # Check that there are no range violations in the sample data
    assert all(violations == 0 for violations in range_violations.values())

def test_calculate_imputation_quality(sample_data, original_data):
    """Test calculation of imputation quality."""
    monitor = DataQualityMonitor()
    imputation_quality = monitor._calculate_imputation_quality(sample_data, original_data)
    
    # Check that imputation quality is calculated for columns with imputed values
    assert "gpa" in imputation_quality
    assert "sat_score" in imputation_quality
    assert "act_score" in imputation_quality
    
    # Check that quality scores are between 0 and 1
    assert all(0 <= score <= 1 for score in imputation_quality.values())

def test_calculate_validation_quality(sample_data):
    """Test calculation of validation quality."""
    monitor = DataQualityMonitor()
    quality_score = monitor._calculate_validation_quality(sample_data)
    
    # Check that quality score is between 0 and 1
    assert 0 <= quality_score <= 1

def test_quality_trends():
    """Test quality trend calculation."""
    monitor = DataQualityMonitor()
    
    # Add some sample metrics
    monitor.metrics_history = [
        {
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "missing_values": {"gpa": 20.0, "sat_score": 20.0},
                "data_type_errors": {"gpa": 0, "sat_score": 0},
                "range_violations": {"gpa": 0, "sat_score": 0},
                "imputation_quality": {"gpa": 0.8, "sat_score": 0.7},
                "validation_quality": 0.85
            }
        }
    ]
    
    trends = monitor.get_quality_trends()
    
    assert "validation_quality" in trends
    assert "completeness" in trends
    assert "consistency" in trends
    assert "uniqueness" in trends
    
    # Check that all trends have the same length
    assert all(len(trend) == 1 for trend in trends.values())

def test_save_load_metrics(tmp_path):
    """Test saving and loading metrics."""
    monitor = DataQualityMonitor(metrics_dir=str(tmp_path))
    
    # Add some sample metrics
    monitor.metrics_history = [
        {
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "missing_values": {"gpa": 20.0},
                "data_type_errors": {"gpa": 0},
                "range_violations": {"gpa": 0},
                "imputation_quality": {"gpa": 0.8},
                "validation_quality": 0.85
            }
        }
    ]
    
    # Save metrics
    monitor.save_metrics()
    
    # Create new monitor and load metrics
    new_monitor = DataQualityMonitor(metrics_dir=str(tmp_path))
    new_monitor.load_metrics()
    
    # Check that metrics were loaded correctly
    assert len(new_monitor.metrics_history) == 1
    assert new_monitor.metrics_history[0]["metrics"]["validation_quality"] == 0.85 