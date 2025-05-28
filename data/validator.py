import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DataValidator:
    """Validates and cleans CRM data."""
    
    def __init__(self):
        self.required_columns = {
            "student_data": [
                "student_id",
                "location",
                "age",
                "intended_major",
                "gpa",
                "sat_score",
                "act_score"
            ],
            "engagement_data": [
                "engagement_id",
                "student_id",
                "engagement_type",
                "timestamp",
                "engagement_response",
                "funnel_stage_before",
                "funnel_stage_after"
            ]
        }
        
        self.data_types = {
            "student_id": "string",
            "age": "integer",
            "gpa": "float",
            "sat_score": "integer",
            "act_score": "integer",
            "timestamp": "datetime"
        }
        
        self.valid_funnel_stages = [
            "Awareness",
            "Interest",
            "Consideration",
            "Decision",
            "Application"
        ]
    
    def validate_and_clean(self, crm_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Validate and clean CRM data.
        
        Args:
            crm_data: Dictionary containing student and engagement DataFrames
            
        Returns:
            Dictionary of cleaned DataFrames
        """
        cleaned_data = {
            "students": self.process_student_data(crm_data["students"]),
            "engagements": self.process_engagement_data(crm_data["engagements"])
        }
        
        return cleaned_data
    
    def process_student_data(self, student_data: pd.DataFrame) -> pd.DataFrame:
        """Process and clean student data."""
        # Check required columns
        missing_cols = set(self.required_columns["student_data"]) - set(student_data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns in student data: {missing_cols}")
        
        # Remove duplicates
        student_data = student_data.drop_duplicates(subset=["student_id"])
        
        # Handle missing values
        student_data["gpa"] = student_data["gpa"].fillna(
            student_data.groupby("intended_major")["gpa"].transform("mean")
        )
        
        student_data["sat_score"] = student_data["sat_score"].fillna(
            student_data.groupby("intended_major")["sat_score"].transform("mean")
        )
        
        student_data["act_score"] = student_data["act_score"].fillna(
            student_data.groupby("intended_major")["act_score"].transform("mean")
        )
        
        # Validate data types
        for column, dtype in self.data_types.items():
            if column in student_data.columns:
                student_data[column] = self.convert_to_type(
                    student_data[column],
                    dtype
                )
        
        # Validate ranges
        student_data["gpa"] = student_data["gpa"].clip(0, 4.0)
        student_data["sat_score"] = student_data["sat_score"].clip(400, 1600)
        student_data["act_score"] = student_data["act_score"].clip(1, 36)
        
        return student_data
    
    def process_engagement_data(self, engagement_data: pd.DataFrame) -> pd.DataFrame:
        """Process and clean engagement data."""
        # Check required columns
        missing_cols = set(self.required_columns["engagement_data"]) - set(engagement_data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns in engagement data: {missing_cols}")
        
        # Remove invalid engagements
        engagement_data = engagement_data[
            engagement_data["student_id"].isin(self.valid_student_ids)
        ]
        
        # Handle timestamps
        engagement_data["timestamp"] = pd.to_datetime(
            engagement_data["timestamp"],
            errors="coerce"
        )
        
        # Remove future dates
        engagement_data = engagement_data[
            engagement_data["timestamp"] <= pd.Timestamp.now()
        ]
        
        # Validate funnel stages
        engagement_data["funnel_stage_before"] = engagement_data["funnel_stage_before"].apply(
            lambda x: x if x in self.valid_funnel_stages else "Awareness"
        )
        
        engagement_data["funnel_stage_after"] = engagement_data["funnel_stage_after"].apply(
            lambda x: x if x in self.valid_funnel_stages else "Awareness"
        )
        
        return engagement_data
    
    def convert_to_type(self, series: pd.Series, dtype: str) -> pd.Series:
        """Convert series to specified data type."""
        try:
            if dtype == "string":
                return series.astype(str)
            elif dtype == "integer":
                return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)
            elif dtype == "float":
                return pd.to_numeric(series, errors="coerce").fillna(0.0)
            elif dtype == "datetime":
                return pd.to_datetime(series, errors="coerce")
            else:
                return series
        except Exception as e:
            logger.warning(f"Error converting {series.name} to {dtype}: {str(e)}")
            return series

class DataImputation:
    """Handles missing data imputation."""
    
    def __init__(self):
        self.imputation_strategies = {
            "demographic": self.impute_demographic,
            "academic": self.impute_academic,
            "engagement": self.impute_engagement
        }
    
    def impute_demographic(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing demographic data."""
        # Location: Use most common location for similar students
        data["location"] = data.groupby("intended_major")["location"].transform(
            lambda x: x.fillna(x.mode()[0])
        )
        
        # Age: Use median age for similar students
        data["age"] = data.groupby("intended_major")["age"].transform(
            lambda x: x.fillna(x.median())
        )
        
        return data
    
    def impute_academic(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing academic data."""
        # GPA: Use weighted average based on other academic scores
        data["gpa"] = data.apply(
            lambda row: self.calculate_estimated_gpa(row)
            if pd.isna(row["gpa"]) else row["gpa"],
            axis=1
        )
        
        # SAT/ACT: Use conversion formulas if one is missing
        data["sat_score"] = data.apply(
            lambda row: self.convert_act_to_sat(row["act_score"])
            if pd.isna(row["sat_score"]) else row["sat_score"],
            axis=1
        )
        
        return data
    
    def impute_engagement(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing engagement data."""
        # Duration: Use median duration for similar engagements
        data["duration"] = data.groupby("engagement_type")["duration"].transform(
            lambda x: x.fillna(x.median())
        )
        
        # Response type: Use most common response for similar engagements
        data["response_type"] = data.groupby("engagement_type")["response_type"].transform(
            lambda x: x.fillna(x.mode()[0])
        )
        
        return data
    
    def calculate_estimated_gpa(self, row: pd.Series) -> float:
        """Calculate estimated GPA based on SAT/ACT scores."""
        if pd.notna(row["sat_score"]):
            # Simple linear model: GPA = (SAT - 400) / 300
            return min(4.0, max(0.0, (row["sat_score"] - 400) / 300))
        elif pd.notna(row["act_score"]):
            # Simple linear model: GPA = (ACT - 1) / 8.75
            return min(4.0, max(0.0, (row["act_score"] - 1) / 8.75))
        else:
            return 2.5  # Default value
    
    def convert_act_to_sat(self, act_score: float) -> int:
        """Convert ACT score to SAT score."""
        if pd.isna(act_score):
            return 1000  # Default value
        
        # Simple linear conversion
        sat_score = int(act_score * 100 + 400)
        return min(1600, max(400, sat_score)) 