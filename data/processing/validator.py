import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DataValidator:
    """Validates and cleans CRM data."""
    
    def __init__(self):
        self.required_student_columns = {
            'student_id', 'first_name', 'last_name', 'birthdate', 'email', 'phone',
            'location', 'intended_major', 'application_status', 'funnel_stage',
            'first_interaction_date', 'last_interaction_date', 'recruiter_id',
            'interaction_count', 'gpa', 'sat_score', 'act_score'
        }
        self.required_engagement_columns = {
            'engagement_id', 'student_id', 'engagement_type', 'timestamp',
            'engagement_response', 'funnel_stage_before', 'funnel_stage_after'
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
    
    def process_student_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process student data to ensure required fields and data types."""
        # Ensure all required columns exist
        missing_cols = self.required_student_columns - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns in student data: {missing_cols}")

        # Convert date fields to datetime
        date_fields = ['birthdate', 'first_interaction_date', 'last_interaction_date']
        for field in date_fields:
            if field in df.columns:
                df[field] = pd.to_datetime(df[field], errors='coerce').dt.normalize()
                df[field] = df[field].replace({pd.NaT: None})
        
        # Clean and validate data
        df = df.copy()
        
        # Convert birthdate to datetime
        df['birthdate'] = pd.to_datetime(df['birthdate'], errors='coerce')
        
        # Convert numeric fields
        numeric_fields = ['gpa', 'sat_score', 'act_score', 'interaction_count']
        for field in numeric_fields:
            df[field] = pd.to_numeric(df[field], errors='coerce')
        
        # Validate GPA range (0-4)
        df['gpa'] = df['gpa'].clip(0, 4)
        
        # Validate SAT score range (400-1600)
        df['sat_score'] = df['sat_score'].clip(400, 1600)
        
        # Validate ACT score range (1-36)
        df['act_score'] = df['act_score'].clip(1, 36)
        
        return df
    
    def process_engagement_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process and validate engagement data.
        """
        # Check for required columns
        missing_cols = self.required_engagement_columns - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns in engagement data: {missing_cols}")

        # Clean and validate data
        df = df.copy()
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Convert engagement metrics if present
        metric_fields = ['open_time', 'click_through', 'time_spent']
        for field in metric_fields:
            if field in df.columns:
                df[field] = pd.to_numeric(df[field], errors='coerce')
        
        return df
    
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

        # If 'age' is present, impute it; otherwise, skip
        if "age" in data.columns:
            data["age"] = data.groupby("intended_major")["age"].transform(
                lambda x: x.fillna(x.median())
            )
        # If 'age' is not present but 'birthdate' is, compute age
        elif "birthdate" in data.columns:
            from datetime import date
            def calculate_age(birthdate):
                if pd.isnull(birthdate):
                    return None
                if isinstance(birthdate, str):
                    try:
                        birthdate = pd.to_datetime(birthdate).date()
                    except Exception:
                        return None
                elif isinstance(birthdate, pd.Timestamp):
                    birthdate = birthdate.date()
                today = date.today()
                return today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))
            data["age"] = data["birthdate"].apply(calculate_age)
        # If neither, do nothing
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