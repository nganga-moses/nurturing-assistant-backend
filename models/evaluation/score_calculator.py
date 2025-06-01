import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from data.processing.feature_engineering import AdvancedFeatureEngineering

class ScoreCalculator:
    """Business rule-based score calculation and adjustment."""
    
    def __init__(self, model, feature_engineering: AdvancedFeatureEngineering):
        """
        Initialize the score calculator.
        
        Args:
            model: The trained model
            feature_engineering: Feature engineering instance
        """
        self.model = model
        self.feature_engineering = feature_engineering
    
    def calculate_application_score(self, student_id: str) -> Dict:
        """Calculate comprehensive application likelihood score."""
        # Get base features
        student_data = self.get_student_data(student_id)
        
        # Add engineered features
        engagement_sequence = self.feature_engineering.create_engagement_sequence_features()
        interaction_quality = self.feature_engineering.create_interaction_quality_features()
        academic_engagement = self.feature_engineering.create_academic_engagement_features()
        
        # Combine all features
        features = pd.concat([
            student_data,
            engagement_sequence.loc[student_id],
            interaction_quality.loc[student_id],
            academic_engagement.loc[student_id]
        ], axis=0)
        
        # Calculate base score from model
        base_score = self.model.predict(features)['likelihood_score']
        
        # Apply business rules and adjustments
        final_score = self.apply_business_rules(base_score, features)
        
        return {
            'student_id': student_id,
            'base_score': float(base_score),
            'final_score': float(final_score),
            'confidence': self.calculate_confidence(features),
            'key_factors': self.identify_key_factors(features)
        }
    
    def apply_business_rules(self, base_score: float, features: pd.Series) -> float:
        """Apply business rules to adjust the score."""
        # Example rules
        if features.get('engagement_velocity', 0) > 0.5:  # High engagement
            base_score *= 1.1
        if features.get('regression_count', 0) > 0:  # Stage regression
            base_score *= 0.9
        if features.get('academic_engagement_ratio', 0) > 0.7:  # High academic interest
            base_score *= 1.15
            
        return min(max(base_score, 0), 1)  # Ensure score is between 0 and 1
    
    def calculate_confidence(self, features: pd.Series) -> float:
        """Calculate confidence in the prediction."""
        # Factors affecting confidence
        confidence_factors = {
            'data_completeness': self.calculate_completeness(features),
            'engagement_recency': self.calculate_recency(features),
            'feature_stability': self.calculate_stability(features)
        }
        
        return float(np.mean(list(confidence_factors.values())))
    
    def calculate_completeness(self, features: pd.Series) -> float:
        """Calculate data completeness score."""
        # Count non-null features
        non_null = features.notna().sum()
        total = len(features)
        
        return float(non_null / total)
    
    def calculate_recency(self, features: pd.Series) -> float:
        """Calculate engagement recency score."""
        if 'last_engagement_date' in features:
            days_since = (pd.Timestamp.now() - pd.to_datetime(features['last_engagement_date'])).days
            return float(max(0, 1 - (days_since / 365)))  # Decay over a year
        return 0.5  # Default value
    
    def calculate_stability(self, features: pd.Series) -> float:
        """Calculate feature stability score."""
        if 'velocity_std' in features:
            return float(1 / (1 + features['velocity_std']))  # Lower std = higher stability
        return 0.5  # Default value
    
    def identify_key_factors(self, features: pd.Series) -> Dict:
        """Identify key factors influencing the score."""
        return {
            'engagement_velocity': float(features.get('engagement_velocity', 0)),
            'funnel_progression': float(features.get('stage_progression_speed', 0)),
            'academic_interest': float(features.get('academic_interest_score', 0)),
            'interaction_quality': float(features.get('avg_satisfaction_score', 0))
        }
    
    def get_student_data(self, student_id: str) -> pd.Series:
        """Get student data from the feature engineering instance."""
        return self.feature_engineering.get_student_features(student_id) 