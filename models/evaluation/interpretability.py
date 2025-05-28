import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import shap
import tensorflow as tf
from data.feature_engineering import AdvancedFeatureEngineering

class ModelInterpretability:
    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        
    def calculate_feature_importance(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate SHAP values for feature importance"""
        # Create explainer
        explainer = shap.DeepExplainer(self.model, data)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(data)
        
        return shap_values
    
    def generate_application_explanation(self, student_id: str, prediction: float) -> Dict:
        """Generate human-readable explanation for application prediction"""
        # Get student data
        student_data = self.data[self.data['student_id'] == student_id]
        
        # Get feature contributions
        shap_values = self.calculate_feature_importance(student_data)
        
        # Generate explanation
        explanation = {
            'prediction': prediction,
            'key_factors': [],
            'recommendations': []
        }
        
        # Add key factors
        for feature, value in zip(self.feature_names, shap_values):
            if abs(value) > 0.1:  # Significant contribution
                explanation['key_factors'].append({
                    'feature': feature,
                    'contribution': value,
                    'student_value': student_data[feature].iloc[0]
                })
        
        # Generate recommendations
        for factor in explanation['key_factors']:
            if factor['contribution'] < 0:  # Negative contribution
                explanation['recommendations'].append(
                    f"Improve {factor['feature']} from {factor['student_value']}"
                )
        
        return explanation

class ApplicationScoreCalculator:
    def __init__(self, model, feature_engineering: AdvancedFeatureEngineering):
        self.model = model
        self.feature_engineering = feature_engineering
        
    def calculate_application_score(self, student_id: str) -> Dict:
        """Calculate comprehensive application likelihood score"""
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
            'base_score': base_score,
            'final_score': final_score,
            'confidence': self.calculate_confidence(features),
            'key_factors': self.identify_key_factors(features)
        }
    
    def apply_business_rules(self, base_score: float, features: pd.Series) -> float:
        """Apply business rules to adjust the score"""
        # Example rules
        if features['engagement_velocity'] > 0.5:  # High engagement
            base_score *= 1.1
        if features['regression_count'] > 0:  # Stage regression
            base_score *= 0.9
        if features['academic_engagement_ratio'] > 0.7:  # High academic interest
            base_score *= 1.15
            
        return min(max(base_score, 0), 1)  # Ensure score is between 0 and 1
    
    def calculate_confidence(self, features: pd.Series) -> float:
        """Calculate confidence in the prediction"""
        # Factors affecting confidence
        confidence_factors = {
            'data_completeness': self.calculate_completeness(features),
            'engagement_recency': self.calculate_recency(features),
            'feature_stability': self.calculate_stability(features)
        }
        
        return np.mean(list(confidence_factors.values()))
    
    def calculate_completeness(self, features: pd.Series) -> float:
        """Calculate data completeness score"""
        # Count non-null features
        non_null = features.notna().sum()
        total = len(features)
        
        return non_null / total
    
    def calculate_recency(self, features: pd.Series) -> float:
        """Calculate engagement recency score"""
        if 'last_engagement_date' in features:
            days_since = (pd.Timestamp.now() - pd.to_datetime(features['last_engagement_date'])).days
            return max(0, 1 - (days_since / 365))  # Decay over a year
        return 0.5  # Default value
    
    def calculate_stability(self, features: pd.Series) -> float:
        """Calculate feature stability score"""
        if 'velocity_std' in features:
            return 1 / (1 + features['velocity_std'])  # Lower std = higher stability
        return 0.5  # Default value
    
    def identify_key_factors(self, features: pd.Series) -> Dict:
        """Identify key factors influencing the score"""
        return {
            'engagement_velocity': features['engagement_velocity'],
            'funnel_progression': features['stage_progression_speed'],
            'academic_interest': features['academic_interest_score'],
            'interaction_quality': features['avg_satisfaction_score']
        } 