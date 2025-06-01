import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import tensorflow as tf
import shap
from data.processing.feature_engineering import AdvancedFeatureEngineering

class ModelEvaluator:
    """Comprehensive model evaluation including cross-validation and feature importance."""
    
    def __init__(self, model, data: pd.DataFrame, feature_names: List[str], n_splits: int = 5):
        """
        Initialize the model evaluator.
        
        Args:
            model: The model to evaluate
            data: DataFrame containing the data
            feature_names: List of feature names
            n_splits: Number of cross-validation splits
        """
        self.model = model
        self.data = data
        self.feature_names = feature_names
        self.n_splits = n_splits
        self.feature_engineering = AdvancedFeatureEngineering(
            student_data=data,
            engagement_data=data
        )
    
    def time_based_split(self) -> List[pd.DataFrame]:
        """Split data based on time to prevent data leakage."""
        # Sort by timestamp
        sorted_data = self.data.sort_values('timestamp')
        
        # Create time-based folds
        fold_size = len(sorted_data) // self.n_splits
        folds = []
        
        for i in range(self.n_splits):
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size if i < self.n_splits - 1 else len(sorted_data)
            folds.append(sorted_data.iloc[start_idx:end_idx])
        
        return folds
    
    def prepare_fold_data(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> Tuple[Dict, pd.Series]:
        """Prepare data for a single fold."""
        # Create engineered features
        train_features = self.feature_engineering.create_all_features()
        val_features = self.feature_engineering.create_all_features()
        
        # Prepare input data
        train_input = {
            'student_id': train_data['student_id'],
            'student_features': train_features,
            'engagement_id': train_data['engagement_id'],
            'engagement_features': train_data['engagement_features']
        }
        
        val_input = {
            'student_id': val_data['student_id'],
            'student_features': val_features,
            'engagement_id': val_data['engagement_id'],
            'engagement_features': val_data['engagement_features']
        }
        
        return train_input, val_input, val_data['applied']
    
    def train_with_cross_validation(self, epochs: int = 10, batch_size: int = 32) -> Dict:
        """Train model using time-based cross-validation."""
        folds = self.time_based_split()
        cv_scores = []
        
        for i in range(self.n_splits):
            print(f"\nTraining fold {i+1}/{self.n_splits}")
            
            # Use all folds except the current one for training
            train_data = pd.concat(folds[:i] + folds[i+1:])
            val_data = folds[i]
            
            # Prepare data
            train_input, val_input, val_labels = self.prepare_fold_data(train_data, val_data)
            
            # Create TensorFlow dataset
            train_dataset = tf.data.Dataset.from_tensor_slices((
                train_input,
                train_data['applied']
            )).shuffle(1000).batch(batch_size)
            
            val_dataset = tf.data.Dataset.from_tensor_slices((
                val_input,
                val_labels
            )).batch(batch_size)
            
            # Train model
            history = self.model.fit(
                train_dataset,
                epochs=epochs,
                validation_data=val_dataset,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=3,
                        restore_best_weights=True
                    )
                ]
            )
            
            # Evaluate on validation set
            val_scores = self.model.evaluate(val_dataset, return_dict=True)
            cv_scores.append(val_scores)
            
            print(f"Fold {i+1} scores:", val_scores)
        
        # Calculate average scores
        avg_scores = {}
        for metric in cv_scores[0].keys():
            avg_scores[metric] = np.mean([fold[metric] for fold in cv_scores])
        
        return {
            'fold_scores': cv_scores,
            'average_scores': avg_scores
        }
    
    def calculate_feature_importance(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate SHAP values for feature importance."""
        # Create explainer
        explainer = shap.DeepExplainer(self.model, data)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(data)
        
        return shap_values
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Calculate feature importance across folds."""
        importance_scores = []
        
        for i in range(self.n_splits):
            # Get fold data
            folds = self.time_based_split()
            train_data = pd.concat(folds[:i] + folds[i+1:])
            val_data = folds[i]
            
            # Get features
            features = self.feature_engineering.create_all_features()
            
            # Calculate SHAP values for this fold
            shap_values = self.calculate_feature_importance(features)
            
            # Convert to DataFrame
            fold_importance = pd.Series(
                np.abs(shap_values).mean(axis=0),
                index=self.feature_names
            )
            
            importance_scores.append(fold_importance)
        
        # Average importance across folds
        avg_importance = pd.concat(importance_scores).groupby(level=0).mean()
        
        return avg_importance.sort_values(ascending=False)
    
    def generate_explanation(self, student_id: str, prediction: float) -> Dict:
        """Generate human-readable explanation for a prediction."""
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
        for feature, value in zip(self.feature_names, shap_values[0]):
            if abs(value) > 0.1:  # Significant contribution
                explanation['key_factors'].append({
                    'feature': feature,
                    'contribution': float(value),
                    'student_value': float(student_data[feature].iloc[0])
                })
        
        # Generate recommendations
        for factor in explanation['key_factors']:
            if factor['contribution'] < 0:  # Negative contribution
                explanation['recommendations'].append(
                    f"Improve {factor['feature']} from {factor['student_value']}"
                )
        
        return explanation 