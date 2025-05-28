import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import tensorflow as tf
from data.feature_engineering import AdvancedFeatureEngineering

class CrossValidationTrainer:
    def __init__(self, model, data: pd.DataFrame, n_splits: int = 5):
        self.model = model
        self.data = data
        self.n_splits = n_splits
        self.feature_engineering = AdvancedFeatureEngineering(
            student_data=data,
            engagement_data=data
        )
        
    def time_based_split(self) -> List[pd.DataFrame]:
        """Split data based on time to prevent data leakage"""
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
        """Prepare data for a single fold"""
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
        """Train model using time-based cross-validation"""
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
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Calculate feature importance across folds"""
        importance_scores = []
        
        for i in range(self.n_splits):
            # Get fold data
            folds = self.time_based_split()
            train_data = pd.concat(folds[:i] + folds[i+1:])
            val_data = folds[i]
            
            # Get features
            features = self.feature_engineering.create_all_features()
            
            # Calculate importance for this fold
            fold_importance = self.model.get_feature_importance(
                features,
                val_data['applied']
            )
            
            importance_scores.append(fold_importance)
        
        # Average importance across folds
        avg_importance = pd.concat(importance_scores).groupby(level=0).mean()
        
        return avg_importance.sort_values(ascending=False) 