import pandas as pd
import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Optional
from data.feature_engineering import AdvancedFeatureEngineering

class DataPreprocessor:
    """Consolidated data preprocessing for all recommendation models."""
    
    def __init__(
        self,
        student_data: pd.DataFrame,
        engagement_data: pd.DataFrame,
        content_data: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42
    ):
        """
        Initialize the data preprocessor.
        
        Args:
            student_data: DataFrame with student information
            engagement_data: DataFrame with engagement history
            content_data: DataFrame with content information
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
        """
        self.student_data = student_data
        self.engagement_data = engagement_data
        self.content_data = content_data
        self.test_size = test_size
        self.random_state = random_state
        
        # Initialize feature engineering
        self.feature_engineering = AdvancedFeatureEngineering(
            student_data=student_data,
            engagement_data=engagement_data
        )
    
    def prepare_data(self) -> Dict:
        """
        Prepare data for training.
        
        Returns:
            Dictionary containing prepared data
        """
        print("Preparing data for training...")
        
        # Check if we have enough data
        if len(self.student_data) == 0 or len(self.engagement_data) == 0 or len(self.content_data) == 0:
            print("Not enough data for training. Generating synthetic data...")
            return self._generate_synthetic_data()
        
        # Create vocabularies
        vocabularies = self._create_vocabularies()
        
        # Create interaction data
        interactions = self._create_interactions()
        
        # Split data
        train_interactions, test_interactions = self._split_data(interactions)
        
        # Create TensorFlow datasets
        train_dataset = self._create_tf_dataset(train_interactions)
        test_dataset = self._create_tf_dataset(test_interactions)
        
        return {
            'train_dataset': train_dataset,
            'test_dataset': test_dataset,
            'vocabularies': vocabularies,
            'dataframes': {
                'students': self.student_data,
                'engagements': self.engagement_data,
                'content': self.content_data
            }
        }
    
    def _create_vocabularies(self) -> Dict[str, List]:
        """Create vocabularies for student IDs, engagement IDs, and content IDs."""
        return {
            'student_ids': self.student_data['student_id'].unique().tolist(),
            'engagement_ids': self.engagement_data['engagement_id'].unique().tolist(),
            'content_ids': self.content_data['content_id'].unique().tolist()
        }
    
    def _create_interactions(self) -> pd.DataFrame:
        """Create interaction data from student and engagement data."""
        try:
            print("Students DataFrame student_id values:", self.student_data['student_id'].unique())
            print("Engagements DataFrame student_id values:", self.engagement_data['student_id'].unique())
            
            interactions = self.engagement_data.merge(
                self.student_data[['student_id', 'funnel_stage', 'dropout_risk_score', 'application_likelihood_score']],
                on='student_id'
            )
            print("Interactions DataFrame shape:", interactions.shape)
            
            # Define funnel stages for ordering
            funnel_stages = ['awareness', 'interest', 'consideration', 'decision', 'application']
            
            # Add effectiveness score based on funnel stage change
            interactions['effectiveness_score'] = interactions.apply(
                lambda row: self._calculate_effectiveness(row, funnel_stages),
                axis=1
            )
            
            return interactions
            
        except Exception as e:
            print(f"Error preparing interaction data: {str(e)}")
            import traceback
            print("Full error details:")
            traceback.print_exc()
            print("Falling back to synthetic data...")
            return self._generate_synthetic_data()
    
    def _calculate_effectiveness(self, row: pd.Series, funnel_stages: List[str]) -> float:
        """Calculate effectiveness score based on funnel stage change."""
        try:
            if row['funnel_stage_after'] != row['funnel_stage_before']:
                if row['funnel_stage_after'] in funnel_stages and row['funnel_stage_before'] in funnel_stages:
                    after_idx = funnel_stages.index(row['funnel_stage_after'])
                    before_idx = funnel_stages.index(row['funnel_stage_before'])
                    if after_idx > before_idx:
                        return 1.0
            return 0.5
        except (KeyError, ValueError, TypeError):
            return 0.5
    
    def _split_data(self, interactions: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into training and testing sets."""
        np.random.seed(self.random_state)
        mask = np.random.rand(len(interactions)) < (1 - self.test_size)
        return interactions[mask], interactions[~mask]
    
    def _create_tf_dataset(self, interactions: pd.DataFrame) -> tf.data.Dataset:
        """Create TensorFlow dataset from interactions."""
        return tf.data.Dataset.from_tensor_slices({
            "student_id": interactions['student_id'].values,
            "engagement_id": interactions['engagement_id'].values,
            "content_id": interactions['engagement_content_id'].values,
            "effectiveness_score": interactions['effectiveness_score'].values,
            "application_likelihood": interactions['application_likelihood_score'].values,
            "dropout_risk": interactions['dropout_risk_score'].values
        }).shuffle(10000).batch(128)
    
    def _generate_synthetic_data(self) -> Dict:
        """Generate synthetic data for testing."""
        # This is a placeholder - implement synthetic data generation
        # based on your specific requirements
        raise NotImplementedError("Synthetic data generation not implemented")
    
    def prepare_cross_validation_data(self, n_splits: int = 5) -> List[Dict]:
        """
        Prepare data for cross-validation.
        
        Args:
            n_splits: Number of cross-validation splits
            
        Returns:
            List of dictionaries containing prepared data for each fold
        """
        # Sort by timestamp
        sorted_data = self.engagement_data.sort_values('timestamp')
        
        # Create time-based folds
        fold_size = len(sorted_data) // n_splits
        folds = []
        
        for i in range(n_splits):
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size if i < n_splits - 1 else len(sorted_data)
            folds.append(sorted_data.iloc[start_idx:end_idx])
        
        # Prepare data for each fold
        cv_data = []
        for i in range(n_splits):
            # Use all folds except the current one for training
            train_data = pd.concat(folds[:i] + folds[i+1:])
            val_data = folds[i]
            
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
            
            cv_data.append({
                'train_input': train_input,
                'val_input': val_input,
                'val_labels': val_data['applied']
            })
        
        return cv_data 