import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.layers import Normalization

class DataProcessor:
    """Class for processing and preparing data for the recommender model."""
    
    def __init__(self, students_df: pd.DataFrame, engagements_df: pd.DataFrame):
        """Initialize the data processor with student and engagement data."""
        self.students_df = students_df
        self.engagements_df = engagements_df
        
        # Initialize normalization layers
        self.student_normalizer = Normalization()
        self.engagement_normalizer = Normalization()
        
        # Flags to track if scalers have been initialized
        self._student_scaler_initialized = False
        self._engagement_scaler_initialized = False
        
        # Process timestamps
        self._process_timestamps()
    
    def _process_timestamps(self):
        """Process timestamp fields in the dataframes."""
        # Convert timestamp to datetime if it's not already
        if 'timestamp' in self.engagements_df.columns:
            self.engagements_df['timestamp'] = pd.to_datetime(self.engagements_df['timestamp'])
    
    def normalize_features(self, features: np.ndarray, feature_type: str) -> np.ndarray:
        """Normalize features using the appropriate scaler."""
        if feature_type == 'student' and not self._student_scaler_initialized:
            self.student_normalizer.adapt(features)
            self._student_scaler_initialized = True
        elif feature_type == 'engagement' and not self._engagement_scaler_initialized:
            self.engagement_normalizer.adapt(features)
            self._engagement_scaler_initialized = True
            
        if feature_type == 'student':
            return self.student_normalizer(features)
        else:
            return self.engagement_normalizer(features)
    
    def prepare_data(self) -> Dict:
        """Prepare data for training."""
        # Extract features
        student_features = self._extract_student_features()
        engagement_features = self._extract_engagement_features()
        
        # Normalize features
        student_features = self.normalize_features(student_features, 'student')
        engagement_features = self.normalize_features(engagement_features, 'engagement')
        
        # Create interaction dataset
        interactions = self._create_interaction_dataset()
        
        # Split into train/test
        train_data, test_data = self._split_data(interactions)
        
        return {
            'train_data': train_data,
            'test_data': test_data,
            'student_features': student_features,
            'engagement_features': engagement_features
        }
    
    def _extract_student_features(self) -> np.ndarray:
        """Extract and preprocess student features."""
        # Select relevant columns and handle missing values
        features = self.students_df.select_dtypes(include=[np.number]).fillna(0)
        return features.values
    
    def _extract_engagement_features(self) -> np.ndarray:
        """Extract and preprocess engagement features."""
        # Select relevant columns and handle missing values
        features = self.engagements_df.select_dtypes(include=[np.number]).fillna(0)
        return features.values
    
    def _create_interaction_dataset(self) -> tf.data.Dataset:
        """Create interaction dataset from student-engagement pairs."""
        # Create positive interactions
        positive_interactions = self._create_positive_interactions()
        
        # Create negative interactions
        negative_interactions = self._create_negative_interactions()
        
        # Combine and shuffle
        all_interactions = np.concatenate([positive_interactions, negative_interactions])
        np.random.shuffle(all_interactions)
        
        return tf.data.Dataset.from_tensor_slices(all_interactions)
    
    def _create_positive_interactions(self) -> np.ndarray:
        """Create positive interaction pairs."""
        positive_pairs = []
        for _, engagement in self.engagements_df.iterrows():
            student_id = engagement['student_id']
            if student_id in self.students_df['student_id'].values:
                positive_pairs.append([
                    student_id,
                    engagement['engagement_id'],
                    1.0  # Positive interaction
                ])
        return np.array(positive_pairs)
    
    def _create_negative_interactions(self) -> np.ndarray:
        """Create negative interaction pairs through sampling."""
        negative_pairs = []
        all_student_ids = self.students_df['student_id'].values
        all_engagement_ids = self.engagements_df['engagement_id'].values
        
        # Sample negative pairs
        n_negative = len(self.engagements_df)  # Match number of positive interactions
        for _ in range(n_negative):
            student_id = np.random.choice(all_student_ids)
            engagement_id = np.random.choice(all_engagement_ids)
            
            # Ensure this is not a positive interaction
            if not self._is_positive_interaction(student_id, engagement_id):
                negative_pairs.append([
                    student_id,
                    engagement_id,
                    0.0  # Negative interaction
                ])
        
        return np.array(negative_pairs)
    
    def _is_positive_interaction(self, student_id: str, engagement_id: str) -> bool:
        """Check if a student-engagement pair is a positive interaction."""
        return any(
            (self.engagements_df['student_id'] == student_id) & 
            (self.engagements_df['engagement_id'] == engagement_id)
        )
    
    def _split_data(self, dataset: tf.data.Dataset, test_size: float = 0.2) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """Split dataset into training and testing sets."""
        dataset_size = len(list(dataset.as_numpy_iterator()))
        train_size = int((1 - test_size) * dataset_size)
        
        train_data = dataset.take(train_size)
        test_data = dataset.skip(train_size)
        
        return train_data, test_data

    def load_data_from_csv(self, students_csv: str, engagements_csv: str) -> Dict[str, pd.DataFrame]:
        """Load data from CSV files."""
        # Load students data
        students_df = pd.read_csv(students_csv)
        
        # Load engagements data with timestamp parsing
        engagements_df = pd.read_csv(engagements_csv, parse_dates=['timestamp'])
        
        return {
            'students': students_df,
            'engagements': engagements_df
        } 