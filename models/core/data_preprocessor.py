import pandas as pd
import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Optional, Any
from data.processing.feature_engineering import AdvancedFeatureEngineering
from utils.application_likelihood import calculate_application_likelihood
from datetime import datetime

class DataPreprocessor:
    """Consolidated data preprocessing for all recommendation models."""
    
    def __init__(
        self,
        student_data: pd.DataFrame,
        engagement_data: pd.DataFrame,
        content_data: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
        db=None
    ):
        """
        Initialize the data preprocessor.
        
        Args:
            student_data: DataFrame with student information
            engagement_data: DataFrame with engagement history
            content_data: DataFrame with content information
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            db: Database session
        """
        self.student_data = student_data
        self.engagement_data = engagement_data
        self.content_data = content_data
        self.test_size = test_size
        self.random_state = random_state
        self.db = db
        
        # Initialize feature engineering
        self.feature_engineering = AdvancedFeatureEngineering(
            student_data=student_data,
            engagement_data=engagement_data
        )
        
        # Initialize data quality monitoring
        self.feature_stats = {}
        self.missing_values = {}
        self.outlier_counts = {}
        self.missing_threshold = 0.1  # 10% missing values threshold
        self.outlier_threshold = 3.0  # 3 standard deviations for outliers
    
    def monitor_data_quality(self, features: Dict[str, tf.Tensor]) -> None:
        """Monitor data quality during model training."""
        for feature_name, feature_values in features.items():
            # Skip non-numeric features
            if not isinstance(feature_values, (tf.Tensor, np.ndarray)):
                continue
            
            # Convert to numpy for easier computation
            values = feature_values.numpy() if isinstance(feature_values, tf.Tensor) else feature_values
            
            # Update feature statistics
            if feature_name not in self.feature_stats:
                self.feature_stats[feature_name] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "count": len(values)
                }
            else:
                stats = self.feature_stats[feature_name]
                n = stats["count"]
                new_n = n + len(values)
                
                # Update mean and standard deviation using Welford's online algorithm
                delta = values - stats["mean"]
                stats["mean"] += np.sum(delta) / new_n
                delta2 = values - stats["mean"]
                stats["std"] = np.sqrt((stats["std"]**2 * n + np.sum(delta * delta2)) / new_n)
                stats["min"] = min(stats["min"], np.min(values))
                stats["max"] = max(stats["max"], np.max(values))
                stats["count"] = new_n
            
            # Update missing values count
            missing_count = np.sum(np.isnan(values))
            if feature_name not in self.missing_values:
                self.missing_values[feature_name] = missing_count
            else:
                self.missing_values[feature_name] += missing_count
            
            # Update outlier count
            if stats["std"] > 0:
                z_scores = np.abs((values - stats["mean"]) / stats["std"])
                outlier_count = np.sum(z_scores > self.outlier_threshold)
                if feature_name not in self.outlier_counts:
                    self.outlier_counts[feature_name] = outlier_count
                else:
                    self.outlier_counts[feature_name] += outlier_count
    
    def get_quality_report(self) -> Dict[str, Dict[str, float]]:
        """Get a report of data quality metrics."""
        report = {}
        for feature_name in self.feature_stats:
            stats = self.feature_stats[feature_name]
            missing_ratio = self.missing_values.get(feature_name, 0) / stats["count"]
            outlier_ratio = self.outlier_counts.get(feature_name, 0) / stats["count"]
            
            report[feature_name] = {
                "mean": float(stats["mean"]),
                "std": float(stats["std"]),
                "min": float(stats["min"]),
                "max": float(stats["max"]),
                "missing_ratio": float(missing_ratio),
                "outlier_ratio": float(outlier_ratio),
                "quality_score": float(1.0 - missing_ratio - outlier_ratio)
            }
        
        return report
    
    def check_quality_issues(self) -> List[str]:
        """Check for data quality issues and return warnings."""
        warnings = []
        report = self.get_quality_report()
        
        for feature_name, metrics in report.items():
            if metrics["missing_ratio"] > self.missing_threshold:
                warnings.append(
                    f"High missing values in {feature_name}: "
                    f"{metrics['missing_ratio']:.1%} missing"
                )
            
            if metrics["outlier_ratio"] > self.missing_threshold:
                warnings.append(
                    f"High outlier ratio in {feature_name}: "
                    f"{metrics['outlier_ratio']:.1%} outliers"
                )
            
            if metrics["quality_score"] < 0.7:
                warnings.append(
                    f"Low quality score in {feature_name}: "
                    f"{metrics['quality_score']:.2f}"
                )
        
        return warnings
    
    def reset_quality_metrics(self) -> None:
        """Reset all quality monitoring statistics."""
        self.feature_stats = {}
        self.missing_values = {}
        self.outlier_counts = {}
    
    def prepare_data(self) -> Dict:
        """Prepare data for training."""
        print("Preparing data for training...")
        
        # Check if we have enough data
        if len(self.student_data) == 0 or len(self.engagement_data) == 0:
            print("Not enough data for training. Generating synthetic data...")
            interactions = self._generate_synthetic_data()
        else:
            # Create vocabularies
            vocabularies = self._create_vocabularies()
            
            # Create interaction data
            interactions = self._create_interactions()
            
        # Process timestamps to ensure proper datetime objects
        self._process_timestamps()
        
        # Prepare engagement features
        engagement_features = self._prepare_engagement_features()
        
        # Split data
        train_interactions, test_interactions = self._split_data(interactions)
        
        # Create TensorFlow datasets
        train_dataset = self._create_tf_dataset(train_interactions)
        test_dataset = self._create_tf_dataset(test_interactions)
        
        # Create vocabularies from the data
        vocabularies = self._create_vocabularies()
        
        return {
            'train_dataset': train_dataset,
            'test_dataset': test_dataset,
            'vocabularies': vocabularies,
            'dataframes': {
                'students': self.student_data,
                'engagements': self.engagement_data
            },
            'engagement_features': engagement_features
        }
    
    def _create_vocabularies(self) -> Dict[str, List]:
        """Create vocabularies for student IDs and engagement IDs."""
        return {
            'student_vocab': self.student_data['student_id'].unique().tolist(),
            'engagement_vocab': self.engagement_data['engagement_id'].unique().tolist(),
            'student_ids': self.student_data['student_id'].unique().tolist(),
            'engagement_ids': self.engagement_data['engagement_id'].unique().tolist()
        }
    
    def _create_interactions(self) -> pd.DataFrame:
        """Create interaction data from student and engagement data."""
        try:
            print("Students DataFrame student_id values:", self.student_data['student_id'].unique())
            print("Engagements DataFrame student_id values:", self.engagement_data['student_id'].unique())
            
            interactions = self.engagement_data.merge(
                self.student_data[['student_id', 'funnel_stage']],
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
            
            # Compute application_likelihood_score for each student
            for student_id in interactions['student_id'].unique():
                student = self.student_data[self.student_data['student_id'] == student_id].iloc[0]
                student_engagements = self.engagement_data[self.engagement_data['student_id'] == student_id].to_dict('records')
                interactions.loc[interactions['student_id'] == student_id, 'application_likelihood_score'] = calculate_application_likelihood(
                    student=student.to_dict(),
                    engagements=student_engagements,
                    db=self.db
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
        # Set default dropout risk score if not present
        if 'dropout_risk_score' not in interactions.columns:
            interactions['dropout_risk_score'] = 0.5
            
        # Create features dictionary
        features = {
            "student_id": interactions['student_id'].values,
            "engagement_id": interactions['engagement_id'].values,
            "student_features": np.zeros((len(interactions), 10), dtype=np.float32),  # Placeholder for student features
            "engagement_features": np.zeros((len(interactions), 10), dtype=np.float32)  # Placeholder for engagement features
        }
        
        # Create labels dictionary with matching keys to model output - ensure float32 types
        labels = {
            "ranking_score": interactions['effectiveness_score'].astype(np.float32).values,
            "likelihood_score": interactions['application_likelihood_score'].astype(np.float32).values,
            "risk_score": interactions['dropout_risk_score'].astype(np.float32).values
        }
        
        # Create dataset with features and labels
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        return dataset.shuffle(10000).batch(128)
    
    def _generate_synthetic_data(self) -> pd.DataFrame:
        """Generate synthetic interaction data for training when real data is not available."""
        # Create empty DataFrame with required columns
        interactions = pd.DataFrame(columns=[
            'student_id', 'engagement_id', 'interaction_type', 'timestamp',
            'duration', 'completion_rate', 'engagement_score',
            'application_likelihood_score', 'risk_score', 'effectiveness_score',
            'student_features', 'engagement_features', 'label', 'weight'
        ])
        
        # Get unique student and engagement IDs
        student_ids = self.student_data['student_id'].unique()
        engagement_ids = self.engagement_data['engagement_id'].unique()
        
        # Generate synthetic interactions
        num_interactions = min(1000, len(student_ids) * len(engagement_ids))
        interactions['student_id'] = np.random.choice(student_ids, num_interactions)
        interactions['engagement_id'] = np.random.choice(engagement_ids, num_interactions)
        
        # Generate random features
        interactions['interaction_type'] = np.random.choice(['view', 'click', 'complete'], num_interactions)
        interactions['timestamp'] = pd.date_range(start='2024-01-01', periods=num_interactions, freq='h')
        interactions['duration'] = np.random.uniform(0, 3600, num_interactions).astype(np.float32)  # 0 to 1 hour in seconds
        interactions['completion_rate'] = np.random.uniform(0, 1, num_interactions).astype(np.float32)
        
        # Generate scores - ensure they are float32
        interactions['engagement_score'] = np.random.uniform(0, 1, num_interactions).astype(np.float32)
        interactions['application_likelihood_score'] = np.random.uniform(0, 1, num_interactions).astype(np.float32)
        interactions['risk_score'] = np.random.uniform(0, 1, num_interactions).astype(np.float32)
        interactions['effectiveness_score'] = np.random.uniform(0, 1, num_interactions).astype(np.float32)
        interactions['dropout_risk_score'] = np.random.uniform(0, 1, num_interactions).astype(np.float32)
        
        # Generate features
        interactions['student_features'] = [np.random.rand(10).astype(np.float32) for _ in range(num_interactions)]
        interactions['engagement_features'] = [np.random.rand(10).astype(np.float32) for _ in range(num_interactions)]
        
        # Generate labels and weights
        interactions['label'] = (interactions['completion_rate'] > 0.5).astype(int)
        interactions['weight'] = (interactions['completion_rate'] * (1 + interactions['engagement_score'])).astype(np.float32)
        
        return interactions
    
    def _prepare_engagement_features(self) -> Dict[str, Any]:
        """
        Prepare engagement features.
        
        Returns:
            Dictionary containing processed features
        """
        # Prepare student features
        student_features = self.student_data.set_index('student_id').to_dict(orient='index')

        # Prepare engagement type features (categorical)
        engagement_types = self.engagement_data['engagement_type'].unique().tolist()
        engagement_type_to_idx = {et: i for i, et in enumerate(engagement_types)}
        self.engagement_data['engagement_type_idx'] = self.engagement_data['engagement_type'].map(engagement_type_to_idx)

        return {
            'merged_df': self.engagement_data,
            'student_features': student_features,
            'engagement_types': engagement_types,
            'engagement_type_to_idx': engagement_type_to_idx
        }
    
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

    def _process_timestamps(self):
        """Process timestamp fields in the data."""
        if self.engagement_data is not None and 'timestamp' in self.engagement_data.columns:
            # Convert timestamp to datetime, handling potential format issues
            self.engagement_data['timestamp'] = pd.to_datetime(
                self.engagement_data['timestamp'],
                format='mixed',  # Allow mixed formats
                errors='coerce'  # Convert invalid dates to NaT
            )
            # Drop rows with invalid timestamps
            self.engagement_data = self.engagement_data.dropna(subset=['timestamp'])
            # Ensure timezone-naive datetime objects
            self.engagement_data['timestamp'] = self.engagement_data['timestamp'].dt.tz_localize(None) 