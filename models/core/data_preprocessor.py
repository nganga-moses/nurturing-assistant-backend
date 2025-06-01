import pandas as pd
import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Optional, Any
from data.processing.feature_engineering import AdvancedFeatureEngineering

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
        
        # Prepare engagement and content features
        engagement_content_features = self._prepare_engagement_content_features()
        
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
            },
            'engagement_content_features': engagement_content_features
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
    
    def _prepare_engagement_content_features(self) -> Dict[str, Any]:
        """
        Prepare engagement and content features.
        
        Returns:
            Dictionary containing processed features
        """
        # Join engagement and content data (left join, so missing content is allowed)
        merged = self.engagement_data.merge(
            self.content_data,
            how='left',
            left_on='engagement_content_id',
            right_on='content_id',
            suffixes=('', '_content')
        )

        # Fill missing content fields with defaults
        merged['content_id'] = merged['content_id'].fillna('NO_CONTENT')
        merged['content_type'] = merged.get('content_type', pd.Series(['none']*len(merged)))
        merged['content_body'] = merged.get('content_body', pd.Series(['']*len(merged)))
        merged['content_meta'] = merged.get('content_meta', pd.Series([{}]*len(merged)))

        # Prepare student features
        student_features = self.student_data.set_index('student_id').to_dict(orient='index')

        # Prepare engagement type features (categorical)
        engagement_types = merged['engagement_type'].unique().tolist()
        engagement_type_to_idx = {et: i for i, et in enumerate(engagement_types)}
        merged['engagement_type_idx'] = merged['engagement_type'].map(engagement_type_to_idx)

        # Prepare content features (text/meta, handle missing)
        content_ids = merged['content_id'].unique().tolist()
        content_id_to_idx = {cid: i for i, cid in enumerate(content_ids)}
        merged['content_id_idx'] = merged['content_id'].map(content_id_to_idx)

        # Mask for missing content
        merged['has_content'] = merged['content_id'] != 'NO_CONTENT'

        return {
            'merged_df': merged,
            'student_features': student_features,
            'engagement_types': engagement_types,
            'engagement_type_to_idx': engagement_type_to_idx,
            'content_ids': content_ids,
            'content_id_to_idx': content_id_to_idx
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