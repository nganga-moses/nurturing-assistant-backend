import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple, Any
from database.session import get_db
from .models import StudentProfile, EngagementHistory, EngagementContent


class DataProcessor:
    """
    Handles data preprocessing, feature engineering, and dataset creation for the recommendation models.
    """
    
    def __init__(self):
        self.session = get_db()
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load data from the database into pandas DataFrames
        
        Returns:
            Tuple of DataFrames (students, engagements, content)
        """
        # Query all data from the database
        students_query = self.session.query(StudentProfile).all()
        engagements_query = self.session.query(EngagementHistory).all()
        content_query = self.session.query(EngagementContent).all()
        
        # Convert to dictionaries
        students_data = [student.to_dict() for student in students_query]
        engagements_data = [engagement.to_dict() for engagement in engagements_query]
        content_data = [content.to_dict() for content in content_query]
        
        # Create DataFrames
        students_df = pd.DataFrame(students_data)
        engagements_df = pd.DataFrame(engagements_data)
        content_df = pd.DataFrame(content_data)
        
        return students_df, engagements_df, content_df
    
    def preprocess_data(self, students_df, engagements_df, content_df):
        """
        Preprocess the data for model training
        
        Args:
            students_df: DataFrame with student profiles
            engagements_df: DataFrame with engagement history
            content_df: DataFrame with engagement content
            
        Returns:
            Preprocessed DataFrames
        """
        # Convert datetime strings to datetime objects
        for df in [students_df, engagements_df]:
            for col in df.columns:
                if 'date' in col.lower() or 'time' in col.lower():
                    df[col] = pd.to_datetime(df[col], errors='ignore')
        
        # Handle missing values
        students_df = students_df.fillna({
            'application_likelihood_score': 0.5,
            'dropout_risk_score': 0.5,
            'interaction_count': 0
        })
        
        # Normalize numerical features
        numerical_cols = ['application_likelihood_score', 'dropout_risk_score', 'interaction_count']
        for col in numerical_cols:
            if col in students_df.columns:
                students_df[col] = (students_df[col] - students_df[col].mean()) / students_df[col].std()
        
        # Process JSON columns
        for df, json_cols in [
            (students_df, ['demographic_features']),
            (engagements_df, ['engagement_metrics']),
            (content_df, ['content_features'])
        ]:
            for col in json_cols:
                if col in df.columns:
                    # Ensure JSON strings are converted to dictionaries
                    df[col] = df[col].apply(
                        lambda x: json.loads(x) if isinstance(x, str) else x
                    )
        
        return students_df, engagements_df, content_df
    
    def engineer_features(self, students_df, engagements_df):
        """
        Create engineered features for the model
        
        Args:
            students_df: DataFrame with student profiles
            engagements_df: DataFrame with engagement history
            
        Returns:
            DataFrame with additional engineered features
        """
        # Create a copy to avoid modifying the original
        students_enhanced = students_df.copy()
        
        # Calculate recency features (days since last engagement)
        current_time = datetime.now()
        
        if 'last_interaction_date' in students_enhanced.columns:
            students_enhanced['days_since_last_interaction'] = students_enhanced['last_interaction_date'].apply(
                lambda x: (current_time - x).days if pd.notnull(x) else 30  # Default to 30 days if null
            )
        
        # Calculate engagement frequency (per week)
        if 'first_interaction_date' in students_enhanced.columns and 'interaction_count' in students_enhanced.columns:
            students_enhanced['weeks_active'] = students_enhanced['first_interaction_date'].apply(
                lambda x: max(1, (current_time - x).days / 7) if pd.notnull(x) else 1
            )
            students_enhanced['engagement_frequency'] = students_enhanced['interaction_count'] / students_enhanced['weeks_active']
        
        # Calculate engagement patterns (increasing, decreasing, stable)
        if not engagements_df.empty and 'timestamp' in engagements_df.columns:
            # Group engagements by student and calculate engagement trend
            engagement_counts = engagements_df.copy()
            min_timestamp = engagement_counts['timestamp'].min()
            engagement_counts['week'] = engagement_counts['timestamp'].apply(
                lambda x: (x - min_timestamp).days // 7 if pd.notnull(x) else 0
            )
            
            # Get counts by student and week
            weekly_counts = engagement_counts.groupby(['student_id', 'week']).size().reset_index(name='count')
            
            # Calculate trend for each student
            student_trends = {}
            for student_id, group in weekly_counts.groupby('student_id'):
                if len(group) >= 2:
                    # Simple linear regression to determine trend
                    x = group['week'].values
                    y = group['count'].values
                    
                    # Calculate slope
                    n = len(x)
                    slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
                    
                    if slope > 0.1:
                        trend = 'increasing'
                    elif slope < -0.1:
                        trend = 'decreasing'
                    else:
                        trend = 'stable'
                else:
                    trend = 'new'
                
                student_trends[student_id] = trend
            
            # Add trend to student dataframe
            students_enhanced['engagement_trend'] = students_enhanced['student_id'].map(student_trends)
            students_enhanced['engagement_trend'] = students_enhanced['engagement_trend'].fillna('new')
        
        # Calculate funnel progression velocity
        if not engagements_df.empty and 'funnel_stage_before' in engagements_df.columns and 'funnel_stage_after' in engagements_df.columns:
            # Define funnel stages and their order
            funnel_stages = {
                'Awareness': 1,
                'Interest': 2,
                'Consideration': 3,
                'Decision': 4,
                'Application': 5
            }
            
            # Calculate progression for each engagement
            engagements_df['stage_progression'] = engagements_df.apply(
                lambda row: funnel_stages.get(row['funnel_stage_after'], 0) - funnel_stages.get(row['funnel_stage_before'], 0),
                axis=1
            )
            
            # Calculate average progression velocity for each student
            progression_velocity = engagements_df.groupby('student_id')['stage_progression'].mean().reset_index()
            progression_velocity.columns = ['student_id', 'funnel_velocity']
            
            # Merge with student dataframe
            students_enhanced = students_enhanced.merge(progression_velocity, on='student_id', how='left')
            students_enhanced['funnel_velocity'] = students_enhanced['funnel_velocity'].fillna(0)
        
        return students_enhanced
    
    def create_tf_dataset(self, students_df, engagements_df, content_df, batch_size=256):
        """
        Create TensorFlow datasets for training the recommendation model
        
        Args:
            students_df: DataFrame with student profiles
            engagements_df: DataFrame with engagement history
            content_df: DataFrame with engagement content
            batch_size: Batch size for the dataset
            
        Returns:
            TensorFlow datasets for training and testing
        """
        # Rename 'engagement_content_id' to 'content_id' in engagements_df
        engagements_df = engagements_df.rename(columns={'engagement_content_id': 'content_id'})
        
        # Merge engagements with students and content
        merged_data = engagements_df.merge(students_df, on='student_id', how='left')
        merged_data = merged_data.merge(content_df, on='content_id', how='left')
        
        # Create features and labels
        features = {
            'student_id': merged_data['student_id'].values,
            'engagement_id': merged_data['engagement_id'].values,
            'content_id': merged_data['content_id'].values,
            'funnel_stage': merged_data['funnel_stage'].values,
            'application_likelihood_score': merged_data['application_likelihood_score'].values,
            'dropout_risk_score': merged_data['dropout_risk_score'].values
        }
        
        # Use engagement response as label (convert to numerical)
        response_mapping = {
            'opened': 1.0,
            'clicked': 2.0,
            'responded': 3.0,
            'attended': 4.0,
            'completed': 5.0,
            'ignored': 0.0,
            'bounced': -1.0
        }
        
        labels = merged_data['engagement_response'].map(response_mapping).fillna(0).values
        
        # Split into training and testing sets (80/20)
        n = len(merged_data)
        train_indices = np.random.choice(n, int(0.8 * n), replace=False)
        test_indices = np.array(list(set(range(n)) - set(train_indices)))
        
        train_features = {k: v[train_indices] for k, v in features.items()}
        test_features = {k: v[test_indices] for k, v in features.items()}
        
        train_labels = labels[train_indices]
        test_labels = labels[test_indices]
        
        # Create TensorFlow datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((train_features, train_labels))
        train_dataset = train_dataset.shuffle(buffer_size=10000).batch(batch_size)
        
        test_dataset = tf.data.Dataset.from_tensor_slices((test_features, test_labels))
        test_dataset = test_dataset.batch(batch_size)
        
        return train_dataset, test_dataset
    
    def prepare_data_for_training(self, batch_size=256):
        """
        Complete pipeline to prepare data for model training
        
        Args:
            batch_size: Batch size for the dataset
            
        Returns:
            TensorFlow datasets and vocabularies
        """
        # Load data
        students_df, engagements_df, content_df = self.load_data()
        
        # Preprocess data
        students_df, engagements_df, content_df = self.preprocess_data(students_df, engagements_df, content_df)
        
        # Engineer features
        students_enhanced = self.engineer_features(students_df, engagements_df)
        
        # Create TensorFlow datasets
        train_dataset, test_dataset = self.create_tf_dataset(
            students_enhanced, engagements_df, content_df, batch_size
        )
        
        # Create vocabularies for categorical features
        student_ids = list(students_df['student_id'].unique())
        engagement_ids = list(engagements_df['engagement_id'].unique())
        content_ids = list(content_df['content_id'].unique())
        funnel_stages = list(students_df['funnel_stage'].unique())
        
        return {
            'train_dataset': train_dataset,
            'test_dataset': test_dataset,
            'vocabularies': {
                'student_ids': student_ids,
                'engagement_ids': engagement_ids,
                'content_ids': content_ids,
                'funnel_stages': funnel_stages
            },
            'dataframes': {
                'students': students_enhanced,
                'engagements': engagements_df,
                'content': content_df
            }
        }
    
    def generate_sample_data(self, num_students=100, num_engagements_per_student=10, num_content_items=50):
        """
        Generate sample data for testing and development
        
        Args:
            num_students: Number of student profiles to generate
            num_engagements_per_student: Average number of engagements per student
            num_content_items: Number of content items to generate
            
        Returns:
            Generated DataFrames
        """
        # Generate student profiles
        students_data = []
        for i in range(num_students):
            student_id = f"S{i+1:05d}"
            funnel_stages = ['Awareness', 'Interest', 'Consideration', 'Decision', 'Application']
            funnel_stage = np.random.choice(funnel_stages, p=[0.2, 0.3, 0.25, 0.15, 0.1])
            
            first_date = datetime.now() - timedelta(days=np.random.randint(1, 180))
            last_date = first_date + timedelta(days=np.random.randint(1, 90))
            
            students_data.append({
                'student_id': student_id,
                'demographic_features': {
                    'location': np.random.choice(['California', 'New York', 'Texas', 'Florida', 'Illinois']),
                    'age_range': np.random.choice(['18-24', '25-34', '35-44']),
                    'intended_major': np.random.choice(['Computer Science', 'Business', 'Engineering', 'Arts', 'Medicine']),
                    'academic_scores': {
                        'GPA': round(np.random.uniform(2.0, 4.0), 2),
                        'SAT': np.random.randint(1000, 1600)
                    }
                },
                'application_status': 'In Progress' if funnel_stage != 'Application' else 'Completed',
                'funnel_stage': funnel_stage,
                'first_interaction_date': first_date,
                'last_interaction_date': last_date,
                'interaction_count': np.random.randint(1, 30),
                'application_likelihood_score': round(np.random.uniform(0, 1), 2),
                'dropout_risk_score': round(np.random.uniform(0, 1), 2),
                'last_recommended_engagement_id': None,
                'last_recommended_engagement_date': None
            })
        
        # Generate content items
        content_data = []
        for i in range(num_content_items):
            content_id = f"C{i+1:05d}"
            engagement_type = np.random.choice(['Email', 'SMS', 'Visit'], p=[0.6, 0.3, 0.1])
            target_funnel_stage = np.random.choice(['Awareness', 'Interest', 'Consideration', 'Decision'])
            
            # Generate random embedding vector
            embedding = [round(np.random.uniform(-1, 1), 2) for _ in range(10)]
            
            content_data.append({
                'content_id': content_id,
                'engagement_type': engagement_type,
                'content_category': np.random.choice(['Program Information', 'Application Help', 'Campus Life', 'Financial Aid']),
                'content_description': f"{engagement_type} content about {np.random.choice(['programs', 'campus', 'admissions', 'scholarships'])}",
                'content_features': {
                    'topics': np.random.choice(['curriculum', 'faculty', 'research', 'admissions', 'deadlines'], size=2, replace=False).tolist(),
                    'embedding': embedding
                },
                'success_rate': round(np.random.uniform(0.3, 0.9), 2),
                'target_funnel_stage': target_funnel_stage,
                'appropriate_for_risk_level': np.random.choice(['low', 'medium', 'high'])
            })
        
        # Generate engagement history
        engagements_data = []
        engagement_counter = 1
        
        for student in students_data:
            student_id = student['student_id']
            num_engagements = np.random.randint(1, num_engagements_per_student * 2)
            
            for j in range(num_engagements):
                engagement_id = f"E{engagement_counter:05d}"
                engagement_counter += 1
                
                # Select a random content item
                content_item = np.random.choice(content_data)
                content_id = content_item['content_id']
                
                # Generate timestamp between first and last interaction
                first_date = student['first_interaction_date']
                last_date = student['last_interaction_date']
                
                timestamp = first_date + (last_date - first_date) * np.random.random()
                
                # Determine funnel stages
                funnel_stages = ['Awareness', 'Interest', 'Consideration', 'Decision', 'Application']
                current_stage_idx = funnel_stages.index(student['funnel_stage'])
                
                # Determine funnel stage before this engagement
                possible_before_stages = funnel_stages[:current_stage_idx + 1]
                funnel_stage_before = np.random.choice(possible_before_stages)
                
                # Determine funnel stage after this engagement
                possible_after_stages = funnel_stages[funnel_stages.index(funnel_stage_before):]
                funnel_stage_after = np.random.choice(possible_after_stages)
                
                # Determine engagement response
                response_options = ['opened', 'clicked', 'responded', 'attended', 'completed', 'ignored', 'bounced']
                response_probs = [0.3, 0.2, 0.15, 0.1, 0.05, 0.15, 0.05]
                engagement_response = np.random.choice(response_options, p=response_probs)
                
                engagements_data.append({
                    'engagement_id': engagement_id,
                    'student_id': student_id,
                    'engagement_type': content_item['engagement_type'],
                    'engagement_content_id': content_id,
                    'timestamp': timestamp,
                    'engagement_response': engagement_response,
                    'engagement_metrics': {
                        'metrics': self._generate_metrics_for_type(content_item['engagement_type'], engagement_response)
                    },
                    'funnel_stage_before': funnel_stage_before,
                    'funnel_stage_after': funnel_stage_after
                })
        
        # Create DataFrames
        students_df = pd.DataFrame(students_data)
        content_df = pd.DataFrame(content_data)
        engagements_df = pd.DataFrame(engagements_data)
        
        return students_df, engagements_df, content_df
    
    def _generate_metrics_for_type(self, engagement_type, response):
        """Helper method to generate type-specific metrics"""
        if engagement_type == 'Email':
            if response in ['opened', 'clicked', 'responded']:
                return {
                    'open_time': (datetime.now() - timedelta(days=np.random.randint(1, 30))).isoformat(),
                    'click_through': response in ['clicked', 'responded'],
                    'time_spent': np.random.randint(10, 300) if response in ['clicked', 'responded'] else 0
                }
            else:
                return {
                    'delivered': response != 'bounced',
                    'bounced': response == 'bounced'
                }
        elif engagement_type == 'SMS':
            return {
                'delivered': response != 'bounced',
                'response_time': np.random.randint(1, 1440) if response == 'responded' else None,
                'response_length': np.random.randint(10, 200) if response == 'responded' else 0
            }
        elif engagement_type == 'Visit':
            if response in ['attended', 'completed']:
                return {
                    'duration_minutes': np.random.randint(30, 180),
                    'activities_participated': np.random.randint(1, 5),
                    'staff_interactions': np.random.randint(0, 3)
                }
            else:
                return {
                    'registered': True,
                    'attended': False,
                    'reason_for_absence': np.random.choice(['Schedule Conflict', 'Transportation Issues', 'Changed Mind', 'Unknown'])
                }
        else:
            return {}
