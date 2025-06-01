import pytest
import pandas as pd
import tensorflow as tf
from datetime import datetime, timedelta
import json
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.models.models import init_db, get_session
from data.sample_data import generate_sample_data

@pytest.fixture(scope="session")
def db_session():
    """Create a database session for testing."""
    init_db()
    session = get_session()
    yield session
    session.close()

@pytest.fixture(scope="session")
def sample_data():
    """Generate sample data for testing."""
    return generate_sample_data(num_students=10, num_engagements_per_student=5, num_content_items=20)

@pytest.fixture(scope="session")
def tf_dataset():
    """Create a TensorFlow dataset for testing."""
    # Create sample data
    num_samples = 100
    student_ids = [f"student{i}" for i in range(num_samples)]
    engagement_ids = [f"engagement{i}" for i in range(num_samples)]
    
    # Create features
    features = {
        'student_id': tf.constant(student_ids),
        'engagement_id': tf.constant(engagement_ids),
        'student_features': tf.random.uniform((num_samples, 10)),
        'engagement_features': tf.random.uniform((num_samples, 5))
    }
    
    # Create labels
    labels = {
        'ranking_label': tf.random.uniform((num_samples, 1)),
        'likelihood_label': tf.random.uniform((num_samples, 1)),
        'risk_label': tf.random.uniform((num_samples, 1))
    }
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    return dataset.batch(32)

@pytest.fixture(scope="session")
def model_config():
    """Return model configuration for testing."""
    return {
        'embedding_dimension': 64,
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 2
    }

@pytest.fixture(scope="session")
def test_data_dir():
    """Return path to test data directory."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures', 'test_data') 