import pandas as pd
import numpy as np
import pytest
from data.processing.engagement_handler import DynamicEngagementHandler
from data.processing.feature_engineering import AdvancedFeatureEngineering
from tests.fixtures.synthetic_data import create_sample_data

@pytest.fixture
def sample_engagements():
    """Create sample engagements for testing."""
    return [
        {
            "type": "campus_visit",
            "features": {
                "duration": 120,
                "attendance": 50,
                "satisfaction": 0.8
            }
        },
        {
            "type": "virtual_tour",
            "features": {
                "duration": 45,
                "attendance": 100,
                "satisfaction": 0.7
            }
        },
        {
            "type": "info_session",
            "features": {
                "duration": 90,
                "attendance": 75,
                "satisfaction": 0.85
            }
        }
    ]

@pytest.fixture
def new_engagement():
    """Create a new engagement type for testing."""
    return {
        "type": "open_house",
        "features": {
            "duration": 180,
            "attendance": 200,
            "satisfaction": 0.9
        }
    }

def test_engagement_handler_initialization():
    """Test engagement handler initialization."""
    # Create sample data
    train_data, val_data, student_ids, engagement_ids = create_sample_data(num_samples=100)
    
    # Initialize engagement handler
    handler = DynamicEngagementHandler()
    
    assert handler is not None
    assert hasattr(handler, 'process_engagement')

def test_engagement_processing():
    """Test engagement processing functionality."""
    # Create sample data
    train_data, val_data, student_ids, engagement_ids = create_sample_data(num_samples=100)
    
    # Initialize engagement handler
    handler = DynamicEngagementHandler()
    
    # Process engagement
    result = handler.process_engagement(train_data)
    
    # Check result structure
    assert 'engagement_id' in result
    assert 'student_id' in result
    assert 'engagement_type' in result
    assert 'engagement_response' in result
    assert 'engagement_metrics' in result
    assert 'funnel_stage_before' in result
    assert 'funnel_stage_after' in result

def test_engagement_feature_engineering():
    """Test engagement feature engineering."""
    # Create sample data
    train_data, val_data, student_ids, engagement_ids = create_sample_data(num_samples=100)
    
    # Initialize feature engineering
    feature_engineering = AdvancedFeatureEngineering(
        student_data=train_data,
        engagement_data=train_data
    )
    
    # Create features
    features = feature_engineering.create_engagement_features()
    
    # Check feature structure
    assert 'engagement_sequence' in features
    assert 'engagement_velocity' in features
    assert 'engagement_quality' in features
    assert 'engagement_diversity' in features
    
    # Verify feature calculations
    for feature in features.values():
        assert not np.isnan(feature).any()
        assert not np.isinf(feature).any()

def test_engagement_metrics_calculation():
    """Test engagement metrics calculation."""
    # Create sample data
    train_data, val_data, student_ids, engagement_ids = create_sample_data(num_samples=100)
    
    # Initialize engagement handler
    handler = DynamicEngagementHandler()
    
    # Calculate metrics
    metrics = handler.calculate_engagement_metrics(train_data)
    
    # Check metrics structure
    assert 'response_rate' in metrics
    assert 'engagement_duration' in metrics
    assert 'interaction_quality' in metrics
    assert 'funnel_progression' in metrics
    
    # Verify metric ranges
    for metric in metrics.values():
        assert 0 <= metric <= 1

def test_add_engagement_type(sample_engagements):
    """Test adding engagement types."""
    handler = DynamicEngagementHandler(embedding_dimension=3)
    
    # Add sample engagements
    for engagement in sample_engagements:
        handler.add_engagement_type(engagement)
    
    # Check that engagements were added
    assert len(handler.vector_store.id_to_embedding) == len(sample_engagements)
    assert all(engagement["type"] in handler.vector_store.id_to_embedding 
              for engagement in sample_engagements)

def test_create_new_embedding(sample_engagements, new_engagement):
    """Test creating new engagement embeddings."""
    handler = DynamicEngagementHandler(embedding_dimension=3)
    
    # Add sample engagements
    for engagement in sample_engagements:
        handler.add_engagement_type(engagement)
    
    # Create new embedding
    embedding = handler.create_new_embedding(new_engagement)
    
    assert embedding.shape == (3,)
    assert not np.isnan(embedding).any()

def test_find_similar_engagements(sample_engagements, new_engagement):
    """Test finding similar engagements."""
    handler = DynamicEngagementHandler(embedding_dimension=3)
    
    # Add sample engagements
    for engagement in sample_engagements:
        handler.add_engagement_type(engagement)
    
    # Find similar engagements
    similar_engagements = handler.find_similar_engagements(new_engagement, k=2)
    
    assert len(similar_engagements) == 2
    assert all(isinstance(engagement, dict) for engagement in similar_engagements)
    assert all("type" in engagement for engagement in similar_engagements)
    assert all("similarity" in engagement for engagement in similar_engagements)

def test_update_engagement_hierarchy(sample_engagements):
    """Test updating engagement hierarchy."""
    handler = DynamicEngagementHandler(embedding_dimension=3)
    
    # Add sample engagements
    for engagement in sample_engagements:
        handler.add_engagement_type(engagement)
    
    # Update hierarchy
    hierarchy = {
        "campus_visit": ["virtual_tour", "info_session"],
        "virtual_tour": ["info_session"]
    }
    handler.update_engagement_hierarchy(hierarchy)
    
    assert handler.engagement_hierarchy == hierarchy

def test_get_engagement_path(sample_engagements):
    """Test getting engagement path."""
    handler = DynamicEngagementHandler(embedding_dimension=3)
    
    # Add sample engagements
    for engagement in sample_engagements:
        handler.add_engagement_type(engagement)
    
    # Set up hierarchy
    hierarchy = {
        "campus_visit": ["virtual_tour", "info_session"],
        "virtual_tour": ["info_session"]
    }
    handler.update_engagement_hierarchy(hierarchy)
    
    # Get path
    path = handler.get_engagement_path("campus_visit", "info_session")
    
    assert len(path) == 2
    assert path[0] == "campus_visit"
    assert path[1] == "info_session"

def test_save_load(tmp_path, sample_engagements):
    """Test saving and loading the engagement handler."""
    handler = DynamicEngagementHandler(embedding_dimension=3)
    
    # Add sample engagements
    for engagement in sample_engagements:
        handler.add_engagement_type(engagement)
    
    # Save handler
    handler.save(str(tmp_path / "handler"))
    
    # Load handler
    new_handler = DynamicEngagementHandler(embedding_dimension=3)
    new_handler.load(str(tmp_path / "handler"))
    
    # Check that engagements were loaded correctly
    assert len(new_handler.vector_store.id_to_embedding) == len(sample_engagements)
    assert all(engagement["type"] in new_handler.vector_store.id_to_embedding 
              for engagement in sample_engagements)

def test_invalid_engagement_type():
    """Test handling of invalid engagement types."""
    handler = DynamicEngagementHandler(embedding_dimension=3)
    
    # Try to add engagement without type
    with pytest.raises(ValueError):
        handler.add_engagement_type({"features": {"duration": 120}})
    
    # Try to find similar engagements for invalid type
    with pytest.raises(ValueError):
        handler.find_similar_engagements({"type": "invalid"})

def test_engagement_metadata(sample_engagements):
    """Test handling of engagement metadata."""
    handler = DynamicEngagementHandler(embedding_dimension=3)
    
    # Add sample engagements with metadata
    for engagement in sample_engagements:
        engagement["metadata"] = {
            "category": "campus",
            "priority": "high"
        }
        handler.add_engagement_type(engagement)
    
    # Check that metadata was stored
    for engagement in sample_engagements:
        assert engagement["type"] in handler.vector_store.id_to_metadata
        assert handler.vector_store.id_to_metadata[engagement["type"]] == engagement["metadata"] 