import pandas as pd
import numpy as np
import pytest
from data.processing.engagement_handler import DynamicEngagementHandler
from data.processing.feature_engineering import AdvancedFeatureEngineering
from tests.fixtures.synthetic_data import create_sample_data
from datetime import datetime

@pytest.fixture
def sample_engagements():
    """Create sample engagements for testing."""
    return [
        {
            "type": "campus_visit",
            "features": {
                "duration": 120,
                "attendance": 50,
                "satisfaction": 0.8,
                "dimensions": ["academic", "social"]
            }
        },
        {
            "type": "virtual_tour",
            "features": {
                "duration": 45,
                "attendance": 100,
                "satisfaction": 0.7,
                "dimensions": ["academic"]
            }
        },
        {
            "type": "info_session",
            "features": {
                "duration": 90,
                "attendance": 75,
                "satisfaction": 0.85,
                "dimensions": ["academic", "social", "financial"]
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
            "satisfaction": 0.9,
            "dimensions": ["academic", "social"]
        }
    }

def test_engagement_handler_initialization():
    """Test engagement handler initialization."""
    handler = DynamicEngagementHandler()
    assert handler is not None
    assert hasattr(handler, 'process_engagement')
    assert hasattr(handler, 'vector_store')

def test_engagement_processing():
    """Test engagement processing functionality."""
    handler = DynamicEngagementHandler()
    
    # Create a sample engagement
    engagement = {
        "type": "campus_visit",
        "features": {
            "duration": 120,
            "attendance": 50,
            "satisfaction": 0.8,
            "dimensions": ["academic", "social"]
        }
    }
    
    # Process engagement
    result = handler.process_engagement(engagement)
    
    # Check result structure
    assert isinstance(result, dict)
    assert "engagement_id" in result
    assert "features" in result
    assert "dimensions" in result["features"]
    assert isinstance(result["features"]["dimensions"], list)

def test_engagement_feature_engineering():
    """Test engagement feature engineering."""
    # Create sample data
    train_data, val_data, student_ids, engagement_ids = create_sample_data(num_students=10, num_engagements=10, batch_size=32)
    
    # Initialize feature engineering with sample data
    feature_engineering = AdvancedFeatureEngineering(student_data=train_data, engagement_data=val_data)
    
    # Create sample engagement
    engagement = {
        "type": "campus_visit",
        "features": {
            "duration": 120,
            "attendance": 50,
            "satisfaction": 0.8,
            "dimensions": ["academic", "social"]
        }
    }
    
    # Create features
    features = feature_engineering.create_all_features()
    
    # Check feature structure
    assert isinstance(features, pd.DataFrame)
    assert not features.empty
    assert "engagement_velocity" in features.columns
    assert "velocity_std" in features.columns
    assert "max_stage_reached" in features.columns
    assert "stage_progression_speed" in features.columns
    assert "regression_count" in features.columns
    assert "avg_time_spent" in features.columns
    assert "avg_scroll_depth" in features.columns
    assert "avg_interaction_count" in features.columns
    assert "completion_rate" in features.columns
    assert "academic_engagement_ratio" in features.columns
    assert "faculty_interaction_count" in features.columns
    assert "department_visit_count" in features.columns
    assert "avg_academic_time_spent" in features.columns
    
    # Verify feature calculations
    for column in features.columns:
        assert not features[column].isna().any()
        assert not np.isinf(features[column]).any()

def test_engagement_metrics_calculation():
    """Test engagement metrics calculation."""
    handler = DynamicEngagementHandler()
    
    # Create sample engagement
    engagement = {
        "type": "campus_visit",
        "features": {
            "duration": 120,
            "attendance": 50,
            "satisfaction": 0.8,
            "dimensions": ["academic", "social"]
        }
    }
    
    # Calculate metrics
    metrics = handler.calculate_engagement_metrics(engagement)
    
    # Check metrics structure
    assert isinstance(metrics, dict)
    assert "response_rate" in metrics
    assert "engagement_duration" in metrics
    assert "interaction_quality" in metrics
    assert "funnel_progression" in metrics
    assert "dimension_coverage" in metrics
    
    # Verify metric ranges
    for metric in metrics.values():
        if isinstance(metric, (int, float)):
            assert 0 <= metric <= 1

def test_add_engagement_type(sample_engagements):
    """Test adding engagement types."""
    handler = DynamicEngagementHandler(embedding_dimension=3)
    
    # Add sample engagements
    for engagement in sample_engagements:
        handler.add_new_engagement_type(engagement)
    
    # Check that engagements were added
    assert len(handler.vector_store.id_to_embedding) == len(sample_engagements)
    assert all(engagement["type"] in handler.vector_store.id_to_embedding 
              for engagement in sample_engagements)

def test_create_new_embedding(sample_engagements, new_engagement):
    """Test creating new engagement embeddings."""
    handler = DynamicEngagementHandler(embedding_dimension=3)
    
    # Add sample engagements
    for engagement in sample_engagements:
        handler.add_new_engagement_type(engagement)
    
    # Create new embedding
    embedding = handler.create_new_embedding(new_engagement)
    
    assert embedding.shape == (3,)
    assert not np.isnan(embedding).any()

def test_find_similar_engagements(sample_engagements, new_engagement):
    """Test finding similar engagements."""
    handler = DynamicEngagementHandler(embedding_dimension=3)
    
    # Add sample engagements
    for engagement in sample_engagements:
        handler.add_new_engagement_type(engagement)
    
    # Find similar engagements
    similar_engagements = handler.find_similar_engagements(new_engagement, k=2)
    
    assert len(similar_engagements) == 2
    assert all(isinstance(engagement, dict) for engagement in similar_engagements)
    assert all("type" in engagement for engagement in similar_engagements)
    assert all("similarity" in engagement for engagement in similar_engagements)

def test_handle_invalid_dimensions():
    """Test handling of invalid dimensions."""
    handler = DynamicEngagementHandler()
    
    # Create engagement with invalid dimension
    engagement = {
        "type": "test",
        "features": {
            "duration": 120,
            "dimensions": ["invalid_dimension"]
        }
    }
    
    # Process engagement
    result = handler.process_engagement(engagement)
    
    # Check that invalid dimension was handled
    assert "dimensions" in result["features"]
    assert "invalid_dimension" not in result["features"]["dimensions"]
    assert "unknown" in result["features"]["dimensions"]

def test_dimension_validation():
    """Test dimension validation."""
    handler = DynamicEngagementHandler()
    
    # Test with valid dimensions
    valid_dimensions = ["academic", "social", "financial"]
    assert handler.validate_dimensions(valid_dimensions) == valid_dimensions
    
    # Test with invalid dimensions
    invalid_dimensions = ["invalid1", "academic", "invalid2"]
    validated = handler.validate_dimensions(invalid_dimensions)
    assert "academic" in validated
    assert "invalid1" not in validated
    assert "invalid2" not in validated
    assert "unknown" in validated

def test_feature_vector_creation():
    """Test feature vector creation."""
    handler = DynamicEngagementHandler()
    
    # Create sample engagement
    engagement = {
        "type": "campus_visit",
        "features": {
            "duration": 120,
            "attendance": 50,
            "satisfaction": 0.8,
            "dimensions": ["academic", "social"]
        }
    }
    
    # Create feature vector
    vector = handler.create_feature_vector(engagement)
    
    assert isinstance(vector, np.ndarray)
    assert not np.isnan(vector).any()
    assert not np.isinf(vector).any()

def test_update_engagement_hierarchy(sample_engagements):
    """Test updating engagement hierarchy."""
    handler = DynamicEngagementHandler(embedding_dimension=3)
    
    # Add sample engagements
    for engagement in sample_engagements:
        handler.add_new_engagement_type(engagement)
    
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
        handler.add_new_engagement_type(engagement)
    
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
        handler.add_new_engagement_type(engagement)
    
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
        handler.add_new_engagement_type({"features": {"duration": 120}})
    
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
        handler.add_new_engagement_type(engagement)
    
    # Check that metadata was stored
    for engagement in sample_engagements:
        assert engagement["type"] in handler.vector_store.id_to_metadata
        assert handler.vector_store.id_to_metadata[engagement["type"]] == engagement["metadata"]

def test_add_new_engagement_type():
    handler = DynamicEngagementHandler(embedding_dimension=8)
    engagement = {
        "type": "webinar",
        "category": "virtual",
        "subcategory": "info"
    }
    handler.add_new_engagement_type(engagement)
    assert "webinar" in handler.engagement_types
    # Should be in vector store
    assert "webinar" in handler.vector_store.id_to_embedding

def test_find_similar_engagements():
    handler = DynamicEngagementHandler(embedding_dimension=8)
    engagement1 = {"type": "webinar", "category": "virtual", "subcategory": "info"}
    engagement2 = {"type": "seminar", "category": "virtual", "subcategory": "info"}
    handler.add_new_engagement_type(engagement1)
    handler.add_new_engagement_type(engagement2)
    similar = handler.find_similar_engagements(engagement2)
    assert isinstance(similar, list)

def test_create_new_embedding():
    handler = DynamicEngagementHandler(embedding_dimension=8)
    engagement = {"type": "workshop", "category": "in-person", "subcategory": "training"}
    embedding = handler.create_new_embedding(engagement)
    assert embedding.shape == (8,)
    assert np.isclose(np.linalg.norm(embedding), 1.0)

def test_create_feature_vector():
    handler = DynamicEngagementHandler(embedding_dimension=9)
    engagement = {"type": "lecture", "category": "academic", "subcategory": "core"}
    vec = handler.create_feature_vector(engagement)
    assert vec.shape == (9,)
    assert np.count_nonzero(vec) == 3

def test_feature_engineering_create_all_features():
    train_data, val_data, student_ids, engagement_ids = create_sample_data(num_students=10, num_engagements=10, batch_size=32)
    # Use val_data as engagement data
    fe = AdvancedFeatureEngineering(student_data=train_data, engagement_data=val_data)
    features = fe.create_all_features()
    assert isinstance(features, pd.DataFrame)
    assert not features.empty 