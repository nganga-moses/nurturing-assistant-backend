import pandas as pd
import numpy as np
import pytest
from data.processing.vector_store import VectorStore
from tests.fixtures.synthetic_data import create_sample_data

@pytest.fixture
def sample_embeddings():
    """Create sample embeddings for testing."""
    return np.array([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
        [0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7]
    ])

@pytest.fixture
def sample_ids():
    """Create sample IDs for testing."""
    return ["id1", "id2", "id3", "id4", "id5"]

def test_vector_store_initialization():
    """Test vector store initialization."""
    # Create sample data
    train_data, val_data, student_ids, engagement_ids = create_sample_data(num_samples=100)
    
    # Initialize vector store
    vector_store = VectorStore(embedding_dimension=64)
    
    assert vector_store is not None
    assert vector_store.embedding_dimension == 64

def test_vector_storage():
    """Test vector storage functionality."""
    # Create sample data
    train_data, val_data, student_ids, engagement_ids = create_sample_data(num_samples=100)
    
    # Initialize vector store
    vector_store = VectorStore(embedding_dimension=64)
    
    # Create sample vectors
    vectors = np.random.randn(100, 64)
    
    # Store vectors
    for i, vector in enumerate(vectors):
        vector_store.store_vector(f"item_{i}", vector)
    
    # Verify storage
    assert len(vector_store) == 100
    for i in range(100):
        assert f"item_{i}" in vector_store
        assert vector_store.get_vector(f"item_{i}").shape == (64,)

def test_vector_search():
    """Test vector search functionality."""
    # Create sample data
    train_data, val_data, student_ids, engagement_ids = create_sample_data(num_samples=100)
    
    # Initialize vector store
    vector_store = VectorStore(embedding_dimension=64)
    
    # Create and store sample vectors
    vectors = np.random.randn(100, 64)
    for i, vector in enumerate(vectors):
        vector_store.store_vector(f"item_{i}", vector)
    
    # Create query vector
    query_vector = np.random.randn(64)
    
    # Search for similar vectors
    results = vector_store.search(query_vector, k=5)
    
    # Verify results
    assert len(results) == 5
    for result in results:
        assert 'id' in result
        assert 'score' in result
        assert 0 <= result['score'] <= 1

def test_vector_update():
    """Test vector update functionality."""
    # Create sample data
    train_data, val_data, student_ids, engagement_ids = create_sample_data(num_samples=100)
    
    # Initialize vector store
    vector_store = VectorStore(embedding_dimension=64)
    
    # Create and store initial vector
    initial_vector = np.random.randn(64)
    vector_store.store_vector("test_item", initial_vector)
    
    # Update vector
    updated_vector = np.random.randn(64)
    vector_store.update_vector("test_item", updated_vector)
    
    # Verify update
    stored_vector = vector_store.get_vector("test_item")
    assert np.array_equal(stored_vector, updated_vector)
    assert not np.array_equal(stored_vector, initial_vector)

def test_add_embeddings(sample_embeddings, sample_ids):
    """Test adding embeddings to the store."""
    store = VectorStore(dimension=3)
    store.add_embeddings(sample_ids, sample_embeddings)
    
    # Check that embeddings were added
    assert len(store.id_to_embedding) == len(sample_ids)
    assert all(id in store.id_to_embedding for id in sample_ids)
    
    # Check that embeddings match
    for id, embedding in zip(sample_ids, sample_embeddings):
        np.testing.assert_array_almost_equal(store.id_to_embedding[id], embedding)

def test_search(sample_embeddings, sample_ids):
    """Test similarity search."""
    store = VectorStore(dimension=3)
    store.add_embeddings(sample_ids, sample_embeddings)
    
    # Test search with k=2
    query = np.array([0.2, 0.3, 0.4])
    ids, scores, _ = store.search(query.reshape(1, -1), k=2)
    
    assert len(ids) == 2
    assert len(scores) == 2
    assert all(id in sample_ids for id in ids)
    assert all(0 <= score <= 1 for score in scores)

def test_search_with_metadata(sample_embeddings, sample_ids):
    """Test similarity search with metadata."""
    store = VectorStore(dimension=3)
    
    # Add embeddings with metadata
    metadata = [
        {"type": "student", "score": 0.8},
        {"type": "student", "score": 0.9},
        {"type": "engagement", "score": 0.7},
        {"type": "student", "score": 0.6},
        {"type": "engagement", "score": 0.5}
    ]
    
    store.add_embeddings(sample_ids, sample_embeddings, metadata)
    
    # Test search with metadata filter
    query = np.array([0.2, 0.3, 0.4])
    ids, scores, metadata = store.search(
        query.reshape(1, -1),
        k=2,
        metadata_filter=lambda m: m["type"] == "student"
    )
    
    assert len(ids) == 2
    assert len(scores) == 2
    assert len(metadata) == 2
    assert all(m["type"] == "student" for m in metadata)

def test_save_load(tmp_path, sample_embeddings, sample_ids):
    """Test saving and loading the vector store."""
    store = VectorStore(dimension=3)
    store.add_embeddings(sample_ids, sample_embeddings)
    
    # Save store
    store.save(str(tmp_path / "vectors"))
    
    # Load store
    new_store = VectorStore(dimension=3)
    new_store.load(str(tmp_path / "vectors"))
    
    # Check that embeddings were loaded correctly
    assert len(new_store.id_to_embedding) == len(sample_ids)
    assert all(id in new_store.id_to_embedding for id in sample_ids)
    
    # Check that embeddings match
    for id, embedding in zip(sample_ids, sample_embeddings):
        np.testing.assert_array_almost_equal(new_store.id_to_embedding[id], embedding)

def test_update_embeddings(sample_embeddings, sample_ids):
    """Test updating existing embeddings."""
    store = VectorStore(dimension=3)
    store.add_embeddings(sample_ids, sample_embeddings)
    
    # Update some embeddings
    new_embeddings = np.array([
        [0.3, 0.4, 0.5],
        [0.6, 0.7, 0.8]
    ])
    new_ids = ["id1", "id2"]
    
    store.add_embeddings(new_ids, new_embeddings)
    
    # Check that embeddings were updated
    for id, embedding in zip(new_ids, new_embeddings):
        np.testing.assert_array_almost_equal(store.id_to_embedding[id], embedding)
    
    # Check that other embeddings were not affected
    for id, embedding in zip(sample_ids[2:], sample_embeddings[2:]):
        np.testing.assert_array_almost_equal(store.id_to_embedding[id], embedding)

def test_invalid_dimension():
    """Test handling of invalid dimensions."""
    store = VectorStore(dimension=3)
    
    # Try to add embeddings with wrong dimension
    with pytest.raises(ValueError):
        store.add_embeddings(["id1"], np.array([[0.1, 0.2]]))
    
    # Try to search with wrong dimension
    with pytest.raises(ValueError):
        store.search(np.array([[0.1, 0.2]])) 