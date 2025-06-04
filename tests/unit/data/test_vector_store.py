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
    train_data, val_data, student_ids, engagement_ids = create_sample_data(num_students=100, num_engagements=100, batch_size=32)
    
    # Initialize vector store
    vector_store = VectorStore(embedding_dimension=64)
    
    assert vector_store is not None
    assert vector_store.embedding_dimension == 64

def test_vector_storage():
    """Test vector storage functionality."""
    # Create sample data
    train_data, val_data, student_ids, engagement_ids = create_sample_data(num_students=100, num_engagements=100, batch_size=32)
    
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
    train_data, val_data, student_ids, engagement_ids = create_sample_data(num_students=100, num_engagements=100, batch_size=32)
    
    # Initialize vector store
    vector_store = VectorStore(embedding_dimension=64)
    
    # Create and store sample vectors
    vectors = np.random.randn(100, 64)
    for i, vector in enumerate(vectors):
        vector_store.store_vector(f"item_{i}", vector)
    
    # Create query vector
    query_vector = np.random.randn(64)
    
    # Search for similar vectors
    ids, scores = vector_store.search(query_vector, k=5)
    
    # Verify results
    assert len(ids) == 5
    assert len(scores) == 5
    for id_, score in zip(ids, scores):
        assert id_ in [f"item_{i}" for i in range(100)]
        assert isinstance(score, float)

def test_vector_update():
    """Test vector update functionality."""
    # Create sample data
    train_data, val_data, student_ids, engagement_ids = create_sample_data(num_students=100, num_engagements=100, batch_size=32)
    
    # Initialize vector store
    vector_store = VectorStore(embedding_dimension=64)
    
    # Create and store sample vectors
    vectors = np.random.randn(100, 64)
    for i, vector in enumerate(vectors):
        vector_store.store_vector(f"item_{i}", vector)
    
    # Update some vectors
    new_vectors = np.random.randn(10, 64)
    for i, vector in enumerate(new_vectors):
        vector_store.store_vector(f"item_{i}", vector)
    
    # Verify updates
    for i in range(10):
        assert vector_store.get_vector(f"item_{i}").shape == (64,)

def test_batch_vector_update():
    """Test batch vector update functionality."""
    # Initialize vector store
    vector_store = VectorStore(embedding_dimension=64)
    
    # Create and store initial vectors
    initial_vectors = np.random.randn(5, 64)
    ids = [f"item_{i}" for i in range(5)]
    for id_, vector in zip(ids, initial_vectors):
        vector_store.store_vector(id_, vector)
    
    # Update vectors in batch
    updated_vectors = np.random.randn(5, 64)
    vector_store.update_vectors(ids, updated_vectors)
    
    # Verify updates
    for id_, updated_vector in zip(ids, updated_vectors):
        stored_vector = vector_store.get_vector(id_)
        assert np.allclose(stored_vector, updated_vector, rtol=1e-5, atol=1e-5)
    
    # Verify search still works
    query_vector = updated_vectors[0]  # Use first updated vector as query
    results_ids, scores = vector_store.search(query_vector, k=5)
    assert len(results_ids) == 5
    assert all(id_ in ids for id_ in results_ids)

def test_add_embeddings(sample_embeddings, sample_ids):
    """Test adding embeddings to the store."""
    store = VectorStore(dimension=3)
    store.add_embeddings(sample_ids, sample_embeddings)
    
    # Check that embeddings were added
    assert len(store.id_to_embedding) == len(sample_ids)
    assert all(id in store.id_to_embedding for id in sample_ids)
    
    # Normalize sample embeddings for comparison
    normalized_embeddings = [embedding / np.linalg.norm(embedding) for embedding in sample_embeddings]
    
    # Check that embeddings match
    for id, embedding in zip(sample_ids, normalized_embeddings):
        np.testing.assert_array_almost_equal(store.id_to_embedding[id], embedding)

def test_search(sample_embeddings, sample_ids):
    """Test similarity search."""
    store = VectorStore(dimension=3)
    store.add_embeddings(sample_ids, sample_embeddings)
    
    # Test search with k=2
    query = np.array([0.2, 0.3, 0.4])
    query = query / np.linalg.norm(query)  # Normalize query
    ids, scores = store.search(query.reshape(1, -1), k=2)
    
    assert len(ids) == 2
    assert len(scores) == 2
    assert all(id in sample_ids for id in ids)
    assert all(isinstance(score, float) for score in scores)

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
    ids, scores, metadatas = store.search_with_metadata(query.reshape(1, -1), k=2)
    
    assert len(ids) == 2
    assert len(scores) == 2
    assert len(metadatas) == 2
    assert all(id in sample_ids for id in ids)
    assert all(isinstance(score, float) for score in scores)
    
    # Filter results manually for 'type' == 'student'
    filtered_ids = [id for id, meta in zip(ids, metadatas) if meta.get("type") == "student"]
    assert all(id in sample_ids for id in filtered_ids)

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
    
    # Normalize sample embeddings for comparison
    normalized_embeddings = [embedding / np.linalg.norm(embedding) for embedding in sample_embeddings]
    
    # Check that embeddings match
    for id, embedding in zip(sample_ids, normalized_embeddings):
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
    
    # Normalize new embeddings for comparison
    normalized_new_embeddings = [embedding / np.linalg.norm(embedding) for embedding in new_embeddings]
    
    # Check that embeddings were updated
    for id, embedding in zip(new_ids, normalized_new_embeddings):
        np.testing.assert_array_almost_equal(store.id_to_embedding[id], embedding)
    
    # Normalize sample embeddings for comparison
    normalized_sample_embeddings = [embedding / np.linalg.norm(embedding) for embedding in sample_embeddings]
    
    # Check that other embeddings were not affected
    for id, embedding in zip(sample_ids[2:], normalized_sample_embeddings[2:]):
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