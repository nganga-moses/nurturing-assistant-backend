import pytest
import numpy as np
from ..data.vector_store import VectorStore

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
    # Test L2 index
    store_l2 = VectorStore(dimension=3, index_type="L2")
    assert store_l2.dimension == 3
    assert store_l2.index_type == "L2"
    
    # Test IP index
    store_ip = VectorStore(dimension=3, index_type="IP")
    assert store_ip.dimension == 3
    assert store_ip.index_type == "IP"
    
    # Test invalid index type
    with pytest.raises(ValueError):
        VectorStore(dimension=3, index_type="invalid")

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