import faiss
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
import os
from datetime import datetime
import pickle

class VectorStore:
    """Vector database for storing and retrieving embeddings efficiently."""
    
    def __init__(self, dimension: int = 64, embedding_dimension: int = None, index_type: str = "L2"):
        """
        Initialize the vector store.
        
        Args:
            dimension: Dimension of the embedding vectors
            embedding_dimension: Alias for dimension (for backward compatibility)
            index_type: Type of FAISS index to use ("L2" or "IP" for inner product)
        """
        self.dimension = embedding_dimension if embedding_dimension is not None else dimension
        self.embedding_dimension = self.dimension  # For backward compatibility
        self.index_type = index_type
        self.index = self._create_index()
        self.id_to_embedding = {}
        self.id_to_metadata = {}
    
    def __len__(self) -> int:
        """Return the number of vectors in the store."""
        return len(self.id_to_embedding)
        
    def __contains__(self, item) -> bool:
        return item in self.id_to_embedding
    
    def _create_index(self) -> faiss.Index:
        """Create a new FAISS index."""
        if self.index_type == "L2":
            return faiss.IndexFlatL2(self.dimension)
        else:  # Inner product
            return faiss.IndexFlatIP(self.dimension)
    
    def store_vector(self, id_: str, embedding: np.ndarray, metadata: Optional[Dict] = None):
        """
        Store a single vector in the index.
        
        Args:
            id_: ID for the embedding
            embedding: Embedding vector
            metadata: Optional metadata dictionary
        """
        # Ensure embedding is float32 and has correct shape
        embedding = embedding.astype(np.float32)
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
            
        # Validate dimension
        if embedding.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension {embedding.shape[1]} does not match expected dimension {self.dimension}")
        
        # Normalize embedding for inner product similarity
        if self.index_type == "IP":
            faiss.normalize_L2(embedding)
        
        # Add to FAISS index
        self.index.add(embedding)
        
        # Store mapping
        self.id_to_embedding[id_] = embedding[0]
        if metadata:
            self.id_to_metadata[id_] = metadata
    
    def update_vector(self, id_: str, embedding: np.ndarray, metadata: Optional[Dict] = None):
        """
        Update an existing vector in the index.
        
        Note: This operation is O(n) where n is the number of vectors, as we need to rebuild the FAISS index.
        For batch updates, use update_vectors() instead.
        
        Args:
            id_: ID of the embedding to update
            embedding: New embedding vector
            metadata: Optional new metadata dictionary
        """
        if id_ not in self.id_to_embedding:
            raise KeyError(f"Vector with ID {id_} not found")
            
        # Update the mapping
        embedding = embedding.astype(np.float32)
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        if embedding.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension {embedding.shape[1]} does not match expected dimension {self.dimension}")
            
        self.id_to_embedding[id_] = embedding[0]
        if metadata:
            self.id_to_metadata[id_] = metadata
            
        # Rebuild the FAISS index with all current embeddings
        self._rebuild_index()
    
    def update_vectors(self, ids: List[str], embeddings: np.ndarray, metadata: Optional[List[Dict]] = None):
        """
        Update multiple vectors in the index efficiently.
        
        This is more efficient than calling update_vector() multiple times as it only rebuilds
        the index once for all updates.
        
        Args:
            ids: List of IDs to update
            embeddings: Array of new embedding vectors
            metadata: Optional list of metadata dictionaries
        """
        # Validate inputs
        if len(ids) != len(embeddings):
            raise ValueError("Number of IDs must match number of embeddings")
        if metadata and len(ids) != len(metadata):
            raise ValueError("Number of IDs must match number of metadata entries")
            
        # Ensure embeddings are float32 and have correct shape
        embeddings = embeddings.astype(np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension {embeddings.shape[1]} does not match expected dimension {self.dimension}")
            
        # Update mappings
        for i, id_ in enumerate(ids):
            if id_ not in self.id_to_embedding:
                raise KeyError(f"Vector with ID {id_} not found")
            self.id_to_embedding[id_] = embeddings[i]
            if metadata:
                self.id_to_metadata[id_] = metadata[i]
                
        # Rebuild the FAISS index once for all updates
        self._rebuild_index()
    
    def _rebuild_index(self):
        """
        Rebuild the FAISS index with all current embeddings.
        This is an internal method used by update operations.
        """
        self.index = self._create_index()
        all_embeddings = np.array(list(self.id_to_embedding.values()))
        if self.index_type == "IP":
            faiss.normalize_L2(all_embeddings)
        self.index.add(all_embeddings)
    
    def get_metadata(self, id_: str) -> Dict:
        """
        Get metadata for a vector.
        
        Args:
            id_: ID of the vector
            
        Returns:
            Metadata dictionary
        """
        return self.id_to_metadata.get(id_, {})
    
    def add_embeddings(self, ids: List[str], embeddings: List[np.ndarray], metadata: Optional[List[Dict]] = None) -> None:
        """Add embeddings to the store."""
        # Validate dimensions
        for embedding in embeddings:
            if embedding.shape[-1] != self.dimension:
                raise ValueError(f"Embedding dimension {embedding.shape[-1]} does not match store dimension {self.dimension}")
        
        # Convert to numpy arrays and normalize only new embeddings
        embeddings = [np.array(embedding) for embedding in embeddings]
        normalized_embeddings = [embedding / np.linalg.norm(embedding) for embedding in embeddings]
        
        # Store embeddings and metadata
        for i, id in enumerate(ids):
            self.id_to_embedding[id] = normalized_embeddings[i]
            if metadata:
                self.id_to_metadata[id] = metadata[i]
    
    def get_vector(self, id_: str) -> np.ndarray:
        """Return the stored vector for the given id."""
        return self.id_to_embedding[id_]

    def search(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[List[str], List[float]]:
        """Search for similar embeddings."""
        # Validate dimension
        if query_embedding.shape[-1] != self.dimension:
            raise ValueError(f"Query embedding dimension {query_embedding.shape[-1]} does not match store dimension {self.dimension}")
        
        # Normalize query embedding
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Calculate similarities
        similarities = []
        for id, embedding in self.id_to_embedding.items():
            similarity = np.dot(query_embedding, embedding)
            similarities.append((id, similarity))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        ids = [x[0] for x in similarities[:k]]
        scores = [float(x[1]) for x in similarities[:k]]
        
        return ids, scores
    
    def update_embeddings(self, ids: List[str], embeddings: List[np.ndarray]) -> None:
        """Update existing embeddings."""
        self.add_embeddings(ids, embeddings)

    def get_embedding(self, id: str) -> Optional[np.ndarray]:
        """Get embedding for an ID."""
        return self.id_to_embedding.get(id)

    def get_metadata(self, id: str) -> Optional[Dict]:
        """Get metadata for an ID."""
        return self.id_to_metadata.get(id)

    def search_with_metadata(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[List[str], List[float], List[Dict]]:
        """
        Search for similar embeddings and return metadata.
        Returns:
            Tuple of (ids, scores, metadata)
        """
        if len(self.id_to_embedding) == 0:
            return [], [], []
        query_embedding = query_embedding.astype(np.float32)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        if query_embedding.shape[1] != self.dimension:
            raise ValueError(f"Query dimension {query_embedding.shape[1]} does not match expected dimension {self.dimension}")
        if self.index_type == "IP":
            faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, k)
        ids = list(self.id_to_embedding.keys())
        results = []
        result_scores = []
        result_metadata = []
        for i, score in zip(indices[0], scores[0]):
            if i < len(ids):
                results.append(ids[i])
                result_scores.append(float(score))
                result_metadata.append(self.id_to_metadata.get(ids[i], {}))
        return results, result_scores, result_metadata
    
    def save(self, path: str) -> None:
        """Save vector store to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Convert embeddings to lists for serialization
        data = {
            'dimension': self.dimension,
            'id_to_embedding': {k: v.tolist() for k, v in self.id_to_embedding.items()},
            'id_to_metadata': self.id_to_metadata
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, path: str) -> None:
        """Load vector store from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.dimension = data['dimension']
        self.id_to_embedding = {k: np.array(v) for k, v in data['id_to_embedding'].items()}
        self.id_to_metadata = data['id_to_metadata'] 