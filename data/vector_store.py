import faiss
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
import os
from datetime import datetime

class VectorStore:
    """Vector database for storing and retrieving embeddings efficiently."""
    
    def __init__(self, dimension: int = 64, index_type: str = "L2"):
        """
        Initialize the vector store.
        
        Args:
            dimension: Dimension of the embedding vectors
            index_type: Type of FAISS index to use ("L2" or "IP" for inner product)
        """
        self.dimension = dimension
        self.index_type = index_type
        self.index = self._create_index()
        self.id_to_embedding = {}
        self.id_to_metadata = {}
        
    def _create_index(self) -> faiss.Index:
        """Create a new FAISS index."""
        if self.index_type == "L2":
            return faiss.IndexFlatL2(self.dimension)
        else:  # Inner product
            return faiss.IndexFlatIP(self.dimension)
    
    def add_embeddings(self, ids: List[str], embeddings: np.ndarray, metadata: Optional[List[Dict]] = None):
        """
        Add embeddings to the index.
        
        Args:
            ids: List of IDs for the embeddings
            embeddings: Numpy array of embeddings
            metadata: Optional list of metadata dictionaries
        """
        # Normalize embeddings for inner product similarity
        if self.index_type == "IP":
            faiss.normalize_L2(embeddings)
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Store mappings
        for i, id_ in enumerate(ids):
            self.id_to_embedding[id_] = embeddings[i]
            if metadata:
                self.id_to_metadata[id_] = metadata[i]
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[List[str], List[float], List[Dict]]:
        """
        Search for similar embeddings.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            Tuple of (ids, scores, metadata)
        """
        # Normalize query for inner product similarity
        if self.index_type == "IP":
            faiss.normalize_L2(query_embedding.reshape(1, -1))
        
        # Search
        scores, indices = self.index.search(query_embedding.reshape(1, -1), k)
        
        # Convert to lists
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
    
    def save(self, directory: str):
        """Save the vector store to disk."""
        os.makedirs(directory, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(directory, "index.faiss"))
        
        # Save mappings
        with open(os.path.join(directory, "mappings.json"), "w") as f:
            json.dump({
                "id_to_embedding": {k: v.tolist() for k, v in self.id_to_embedding.items()},
                "id_to_metadata": self.id_to_metadata
            }, f)
    
    def load(self, directory: str):
        """Load the vector store from disk."""
        # Load FAISS index
        self.index = faiss.read_index(os.path.join(directory, "index.faiss"))
        
        # Load mappings
        with open(os.path.join(directory, "mappings.json"), "r") as f:
            data = json.load(f)
            self.id_to_embedding = {k: np.array(v) for k, v in data["id_to_embedding"].items()}
            self.id_to_metadata = data["id_to_metadata"] 