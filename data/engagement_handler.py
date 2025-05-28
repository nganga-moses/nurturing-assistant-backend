import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
from .vector_store import VectorStore
import json

logger = logging.getLogger(__name__)

class DynamicEngagementHandler:
    """Handles dynamic engagement types and their embeddings."""
    
    def __init__(self, embedding_dimension: int = 64):
        self.embedding_dimension = embedding_dimension
        self.vector_store = VectorStore(dimension=embedding_dimension, index_type="IP")
        self.engagement_types = set()
        self.engagement_categories = set()
        self.engagement_subcategories = set()
        self.similarity_threshold = 0.8
    
    def add_new_engagement_type(self, new_engagement: Dict) -> None:
        """
        Add new engagement type without full retraining.
        
        Args:
            new_engagement: Dictionary containing engagement information
        """
        # Extract engagement features
        engagement_type = new_engagement["type"]
        category = new_engagement.get("category", "general")
        subcategory = new_engagement.get("subcategory", "general")
        
        # Add to sets
        self.engagement_types.add(engagement_type)
        self.engagement_categories.add(category)
        self.engagement_subcategories.add(subcategory)
        
        # Find similar engagements
        similar_engagements = self.find_similar_engagements(new_engagement)
        
        if similar_engagements:
            # Initialize embedding from similar engagements
            new_embedding = self.initialize_from_similar(
                new_engagement,
                similar_engagements
            )
        else:
            # Create new embedding
            new_embedding = self.create_new_embedding(new_engagement)
        
        # Add to vector store
        self.vector_store.add_embeddings(
            ids=[engagement_type],
            embeddings=new_embedding.reshape(1, -1),
            metadata=[{
                "category": category,
                "subcategory": subcategory,
                "created_at": datetime.now().isoformat()
            }]
        )
    
    def find_similar_engagements(self, engagement: Dict) -> List[Tuple[str, float]]:
        """
        Find similar existing engagements.
        
        Args:
            engagement: Dictionary containing engagement information
            
        Returns:
            List of (engagement_type, similarity_score) tuples
        """
        # Create feature vector for the engagement
        feature_vector = self.create_feature_vector(engagement)
        
        # Search for similar engagements
        similar_ids, scores, _ = self.vector_store.search(
            feature_vector.reshape(1, -1),
            k=5
        )
        
        # Filter by similarity threshold
        similar_engagements = [
            (id_, score) for id_, score in zip(similar_ids, scores)
            if score > self.similarity_threshold
        ]
        
        return similar_engagements
    
    def initialize_from_similar(self, new_engagement: Dict, similar_engagements: List[Tuple[str, float]]) -> np.ndarray:
        """
        Initialize new embedding using similar engagements.
        
        Args:
            new_engagement: Dictionary containing new engagement information
            similar_engagements: List of (engagement_type, similarity_score) tuples
            
        Returns:
            New embedding vector
        """
        # Get embeddings of similar engagements
        similar_embeddings = []
        weights = []
        
        for engagement_type, similarity in similar_engagements:
            embedding = self.vector_store.id_to_embedding.get(engagement_type)
            if embedding is not None:
                similar_embeddings.append(embedding)
                weights.append(similarity)
        
        if not similar_embeddings:
            return self.create_new_embedding(new_engagement)
        
        # Convert to numpy arrays
        similar_embeddings = np.array(similar_embeddings)
        weights = np.array(weights)
        
        # Normalize weights
        weights = weights / weights.sum()
        
        # Weighted average of similar embeddings
        weighted_embedding = np.sum(
            similar_embeddings * weights.reshape(-1, 1),
            axis=0
        )
        
        # Normalize the result
        weighted_embedding = weighted_embedding / np.linalg.norm(weighted_embedding)
        
        return weighted_embedding
    
    def create_new_embedding(self, engagement: Dict) -> np.ndarray:
        """
        Create new embedding from scratch.
        
        Args:
            engagement: Dictionary containing engagement information
            
        Returns:
            New embedding vector
        """
        # Create feature vector
        feature_vector = self.create_feature_vector(engagement)
        
        # Normalize
        feature_vector = feature_vector / np.linalg.norm(feature_vector)
        
        return feature_vector
    
    def create_feature_vector(self, engagement: Dict) -> np.ndarray:
        """
        Create feature vector for an engagement.
        
        Args:
            engagement: Dictionary containing engagement information
            
        Returns:
            Feature vector
        """
        # Initialize feature vector
        feature_vector = np.zeros(self.embedding_dimension)
        
        # Add type features
        type_idx = hash(engagement["type"]) % (self.embedding_dimension // 3)
        feature_vector[type_idx] = 1.0
        
        # Add category features
        category = engagement.get("category", "general")
        category_idx = hash(category) % (self.embedding_dimension // 3) + self.embedding_dimension // 3
        feature_vector[category_idx] = 1.0
        
        # Add subcategory features
        subcategory = engagement.get("subcategory", "general")
        subcategory_idx = hash(subcategory) % (self.embedding_dimension // 3) + 2 * self.embedding_dimension // 3
        feature_vector[subcategory_idx] = 1.0
        
        return feature_vector
    
    def save(self, directory: str) -> None:
        """Save the engagement handler state."""
        self.vector_store.save(directory)
        
        # Save sets
        with open(f"{directory}/engagement_sets.json", "w") as f:
            json.dump({
                "types": list(self.engagement_types),
                "categories": list(self.engagement_categories),
                "subcategories": list(self.engagement_subcategories)
            }, f)
    
    def load(self, directory: str) -> None:
        """Load the engagement handler state."""
        self.vector_store.load(directory)
        
        # Load sets
        with open(f"{directory}/engagement_sets.json", "r") as f:
            data = json.load(f)
            self.engagement_types = set(data["types"])
            self.engagement_categories = set(data["categories"])
            self.engagement_subcategories = set(data["subcategories"]) 