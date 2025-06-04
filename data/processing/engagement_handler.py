import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Set
import logging
from datetime import datetime
from .vector_store import VectorStore
import json
import pandas as pd
import os
import pickle

logger = logging.getLogger(__name__)

class DynamicEngagementHandler:
    """Handler for processing and managing student engagements."""
    
    VALID_DIMENSIONS = {"academic", "social", "financial", "campus_life", "career", "unknown"}
    
    def __init__(self, embedding_dimension: int = 8):
        self.embedding_dimension = embedding_dimension
        self.vector_store = VectorStore(dimension=embedding_dimension)
        self.engagement_hierarchy = {}
        self.engagement_metrics = {}
        self.engagement_types = set()
        self.engagement_categories = set()
        self.engagement_subcategories = set()
    
    def process_engagement(self, engagement: Dict[str, Any]) -> Dict[str, Any]:
        """Process an engagement event."""
        if 'type' not in engagement:
            raise ValueError("Engagement type is required")

        # Extract features
        features = self._extract_features(engagement)
        
        # Calculate metrics
        metrics = self.calculate_engagement_metrics(pd.DataFrame([engagement]))
        
        # Validate dimensions
        dimensions = self.validate_dimensions(engagement.get('features', {}).get('dimensions', []))
        
        # Prepare metadata
        metadata = engagement.get('metadata', {})
        
        # Update vector store
        self.vector_store.add_embeddings(
            ids=[engagement['type']],
            embeddings=[features],
            metadata=[metadata]
        )
        
        # Add to engagement types
        self.engagement_types.add(engagement['type'])
        if 'category' in engagement:
            self.engagement_categories.add(engagement['category'])
        if 'subcategory' in engagement:
            self.engagement_subcategories.add(engagement['subcategory'])
        
        # Convert features to dict with dimensions
        feature_dict = {
            'vector': features.tolist(),
            'dimensions': dimensions
        }
        
        return {
            'type': engagement['type'],
            'features': feature_dict,
            'metrics': metrics,
            'engagement_id': f"{engagement['type']}_{datetime.now().timestamp()}"
        }
    
    def calculate_engagement_metrics(self, engagements: Union[pd.DataFrame, Dict[str, Any]]) -> Dict[str, float]:
        """Calculate engagement metrics."""
        metrics = {
            "response_rate": 0.0,
            "engagement_duration": 0.0,
            "interaction_quality": 0.0,
            "funnel_progression": 0.0,
            "dimension_coverage": 0.0
        }
        
        if isinstance(engagements, dict):
            features = engagements.get('features', {})
            metrics['engagement_duration'] = min(features.get('duration', 0.0) / 3600, 1.0)
            metrics['interaction_quality'] = features.get('satisfaction', 0.0)
            metrics['funnel_progression'] = features.get('completion_rate', 0.0)
            metrics['response_rate'] = features.get('response_rate', 0.0)
            metrics['dimension_coverage'] = len(features.get('dimensions', [])) / len(self.VALID_DIMENSIONS)
            return metrics
            
        if len(engagements) == 0:
            return metrics
            
        # Calculate response rate
        if 'engagement_response' in engagements.columns:
            metrics['response_rate'] = engagements['engagement_response'].mean()
            
        # Calculate engagement duration
        if 'duration' in engagements.columns:
            metrics['engagement_duration'] = engagements['duration'].apply(
                lambda x: min(x / 3600, 1.0)
            ).mean()
            
        # Calculate interaction quality
        if 'satisfaction' in engagements.columns:
            metrics['interaction_quality'] = engagements['satisfaction'].mean()
            
        # Calculate funnel progression
        if 'completed' in engagements.columns:
            metrics['funnel_progression'] = engagements['completed'].mean()
            
        # Calculate dimension coverage
        if 'dimensions' in engagements.columns:
            metrics['dimension_coverage'] = engagements['dimensions'].apply(
                lambda x: len(x) / len(self.VALID_DIMENSIONS) if isinstance(x, list) else 0.0
            ).mean()
            
        return metrics
    
    def validate_dimensions(self, dimensions: List[str]) -> List[str]:
        """Validate and filter dimensions."""
        if not isinstance(dimensions, list):
            return []
        return [dim if dim in self.VALID_DIMENSIONS else "unknown" for dim in dimensions]
    
    def update_engagement_hierarchy(self, hierarchy: Dict[str, List[str]]) -> None:
        """Update engagement hierarchy."""
        self.engagement_hierarchy = hierarchy
    
    def get_engagement_path(self, start_type: str, end_type: str) -> List[str]:
        """Get the path between two engagement types."""
        if start_type not in self.engagement_types or end_type not in self.engagement_types:
            return []
        
        # Use breadth-first search to find the shortest path
        queue = [(start_type, [start_type])]
        visited = {start_type}
        
        while queue:
            current, path = queue.pop(0)
            if current == end_type:
                return path
            
            # Get next engagement types from hierarchy
            next_types = self.engagement_hierarchy.get(current, [])
            for next_type in next_types:
                if next_type not in visited:
                    visited.add(next_type)
                    queue.append((next_type, path + [next_type]))
        
        return []
    
    def save(self, path: str) -> None:
        """Save handler state to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save vector store
        vector_store_path = os.path.join(os.path.dirname(path), 'vector_store.pkl')
        self.vector_store.save(vector_store_path)
        
        # Save engagement sets
        data = {
            'engagement_types': list(self.engagement_types),
            'engagement_categories': list(self.engagement_categories),
            'engagement_subcategories': list(self.engagement_subcategories),
            'engagement_hierarchy': self.engagement_hierarchy
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, path: str) -> None:
        """Load handler state from disk."""
        # Load vector store
        vector_store_path = os.path.join(os.path.dirname(path), 'vector_store.pkl')
        self.vector_store.load(vector_store_path)
        
        # Load engagement sets
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.engagement_types = set(data['engagement_types'])
        self.engagement_categories = set(data['engagement_categories'])
        self.engagement_subcategories = set(data['engagement_subcategories'])
        self.engagement_hierarchy = data['engagement_hierarchy']
    
    def add_new_engagement_type(self, new_engagement: Dict[str, Any]) -> None:
        """Add a new engagement type."""
        if 'type' not in new_engagement:
            raise ValueError("Engagement type is required")

        # Extract and validate features
        features = new_engagement.get('features', {})
        dimensions = self.validate_dimensions(features.get('dimensions', []))

        # Store only the metadata field
        metadata = new_engagement.get('metadata', {})

        # Create feature vector
        feature_vector = self.create_feature_vector(features)

        # Store in vector store
        self.vector_store.store_vector(new_engagement['type'], feature_vector, metadata)

        # Add to engagement types
        self.engagement_types.add(new_engagement['type'])

        # Add to categories and subcategories
        if 'category' in new_engagement:
            self.engagement_categories.add(new_engagement['category'])
        if 'subcategory' in new_engagement:
            self.engagement_subcategories.add(new_engagement['subcategory'])
    
    def _extract_features(self, engagement: Dict[str, Any]) -> np.ndarray:
        """Extract features from engagement data."""
        features = np.zeros(self.embedding_dimension)
        
        # Map engagement features to embedding dimensions
        feature_mapping = {
            'duration': 0,
            'satisfaction': 1,
            'attendance': 2,
            'completion_rate': 3,
            'response_rate': 4
        }
        
        for feature, idx in feature_mapping.items():
            if feature in engagement.get('features', {}):
                features[idx] = engagement['features'][feature]
            elif feature in engagement:
                features[idx] = engagement[feature]
                
        # Normalize features
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
            
        return features
    
    def create_feature_vector(self, engagement: Dict[str, Any]) -> np.ndarray:
        """Create a feature vector from engagement data."""
        # Initialize feature vector
        features = np.zeros(self.embedding_dimension)
        
        # Map engagement features to specific indices
        feature_mapping = {
            'type': 0,
            'category': 1,
            'subcategory': 2
        }
        
        # Set type, category, and subcategory if present
        if 'type' in engagement:
            features[feature_mapping['type']] = 1.0
        if 'category' in engagement:
            features[feature_mapping['category']] = 1.0
        if 'subcategory' in engagement:
            features[feature_mapping['subcategory']] = 1.0
        
        # Extract features from engagement
        engagement_features = engagement.get('features', {})
        if isinstance(engagement_features, dict):
            # Map duration, attendance, and satisfaction to remaining dimensions
            if 'duration' in engagement_features and feature_mapping['subcategory'] + 1 < self.embedding_dimension:
                features[feature_mapping['subcategory'] + 1] = min(engagement_features['duration'] / 3600, 1.0)
            if 'attendance' in engagement_features and feature_mapping['subcategory'] + 2 < self.embedding_dimension:
                features[feature_mapping['subcategory'] + 2] = min(engagement_features['attendance'] / 100, 1.0)
            if 'satisfaction' in engagement_features and feature_mapping['subcategory'] + 3 < self.embedding_dimension:
                features[feature_mapping['subcategory'] + 3] = engagement_features['satisfaction']
        
        # Normalize feature vector
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        else:
            features[0] = 1.0  # Set default value if all features are zero
        
        return features
    
    def find_similar_engagements(self, engagement: Dict[str, Any], k: int = 5) -> List[Dict[str, Any]]:
        """Find similar engagements."""
        if 'type' not in engagement:
            raise ValueError("Engagement type is required")
            
        if engagement['type'] not in self.engagement_types:
            raise ValueError(f"Invalid engagement type: {engagement['type']}")
            
        features = self._extract_features(engagement)
        ids, scores = self.vector_store.search(features, k=k)
        
        results = []
        for id, score in zip(ids, scores):
            results.append({
                'type': id,
                'similarity_score': float(score)
            })
            
        return results
    
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