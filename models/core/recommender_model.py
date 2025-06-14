import tensorflow as tf
import keras
import numpy as np
from typing import Dict, List, Optional, Text, Tuple, Any, Union


@keras.saving.register_keras_serializable(package="Custom")
class StudentTower(tf.keras.Model):
    """Tower for processing student features."""
    
    def __init__(self, embedding_dimension: int, student_vocab: Dict[str, tf.keras.layers.StringLookup] = None):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        # Don't store non-serializable objects in the model
        # Vector stores will be handled separately
        
        # Replace Sequential with explicit Dense layers for proper serialization
        self.student_dense_256 = tf.keras.layers.Dense(256, activation="relu", name='student_dense_256')
        self.student_dense_128 = tf.keras.layers.Dense(128, activation="relu", name='student_dense_128')
        self.student_dense_32 = tf.keras.layers.Dense(32, activation="relu", name='student_dense_32')
        
        # Student embedding layer
        self.student_embedding = tf.keras.layers.Dense(
            embedding_dimension, 
            activation="tanh", 
            name='student_embedding'
        )

    def get_config(self):
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({
            'embedding_dimension': self.embedding_dimension,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        """Create from configuration."""
        return cls(**config)
        
    def call(self, inputs, training=False):
        if isinstance(inputs, dict):
            student_id = inputs['student_id']
            student_features = inputs['student_features']
        else:
            student_features = inputs
            
        # Process student features through explicit Dense layers
        x = self.student_dense_256(student_features, training=training)
        x = self.student_dense_128(x, training=training)
        processed_features = self.student_dense_32(x, training=training)
        
        # Generate embedding
        embedding = self.student_embedding(processed_features, training=training)
        
        return embedding

    def build(self, input_shape):
        """Build the layers with proper input shapes and initialize weights."""
        super().build(input_shape)
        # Build the layers with feature shape
        if isinstance(input_shape, dict):
            features_shape = input_shape.get('student_features', (None, 10))
        else:
            features_shape = input_shape
        
        print(f"🔧 Building StudentTower with features shape: {features_shape}")
        
        # Build explicit Dense layers with proper shapes
        self.student_dense_256.build(features_shape)
        self.student_dense_128.build((None, 256))
        self.student_dense_32.build((None, 128))
        self.student_embedding.build((None, 32))
        
        # CRITICAL: Initialize weights by doing a forward pass with dummy data
        # This ensures the layers have actual weights to save
        import tensorflow as tf
        dummy_shape = [1 if dim is None else dim for dim in features_shape]
        dummy_input = tf.zeros(dummy_shape, dtype=tf.float32)
        
        # Force weight initialization by calling the layers
        x = self.student_dense_256(dummy_input)
        x = self.student_dense_128(x)
        x = self.student_dense_32(x)
        _ = self.student_embedding(x)
        
        print(f"✅ StudentTower build complete with weight initialization")
        print(f"   - student_dense_256 weights: {len(self.student_dense_256.weights)}")
        print(f"   - student_dense_128 weights: {len(self.student_dense_128.weights)}")
        print(f"   - student_dense_32 weights: {len(self.student_dense_32.weights)}")
        print(f"   - student_embedding weights: {len(self.student_embedding.weights)}")
    
    # Vector store operations moved to ModelTrainer
    # This keeps the tower serializable


@keras.saving.register_keras_serializable(package="Custom")
class EngagementTower(tf.keras.Model):
    """Tower for processing engagement features."""
    
    def __init__(self, embedding_dimension: int, engagement_vocab: Dict[str, tf.keras.layers.StringLookup] = None):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        # Don't store non-serializable objects in the model
        # Vector stores will be handled separately
        
        # Replace Sequential with explicit Dense layers for proper serialization
        self.engagement_dense_128 = tf.keras.layers.Dense(128, activation="relu", name='engagement_dense_128')
        self.engagement_dense_64 = tf.keras.layers.Dense(64, activation="relu", name='engagement_dense_64')
        self.engagement_dense_16 = tf.keras.layers.Dense(16, activation="relu", name='engagement_dense_16')
        
        # Engagement embedding layer
        self.engagement_embedding = tf.keras.layers.Dense(
            embedding_dimension, 
            activation="tanh", 
            name='engagement_embedding'
        )

    def get_config(self):
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({
            'embedding_dimension': self.embedding_dimension,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        """Create from configuration."""
        return cls(**config)
        
    def call(self, inputs, training=False):
        if isinstance(inputs, dict):
            engagement_id = inputs['engagement_id']
            engagement_features = inputs['engagement_features']
        else:
            engagement_features = inputs
            
        # Process engagement features through explicit Dense layers
        x = self.engagement_dense_128(engagement_features, training=training)
        x = self.engagement_dense_64(x, training=training)
        processed_features = self.engagement_dense_16(x, training=training)
        
        # Generate embedding
        embedding = self.engagement_embedding(processed_features, training=training)
        
        return embedding

    def build(self, input_shape):
        """Build the layers with proper input shapes and initialize weights."""
        super().build(input_shape)
        # Build the layers with feature shape
        if isinstance(input_shape, dict):
            features_shape = input_shape.get('engagement_features', (None, 10))
        else:
            features_shape = input_shape
        
        print(f"🔧 Building EngagementTower with features shape: {features_shape}")
        
        # Build explicit Dense layers with proper shapes
        self.engagement_dense_128.build(features_shape)
        self.engagement_dense_64.build((None, 128))
        self.engagement_dense_16.build((None, 64))
        self.engagement_embedding.build((None, 16))
        
        # CRITICAL: Initialize weights by doing a forward pass with dummy data
        # This ensures the layers have actual weights to save
        import tensorflow as tf
        dummy_shape = [1 if dim is None else dim for dim in features_shape]
        dummy_input = tf.zeros(dummy_shape, dtype=tf.float32)
        
        # Force weight initialization by calling the layers
        x = self.engagement_dense_128(dummy_input)
        x = self.engagement_dense_64(x)
        x = self.engagement_dense_16(x)
        _ = self.engagement_embedding(x)
        
        print(f"✅ EngagementTower build complete with weight initialization")
        print(f"   - engagement_dense_128 weights: {len(self.engagement_dense_128.weights)}")
        print(f"   - engagement_dense_64 weights: {len(self.engagement_dense_64.weights)}")
        print(f"   - engagement_dense_16 weights: {len(self.engagement_dense_16.weights)}")
        print(f"   - engagement_embedding weights: {len(self.engagement_embedding.weights)}")
    
    # Vector store operations moved to ModelTrainer
    # This keeps the tower serializable


@keras.saving.register_keras_serializable(package="Custom")
class RecommenderModel(tf.keras.Model):
    """Hybrid recommender model combining collaborative and content-based approaches."""
    
    def __init__(
        self,
        embedding_dimension: int = 128,
        dropout_rate: float = 0.2,
        l2_reg: float = 0.01,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Store constructor parameters for serialization
        self.embedding_dimension = embedding_dimension
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        
        # Create towers (simplified without external vocabularies)
        self.student_tower = StudentTower(embedding_dimension, {})
        self.engagement_tower = EngagementTower(embedding_dimension, {})
        
        # Student processing layers
        self.student_dense1 = tf.keras.layers.Dense(
            256, 
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
            name='student_dense1'
        )
        self.student_dense2 = tf.keras.layers.Dense(
            128,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
            name='student_dense2'
        )
        self.student_dropout = tf.keras.layers.Dropout(dropout_rate, name='student_dropout')
        
        # Engagement processing layers
        self.engagement_dense1 = tf.keras.layers.Dense(
            256,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
            name='engagement_dense1'
        )
        self.engagement_dense2 = tf.keras.layers.Dense(
            128,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
            name='engagement_dense2'
        )
        self.engagement_dropout = tf.keras.layers.Dropout(dropout_rate, name='engagement_dropout')
        
        # FIXED: Add missing output heads
        self.ranking_head = tf.keras.layers.Dense(
            1,
            activation='sigmoid',
            name='ranking_head'
        )
        self.likelihood_head = tf.keras.layers.Dense(
            1,
            activation='sigmoid',
            name='likelihood_head'
        )
        self.risk_head = tf.keras.layers.Dense(
            1,
            activation='sigmoid',
            name='risk_head'
        )

    def get_config(self):
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({
            'embedding_dimension': self.embedding_dimension,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        """Create from configuration."""
        return cls(**config)

    def call(self, inputs, training=False):
        """Forward pass through the model."""
        # Get student and engagement features
        student_features = inputs['student_features']
        engagement_features = inputs['engagement_features']
        
        # Get embeddings from towers
        student_embeddings = self.student_tower(student_features)
        engagement_embeddings = self.engagement_tower(engagement_features)
        
        # Apply additional layers with dropout during training
        student_embeddings = self.student_dense1(student_embeddings)
        student_embeddings = self.student_dropout(student_embeddings, training=training)
        student_embeddings = self.student_dense2(student_embeddings)
        
        engagement_embeddings = self.engagement_dense1(engagement_embeddings)
        engagement_embeddings = self.engagement_dropout(engagement_embeddings, training=training)
        engagement_embeddings = self.engagement_dense2(engagement_embeddings)
        
        # Compute similarity scores
        similarity_scores = tf.reduce_sum(
            student_embeddings * engagement_embeddings,
            axis=1,
            keepdims=True
        )
        
        # Generate predictions from each head
        ranking_score = self.ranking_head(similarity_scores)
        likelihood_score = self.likelihood_head(similarity_scores)
        risk_score = self.risk_head(similarity_scores)
        
        return {
            'ranking_score': ranking_score,
            'likelihood_score': likelihood_score,
            'risk_score': risk_score
        }
    
    def build(self, input_shape):
        """Build the model with the given input shape."""
        super().build(input_shape)
        
        print(f"🔧 Building RecommenderModel with input shapes: {input_shape}")
        
        # Build towers first
        print("🔧 Building student tower...")
        self.student_tower.build(input_shape['student_features'])
        print("🔧 Building engagement tower...")
        self.engagement_tower.build(input_shape['engagement_features'])
        
        # Build processing layers - towers output embedding_dimension
        batch_size = input_shape['student_features'][0]
        tower_output_shape = (batch_size, self.embedding_dimension)
        
        print(f"🔧 Building processing layers with tower output shape: {tower_output_shape}")
        self.student_dense1.build(tower_output_shape)
        self.student_dense2.build((batch_size, 256))  # After student_dense1
        
        self.engagement_dense1.build(tower_output_shape)
        self.engagement_dense2.build((batch_size, 256))  # After engagement_dense1
        
        # Build output heads - they take similarity scores (shape: batch_size, 1)
        head_input_shape = (batch_size, 1)
        print(f"🔧 Building output heads with input shape: {head_input_shape}")
        self.ranking_head.build(head_input_shape)
        self.likelihood_head.build(head_input_shape)
        self.risk_head.build(head_input_shape)
        
        print("✅ RecommenderModel build complete")

    def create_vector_stores(self):
        """Create vector stores after model is trained (not during init)."""
        # For now, we'll skip the vector store creation in the simplified model
        # This functionality will be handled by the ModelManager
        pass
        
    def update_vector_stores(self, student_data, engagement_data):
        """Update vector stores - only call this after create_vector_stores()."""
        # For now, we'll skip the vector store updates in the simplified model
        # This functionality will be handled by the ModelManager
        pass


class ModelTrainer:
    """Class for training and evaluating the student engagement model."""
    
    def __init__(self, data_dict, embedding_dimension=64):
        self.train_dataset = data_dict['train_dataset']
        self.test_dataset = data_dict['test_dataset']
        self.vocabularies = data_dict['vocabularies']
        self.dataframes = data_dict['dataframes']
        self.embedding_dimension = embedding_dimension
        
        # Create model
        self.model = RecommenderModel(
            embedding_dimension=embedding_dimension,
            dropout_rate=0.2,
            l2_reg=0.01
        )
        
        # Define optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    def train(self, epochs=5, batch_size=32):
        """Train the model."""
        # Build the model properly before training
        print("🔧 Building model layers explicitly...")
        
        # Get a sample batch to build the model
        sample_batch = next(iter(self.train_dataset))
        sample_inputs, sample_targets = sample_batch
        
        print(f"Sample input shapes:")
        print(f"  student_features: {sample_inputs['student_features'].shape}")
        print(f"  engagement_features: {sample_inputs['engagement_features'].shape}")
        
        # Build the model with correct input specification
        input_shape = {
            'student_features': sample_inputs['student_features'].shape,
            'engagement_features': sample_inputs['engagement_features'].shape
        }
        
        # Build the model explicitly
        self.model.build(input_shape)
        
        # Do a forward pass to ensure everything is connected properly
        print("🔧 Testing forward pass...")
        test_output = self.model(sample_inputs, training=False)
        print(f"✅ Forward pass successful! Output shapes:")
        for key, value in test_output.items():
            print(f"  {key}: {value.shape}")
        
        print("✅ Model layers built successfully")
        
        # Compile model with loss functions for each head
        self.model.compile(
            optimizer=self.optimizer,
            loss={
                'ranking_head': tf.keras.losses.BinaryCrossentropy(),
                'likelihood_head': tf.keras.losses.BinaryCrossentropy(),
                'risk_head': tf.keras.losses.BinaryCrossentropy()
            },
            metrics={
                'ranking_head': ['accuracy'],
                'likelihood_head': ['accuracy'],
                'risk_head': ['accuracy']
            }
        )
        
        # Train model
        history = self.model.fit(
            self.train_dataset,
            epochs=epochs,
            validation_data=self.test_dataset
        )
        
        return history
    
    def evaluate(self):
        """Evaluate the model."""
        return self.model.evaluate(self.test_dataset, return_dict=True)
    
    def save_model(self, model_dir: str = "models/saved_models") -> None:
        """
        Save the trained model using proper Keras serialization.
        
        Args:
            model_dir: Directory to save the model
        """
        import os
        import json
        
        # Ensure the model directory exists
        model_path = os.path.join(model_dir, "recommender_model")
        os.makedirs(model_path, exist_ok=True)
        
        # Save the full model using Keras format
        model_file = os.path.join(model_path, "recommender_model.keras")
        self.model.save(model_file)
        
        # Save vocabularies
        vocab_file = os.path.join(model_path, "vocabularies.json") 
        with open(vocab_file, "w") as f:
            json.dump(self.vocabularies, f)
        
        print(f"✅ Model saved to {model_file}")
        print(f"✅ Vocabularies saved to {vocab_file}")
    
    def load_model(self, model_dir="models/saved_models"):
        """Load a trained model."""
        import os
        import json
        
        model_file = os.path.join(model_dir, "recommender_model", "recommender_model.keras")
        vocab_file = os.path.join(model_dir, "recommender_model", "vocabularies.json")
        
        # Load model with custom objects
        custom_objects = {
            'RecommenderModel': RecommenderModel,
            'StudentTower': StudentTower,
            'EngagementTower': EngagementTower
        }
        
        self.model = tf.keras.models.load_model(model_file, custom_objects=custom_objects)
        
        # Load vocabularies
        with open(vocab_file, "r") as f:
            self.vocabularies = json.load(f)
        
        return self.model
