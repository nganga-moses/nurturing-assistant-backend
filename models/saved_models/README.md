# Saved Models Directory

This directory contains serialized models and their metadata.

## Directory Structure

```
saved_models/
├── collaborative/    # Collaborative filtering models
├── content_based/    # Content-based models
├── hybrid/          # Hybrid recommendation models
└── metadata/        # Model metadata and configurations
```

## Usage

- `collaborative/`: Stores serialized collaborative filtering models
- `content_based/`: Stores serialized content-based models
- `hybrid/`: Stores serialized hybrid recommendation models
- `metadata/`: Contains model configurations and metadata

## File Formats

- Models: `.joblib` (Joblib serialization)
- Metadata: `.json` (JSON format)
- Configurations: `.yaml` (YAML format)

## Model Types

1. Collaborative Filtering
   - Matrix factorization models
   - Neural collaborative filtering models

2. Content-Based
   - TF-IDF based models
   - Neural content models

3. Hybrid
   - Weighted hybrid models
   - Switching hybrid models 