# Model Artifacts Directory

This directory contains all model artifacts and training checkpoints.

## Directory Structure

```
saved/
├── checkpoints/     # Training checkpoints
├── embeddings/      # Model embeddings
├── model_weights/   # Model weight files
└── vocabularies/    # Vocabulary files
```

## Usage

- `checkpoints/`: Contains training checkpoints for model recovery
- `embeddings/`: Stores model embeddings for student and content
- `model_weights/`: Contains saved model weights in HDF5 format
- `vocabularies/`: Stores vocabulary mappings for text processing

## File Formats

- Model weights: `.h5` (HDF5 format)
- Embeddings: `.npy` (NumPy format)
- Vocabularies: `.json` (JSON format)
- Checkpoints: TensorFlow checkpoint format 