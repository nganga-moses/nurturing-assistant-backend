# Models Directory

This directory contains all model-related code for the Student Engagement Recommender System. The codebase is organized into the following submodules:

## Structure

- **core/**: Core deep learning model logic, data processing, and training utilities.
  - `recommender_model.py`: Implements the main two-tower deep learning model (`RecommenderModel`) for student-engagement matching, and `ModelTrainer` for training and evaluation.
  - `recommendation_service.py`, `data_preprocessor.py`, `model_trainer.py`, `data_generator.py`: Core logic for data processing, training, and orchestration.

- **recommenders/**: Classical and hybrid recommender system implementations.
  - `base_recommender.py`: Abstract base class for all recommenders.
  - `collaborative.py`: Collaborative filtering model using TensorFlow Recommenders.
  - `content_based.py`: Content-based and simple recommenders (TF-IDF, Nearest Neighbors, etc.).
  - `utils.py`: Helper functions for recommenders.

- **evaluation/**: Model evaluation and interpretability tools.
  - `model_evaluator.py`: Cross-validation, feature importance, and explanation generation.
  - `score_calculator.py`, `interpretability.py`: Additional evaluation and interpretability logic.

- **saved_models/**, **saved/**, **save_modules/**: Directories for storing trained model artifacts and weights.

## Notes
- The previous `StudentEngagementModel` class has been replaced by `RecommenderModel` and related classes in `core/recommender_model.py`.
- Functionality is now modularized for clarity and extensibility.
- For training, use the scripts in the project root (e.g., `train_model.py`) which leverage these modules.

---
For more details, see the docstrings in each module or contact the maintainers. 