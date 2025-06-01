# Tests Directory

This directory contains all test files for the Student Engagement Recommender System.

## Directory Structure

```
tests/
├── unit/                      # Unit tests
│   ├── models/               # Model tests
│   │   ├── test_recommender.py
│   │   └── test_training.py
│   ├── data/                 # Data processing tests
│   │   ├── test_data_quality.py
│   │   ├── test_engagement_handler.py
│   │   └── test_vector_store.py
│   └── api/                  # API tests
│       └── test_routes.py
├── integration/              # Integration tests
│   ├── test_data_pipeline.py
│   └── test_model_pipeline.py
├── fixtures/                 # Test fixtures and utilities
│   ├── synthetic_data.py
│   └── test_data.py
└── conftest.py              # Pytest configuration
```

## Test Categories

### Unit Tests
- `models/`: Tests for recommendation models and training
- `data/`: Tests for data processing, quality monitoring, and vector storage
- `api/`: Tests for API endpoints and routes

### Integration Tests
- Data pipeline tests
- Model pipeline tests
- End-to-end workflow tests

### Fixtures
- Synthetic data generation
- Test data utilities
- Common test setup

## Running Tests

1. Run all tests:
```bash
pytest
```

2. Run specific test categories:
```bash
# Run unit tests
pytest tests/unit/

# Run integration tests
pytest tests/integration/

# Run specific test file
pytest tests/unit/models/test_recommender.py
```

3. Run with coverage:
```bash
pytest --cov=.
```

## Test Data

- Synthetic data is generated in `fixtures/synthetic_data.py`
- Test data is stored in `fixtures/test_data.py`
- Each test module can use these fixtures for consistent testing

## Adding New Tests

1. Place unit tests in appropriate subdirectory under `unit/`
2. Place integration tests in `integration/`
3. Add shared test utilities to `fixtures/`
4. Follow existing test patterns and naming conventions
5. Include docstrings and comments for complex test cases 