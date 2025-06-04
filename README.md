# University Student Engagement Recommender System

A machine learning-powered system that provides personalized engagement recommendations for university admissions departments to optimize student interactions throughout the application funnel.

## Features

- Personalized engagement recommendations (email, SMS, campus visits)
- Application likelihood prediction
- Dropout risk assessment
- Multi-stage funnel tracking
- Interactive dashboard for student management
- Bulk action management for engagement campaigns

## Project Overview

This system uses TensorFlow Recommenders (TFRS) to predict the most effective next engagement for prospective students. It tracks students through the application funnel (Awareness → Interest → Consideration → Decision → Application) and provides targeted recommendations based on their profile and interaction history.

### Key Components

1. **Recommendation Engine**: A two-tower retrieval model for matching students with appropriate engagements
2. **Application Likelihood Prediction**: Estimates probability of application completion
3. **Dropout Risk Assessment**: Identifies students at risk of dropping off the funnel
4. **Dashboard**: Visualizes key metrics and student distribution
5. **Student Management**: Interface for viewing and managing individual students
6. **Bulk Actions**: Tools for applying engagement strategies to student segments

## Project Structure

```
student-engagement-recommender/
├── api/                    # FastAPI application
│   ├── routes/            # API route definitions
│   ├── services/          # Business logic services
│   ├── auth/              # Authentication and authorization
│   └── dashboard.py       # Dashboard-specific endpoints
├── models/                # ML models implementation
├── data/                  # Data processing and management
├── utils/                 # Helper functions and utilities
├── database/             # Database files and migrations
├── migrations/           # Alembic database migrations
├── tests/                # Test suite
├── docs/                 # Documentation
├── scripts/              # Utility scripts
├── configs/              # Configuration files
├── na_frontend/         # Frontend application
├── main.py              # Application entry point
├── requirements.txt     # Python dependencies
└── README.md           # Project documentation
```

## Quick Start

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Initialize the database:
   ```bash
   alembic upgrade head
   ```

4. Start the backend server:
   ```bash
   python main.py
   ```
   Or for development with auto-reload:
   ```bash
   uvicorn api.main:app --reload
   ```

5. Start the frontend development server:
   ```bash
   cd na_frontend
   npm install
   npm start
   ```

## API Documentation

Once the backend server is running, visit:
- Swagger UI: http://localhost:8000/api/docs
- ReDoc: http://localhost:8000/api/redoc

## Technologies Used

### Backend
- Python 3.8+
- FastAPI
- SQLAlchemy
- Alembic
- Pydantic
- TensorFlow & TensorFlow Recommenders

### Frontend
- React
- TypeScript
- Redux Toolkit
- Material-UI
- Chart.js
- Axios

## Development

### Database Migrations

To create a new migration:
```bash
alembic revision --autogenerate -m "description of changes"
```

To apply migrations:
```bash
alembic upgrade head
```

### Running Tests

```bash
pytest
```

## License

MIT License
