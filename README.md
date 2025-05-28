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

## Completed Features

### At-Risk Students Page
- Created the AtRiskStudents component to display students at high risk of dropping off
- Implemented backend API endpoint for fetching at-risk students
- Added Redux thunk to fetch at-risk students from the API

### High-Potential Students Page
- Created the HighPotentialStudents component to display students with high application likelihood
- Implemented backend API endpoint for fetching high-potential students
- Added Redux thunk to fetch high-potential students from the API

### Recommended Nudges Page
- Created the RecommendedNudges component to display student-specific recommendations
- Added support for fetching all recommendations in the Redux store
- Implemented UI to display recommendations grouped by student

### Fixed Student List Page
- Enhanced the StudentList component to better display application likelihood
- Added text labels to make likelihood scores more understandable
- Ensured proper display of risk levels using the RiskIndicator component

### Backend API Enhancements
- Added endpoint for high-potential students
- Fixed the at-risk students endpoint to use query parameters
- Implemented mock data for development and testing

## Project Structure

```
student-engagement-recommender/
├── backend/
│   ├── api/           # FastAPI endpoints
│   ├── data/          # Data processing and management
│   ├── models/        # ML models implementation
│   └── utils/         # Helper functions and utilities
├── frontend/          # React.js frontend application
│   ├── src/
│   │   ├── components/  # UI components
│   │   ├── store/       # Redux state management
│   │   ├── services/    # API services
│   │   └── types/       # TypeScript type definitions
├── setup_dev.sh      # Development environment setup script
└── README.md         # Project documentation
```

## Quick Start

To quickly set up the development environment, run the setup script:

```bash
./setup_dev.sh
```

This script will:
1. Create a Python virtual environment for the backend
2. Install all required Python dependencies
3. Initialize the database with sample data
4. Install all required Node.js dependencies for the frontend

## Manual Setup Instructions

### Backend Setup

1. Create and activate a virtual environment:
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the backend server:
   ```bash
   python main.py
   ```
   Or for development with auto-reload:
   ```bash
   uvicorn api.main:app --reload
   ```

### Frontend Setup

1. Install Node.js dependencies:
   ```bash
   cd frontend
   npm install
   ```

2. Start the development server:
   ```bash
   npm start
   ```

## Using the System

### Dashboard

The dashboard provides an overview of key metrics:
- Total active students
- Application rate
- Number of at-risk students
- Funnel stage distribution
- Engagement effectiveness by type

### Student Management

The student list allows you to:
- Filter students by funnel stage and risk level
- Search for specific students
- View detailed student profiles
- Generate personalized recommendations

### Bulk Actions

The bulk actions interface enables you to:
- Select an action type (email campaign, SMS campaign, etc.)
- Target a specific student segment
- Preview the action before applying
- Apply the action to multiple students at once

## API Documentation

Once the backend server is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Technologies Used

### Backend
- Python 3.8+
- FastAPI
- TensorFlow & TensorFlow Recommenders
- SQLAlchemy
- Pydantic

### Frontend
- React
- TypeScript
- Redux Toolkit
- Material-UI
- Chart.js
- Axios

## License

MIT License
