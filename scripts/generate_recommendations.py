import pandas as pd
from models.recommenders.collaborative import CollaborativeFilteringModel
from models.core.recommendation_logger import RecommendationLogger

def generate_and_log_recommendations(
    students_path: str,
    engagements_path: str,
    content_path: str,
    model_dir: str = "models/saved_models",
    n_recommendations: int = 5,
    log_path: str = "recommendation_logs.csv"
):
    """
    Generate and log recommendations for all students using the trained collaborative model.
    Returns the logger instance with all logs.
    """
    # Load data
    students_df = pd.read_csv(students_path)
    engagements_df = pd.read_csv(engagements_path)
    content_df = pd.read_csv(content_path)

    # Initialize and load model
    model = CollaborativeFilteringModel(model_dir=model_dir)
    model.load()

    # Initialize logger
    logger = RecommendationLogger()

    # Generate and log recommendations for each student
    for student_id in students_df['student_id']:
        recommendations = model.get_recommendations(student_id, n_recommendations=n_recommendations)
        for rec in recommendations:
            logger.log_recommendation(rec, status="active", rationale="Top recommendation for current funnel stage.")

    # Save logs for reporting
    logger.save_to_csv(log_path)
    return logger

if __name__ == "__main__":
    logger = generate_and_log_recommendations(
        students_path="data/students.csv",
        engagements_path="data/engagements.csv",
        content_path="data/content.csv",
        log_path="recommendation_logs.csv"
    )
    print(f"Logged {len(logger.logs)} recommendations.") 