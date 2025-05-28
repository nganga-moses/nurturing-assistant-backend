import pandas as pd
from models.recommenders.collaborative import CollaborativeFilteringModel
from models.core.engagement_content_preprocessor import EngagementContentPreprocessor

def train_collaborative_model(
    students_path: str,
    engagements_path: str,
    content_path: str,
    embedding_dimension: int = 64,
    learning_rate: float = 0.05,
    epochs: int = 10,
    batch_size: int = 32
):
    """
    Train the collaborative filtering model using the new engagement+content pipeline.
    Returns the trained model and training history.
    """
    # Load data
    students_df = pd.read_csv(students_path)
    engagements_df = pd.read_csv(engagements_path)
    content_df = pd.read_csv(content_path)

    # Initialize model
    model = CollaborativeFilteringModel(
        embedding_dimension=embedding_dimension,
        learning_rate=learning_rate
    )

    # Train model
    history = model.train(
        students_df,
        engagements_df,
        content_df,
        epochs=epochs,
        batch_size=batch_size
    )

    # Save model
    model.save()
    return model, history

if __name__ == "__main__":
    # Example usage
    model, history = train_collaborative_model(
        students_path="data/students.csv",
        engagements_path="data/engagements.csv",
        content_path="data/content.csv"
    )
    print("Training complete. History:", history) 