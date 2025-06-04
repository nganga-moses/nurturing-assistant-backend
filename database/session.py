from database.engine import SessionLocal


def get_db():
    """
    Provides a synchronous database session.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
