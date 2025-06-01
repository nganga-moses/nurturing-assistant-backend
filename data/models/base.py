from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os

Base = declarative_base()

def get_engine(db_url=None):
    if db_url is None:
        db_url = os.environ.get("DATABASE_URL", "sqlite:///./student_engagement.db")
    return create_engine(db_url)

def get_session():
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()

def init_db():
    # All models are imported in __init__.py, so Base.metadata.create_all works
    Base.metadata.create_all(get_engine()) 