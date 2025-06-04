import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Load environment variables
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set.")

# Synchronous engine
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# Synchronous session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)