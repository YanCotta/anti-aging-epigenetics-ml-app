from sqlalchemy import create_engine, Column, Integer, String, JSON, DateTime, ForeignKey, Float
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from datetime import datetime
import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://antiaging_user:antiaging_password@db:5432/antiaging_db")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)

class GeneticProfile(Base):
    __tablename__ = "genetic_profiles"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    snp_data = Column(JSON)
    upload_date = Column(DateTime, default=datetime.utcnow)

class Habits(Base):
    __tablename__ = "habits"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    exercises_per_week = Column(Integer)
    daily_calories = Column(Integer)
    alcohol_doses_per_week = Column(Float)
    years_smoking = Column(Integer)
    hours_of_sleep = Column(Float)
    stress_level = Column(Integer)
    entry_date = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
