"""Sync SQLAlchemy 2.0 Pattern for FastAPI.

Keywords: sqlalchemy, database, session, sqlite, postgres, sync

Use this pattern when:
- Using SQLite (doesn't support async well)
- Simpler requirements
- Not CPU-bound (async doesn't help much)

Requirements:
    pip install sqlalchemy

For production with PostgreSQL, prefer the async pattern.
"""

from contextlib import contextmanager
from typing import Generator

from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy import create_engine, select, Column, Integer, String
from sqlalchemy.orm import Session, sessionmaker, DeclarativeBase
from pydantic import BaseModel
import os

# Database URL
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./app.db")

# Create engine
engine = create_engine(
    DATABASE_URL,
    # For SQLite, need check_same_thread=False for FastAPI
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    pool_pre_ping=True,
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# SQLAlchemy 2.0 base class
class Base(DeclarativeBase):
    pass


# Example model
class Item(Base):
    __tablename__ = "items"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(String)


# Pydantic schemas
class ItemCreate(BaseModel):
    name: str
    description: str = ""


class ItemResponse(BaseModel):
    id: int
    name: str
    description: str

    model_config = {"from_attributes": True}


# Dependency: Get database session
def get_db() -> Generator[Session, None, None]:
    """Dependency that provides a database session.

    Creates a new session for each request.
    Properly closes session after request completes.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


app = FastAPI()


@app.on_event("startup")
def startup():
    """Create tables on startup."""
    Base.metadata.create_all(bind=engine)


@app.post("/items", response_model=ItemResponse)
def create_item(item: ItemCreate, db: Session = Depends(get_db)):
    """Create a new item."""
    db_item = Item(name=item.name, description=item.description)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item


@app.get("/items/{item_id}", response_model=ItemResponse)
def get_item(item_id: int, db: Session = Depends(get_db)):
    """Get an item by ID."""
    # SQLAlchemy 2.0 style query
    stmt = select(Item).where(Item.id == item_id)
    item = db.execute(stmt).scalar_one_or_none()

    if not item:
        raise HTTPException(status_code=404, detail="Item not found")

    return item


@app.get("/items", response_model=list[ItemResponse])
def list_items(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """List all items with pagination."""
    stmt = select(Item).offset(skip).limit(limit)
    items = db.execute(stmt).scalars().all()
    return items


@app.put("/items/{item_id}", response_model=ItemResponse)
def update_item(item_id: int, item: ItemCreate, db: Session = Depends(get_db)):
    """Update an item."""
    stmt = select(Item).where(Item.id == item_id)
    db_item = db.execute(stmt).scalar_one_or_none()

    if not db_item:
        raise HTTPException(status_code=404, detail="Item not found")

    db_item.name = item.name
    db_item.description = item.description
    db.commit()
    db.refresh(db_item)
    return db_item


@app.delete("/items/{item_id}")
def delete_item(item_id: int, db: Session = Depends(get_db)):
    """Delete an item."""
    stmt = select(Item).where(Item.id == item_id)
    item = db.execute(stmt).scalar_one_or_none()

    if not item:
        raise HTTPException(status_code=404, detail="Item not found")

    db.delete(item)
    db.commit()
    return {"message": "Item deleted"}
