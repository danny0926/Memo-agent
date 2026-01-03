from sqlmodel import SQLModel, Field
from datetime import datetime
from typing import Optional

class Note(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    title: str
    content: str
    summary: str
    tags: str
    created_at: datetime

    class Config:
        table_name = "notes"
