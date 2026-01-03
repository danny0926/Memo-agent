import os
from sqlmodel import SQLModel, Field, create_engine, Session as SQLModelSession
from datetime import datetime
from typing import Optional, Generator
from contextlib import contextmanager

# 資料庫路徑
DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
DATABASE_URL = f"sqlite:///{DATA_DIR}/memo_agent.db"


class Note(SQLModel, table=True):
    """筆記資料模型"""
    __tablename__ = "notes"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str = Field(index=True)
    content: str
    summary: str
    tags: str
    created_at: datetime = Field(default_factory=lambda: datetime.utcnow())


# 建立資料庫引擎
engine = create_engine(DATABASE_URL, echo=False)


def init_db():
    """初始化資料庫，建立所有表格"""
    os.makedirs(DATA_DIR, exist_ok=True)
    SQLModel.metadata.create_all(engine)


@contextmanager
def get_session() -> Generator[SQLModelSession, None, None]:
    """取得資料庫 session 的 context manager"""
    session = SQLModelSession(engine)
    try:
        yield session
    finally:
        session.close()
