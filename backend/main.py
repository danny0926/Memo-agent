from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from database import Note, get_session, init_db
from ai_service import AIService

# Initialize AI service
ai_service = AIService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    init_db()
    yield


app = FastAPI(title="Memo-Agent API", lifespan=lifespan)


class NoteRequest(BaseModel):
    title: str
    content: str


class NoteResponse(BaseModel):
    id: int
    title: str
    content: str
    summary: str
    tags: str
    created_at: datetime


class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]


@app.post("/notes/", response_model=dict)
async def create_note(note: NoteRequest):
    """建立新筆記，自動生成摘要和標籤"""
    try:
        # 使用 Gemini 生成摘要和標籤
        summary = ai_service.get_summary(note.content)
        tags = ai_service.get_tags(note.content)
        
        # 建立筆記物件
        note_obj = Note(
            title=note.title,
            content=note.content,
            summary=summary,
            tags=tags,
            created_at=datetime.now(timezone.utc)
        )
        
        # 儲存到 SQLite
        with get_session() as session:
            session.add(note_obj)
            session.commit()
            session.refresh(note_obj)
            note_id = note_obj.id
        
        # 儲存向量到 ChromaDB
        ai_service.add_to_vector_store(
            note_id=note_id,
            content=note.content,
            title=note.title
        )
        
        return {
            "message": "筆記建立成功",
            "id": note_id,
            "summary": summary,
            "tags": tags
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/notes/", response_model=List[NoteResponse])
async def get_notes():
    """取得所有筆記清單（依時間排序）"""
    from sqlmodel import select
    with get_session() as session:
        notes = session.exec(select(Note).order_by(Note.created_at.desc())).all()
        return [
            NoteResponse(
                id=note.id,
                title=note.title,
                content=note.content,
                summary=note.summary,
                tags=note.tags,
                created_at=note.created_at
            )
            for note in notes
        ]


@app.post("/chat/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """RAG 對話功能"""
    try:
        # 語意搜尋取得相關筆記
        search_results = ai_service.search_notes(request.query, top_k=3)
        
        # 取得筆記內容作為 context
        contexts = []
        sources = []
        with get_session() as session:
            for result in search_results:
                note_id = result.get("note_id")
                if note_id:
                    note = session.query(Note).filter(Note.id == note_id).first()
                    if note:
                        contexts.append(f"標題: {note.title}\n內容: {note.content}")
                        sources.append({
                            "id": note.id,
                            "title": note.title,
                            "summary": note.summary,
                            "score": result.get("score", 0)
                        })
        
        # 使用 RAG 生成回答
        answer = ai_service.generate_rag_response(request.query, contexts)
        
        return ChatResponse(answer=answer, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health/")
async def health():
    """健康檢查"""
    return {"status": "ok"}
