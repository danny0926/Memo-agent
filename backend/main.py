from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from backend.ai_service import get_summary, get_tags, embed_content, search_notes
from backend.database import Note, Session
from datetime import datetime

app = FastAPI()

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

@app.post("/notes/")
async def create_note(note: NoteRequest):
    summary = get_summary(note.content)
    tags = get_tags(note.content)
    vector = embed_content(note.content)
    note_obj = Note(title=note.title, content=note.content, summary=summary, tags=tags, created_at=datetime.now())
    session = Session()
    session.add(note_obj)
    session.commit()
    session.close()
    return {"message": "Note created successfully"}

@app.get("/notes/")
async def get_notes():
    session = Session()
    notes = session.query(Note).all()
    session.close()
    return [{"id": note.id, "title": note.title, "content": note.content, "summary": note.summary, "tags": note.tags, "created_at": note.created_at} for note in notes]

@app.post("/chat/")
async def chat(query: str):
    results = search_notes(query)
    return {"answer": "這是答案", "sources": results}

@app.get("/health/")
async def health():
    return {"status": "ok"}
