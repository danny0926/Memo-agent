from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import fitz  # PyMuPDF

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
    """RAG 對話功能（使用 chunk 搜尋，避免載入整本書）"""
    try:
        # 語意搜尋取得相關的 chunks（不是整本書！）
        chunk_results = ai_service.search_chunks(request.query, top_k=5)
        
        # 使用 chunk 內容作為 context
        contexts = []
        sources = []
        seen_note_ids = set()
        
        for result in chunk_results:
            title = result.get("title", "未知")
            content = result.get("content", "")
            chunk_index = result.get("chunk_index", 0)
            total_chunks = result.get("total_chunks", 1)
            note_id = result.get("note_id")
            
            # 加入 context（標註來源和區塊位置）
            contexts.append(f"來源: {title} (片段 {chunk_index + 1}/{total_chunks})\n內容: {content}")
            
            # 避免重複的 source（同一本書可能有多個 chunk 被選中）
            if note_id and note_id not in seen_note_ids:
                seen_note_ids.add(note_id)
                with get_session() as session:
                    note = session.query(Note).filter(Note.id == note_id).first()
                    if note:
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


@app.post("/upload-pdf/", response_model=dict)
async def upload_pdf(file: UploadFile = File(...), custom_title: Optional[str] = None):
    """上傳 PDF 檔案，自動解析並建立筆記（支援長文檔分塊處理）"""
    try:
        # 驗證檔案類型
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="只支援 PDF 檔案")
        
        # 讀取 PDF 內容
        pdf_content = await file.read()
        
        # 使用 PyMuPDF 解析 PDF
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        page_count = len(doc)
        
        # 提取所有文字
        full_text = ""
        for page_num in range(page_count):
            page = doc[page_num]
            full_text += page.get_text() + "\n\n"
        
        doc.close()
        
        if not full_text.strip():
            raise HTTPException(status_code=400, detail="無法從 PDF 提取文字，可能是掃描檔或圖片 PDF")
        
        # 使用檔名或自訂標題
        title = custom_title if custom_title else file.filename.replace('.pdf', '').replace('.PDF', '')
        
        # 對長文檔使用階層式摘要
        if len(full_text) > 10000:
            print(f"處理長文檔: {title}，共 {len(full_text)} 字，使用階層式摘要...")
            summary = ai_service.get_hierarchical_summary(full_text)
        else:
            summary = ai_service.get_summary(full_text)
        
        # 生成標籤（只用前 10000 字）
        content_for_tags = full_text[:10000] if len(full_text) > 10000 else full_text
        tags = ai_service.get_tags(content_for_tags)
        
        # 建立筆記物件
        note_obj = Note(
            title=title,
            content=full_text,
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
        
        # 使用分塊處理加入向量資料庫
        chunk_result = ai_service.process_long_document(
            content=full_text,
            title=title,
            note_id=note_id
        )
        
        return {
            "message": "PDF 上傳並建立筆記成功",
            "id": note_id,
            "title": title,
            "summary": summary,
            "tags": tags,
            "content_length": len(full_text),
            "pages": page_count,
            "chunks": chunk_result["total_chunks"],
            "chunks_indexed": chunk_result["success_chunks"]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"處理 PDF 時發生錯誤: {str(e)}")


@app.post("/upload-pdfs-batch/", response_model=dict)
async def upload_pdfs_batch(files: List[UploadFile] = File(...)):
    """批次上傳多個 PDF 檔案"""
    results = []
    errors = []
    
    for file in files:
        try:
            if not file.filename.lower().endswith('.pdf'):
                errors.append({"filename": file.filename, "error": "不是 PDF 檔案"})
                continue
            
            # 讀取 PDF 內容
            pdf_content = await file.read()
            doc = fitz.open(stream=pdf_content, filetype="pdf")
            
            # 提取文字
            full_text = ""
            page_count = len(doc)
            for page_num in range(page_count):
                page = doc[page_num]
                full_text += page.get_text() + "\n\n"
            doc.close()
            
            if not full_text.strip():
                errors.append({"filename": file.filename, "error": "無法提取文字"})
                continue
            
            title = file.filename.replace('.pdf', '').replace('.PDF', '')
            content_for_summary = full_text[:10000] if len(full_text) > 10000 else full_text
            
            # AI 生成摘要和標籤
            summary = ai_service.get_summary(content_for_summary)
            tags = ai_service.get_tags(content_for_summary)
            
            # 儲存筆記
            note_obj = Note(
                title=title,
                content=full_text,
                summary=summary,
                tags=tags,
                created_at=datetime.now(timezone.utc)
            )
            
            with get_session() as session:
                session.add(note_obj)
                session.commit()
                session.refresh(note_obj)
                note_id = note_obj.id
            
            # 儲存向量
            ai_service.add_to_vector_store(
                note_id=note_id,
                content=full_text[:5000],
                title=title
            )
            
            results.append({
                "filename": file.filename,
                "id": note_id,
                "title": title,
                "pages": page_count
            })
            
        except Exception as e:
            errors.append({"filename": file.filename, "error": str(e)})
    
    return {
        "message": f"成功處理 {len(results)} 個檔案，失敗 {len(errors)} 個",
        "success": results,
        "errors": errors
    }

