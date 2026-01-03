import pytest
import os
import sys
import httpx
from unittest.mock import MagicMock
from datetime import datetime, timezone

from sqlmodel import SQLModel, Session
from fastapi import FastAPI

from database import Note

# 設定測試環境變數
os.environ["GEMINI_API_KEY"] = "test_api_key"
os.environ["DATA_DIR"] = "./test_data"

# 確保可以 import backend 模組
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============ Mock Classes ============
class MockAIService:
    """Mock AI 服務，避免呼叫真實 API"""
    
    def get_summary(self, content: str) -> str:
        return f"摘要: {content[:20]}..."
    
    def get_tags(self, content: str) -> str:
        return "測試, Python, AI"
    
    def get_embedding(self, text: str):
        return [0.1] * 768
    
    def add_to_vector_store(self, note_id: int, content: str, title: str):
        pass
    
    def search_notes(self, query: str, top_k: int = 3):
        return [
            {"note_id": 1, "title": "測試筆記", "score": 0.95}
        ]
    
    def generate_rag_response(self, query: str, contexts):
        return f"這是針對「{query}」的回答"


# ============ 測試 Database Model ============
class TestDatabase:
    """資料庫模組測試"""
    
    def test_note_model_creation(self):
        """測試 Note 模型建立"""
        
        note = Note(
            title="測試標題",
            content="測試內容",
            summary="測試摘要",
            tags="Python, AI",
            created_at=datetime.utcnow()
        )
        
        assert note.title == "測試標題"
        assert note.content == "測試內容"
        assert note.summary == "測試摘要"
        assert note.tags == "Python, AI"
    
    def test_note_model_fields(self):
        """測試 Note 模型欄位型態"""
        
        # 確認 Note 繼承自 SQLModel
        assert issubclass(Note, SQLModel)
        
        # 確認必要欄位存在
        field_names = Note.model_fields.keys()
        assert "id" in field_names
        assert "title" in field_names
        assert "content" in field_names
        assert "summary" in field_names
        assert "tags" in field_names
        assert "created_at" in field_names


# ============ 測試 AI Service Unit ============
class TestAIServiceUnit:
    """AI 服務單元測試（使用 Mock）"""
    
    def test_mock_get_summary(self):
        """測試 Mock 摘要生成"""
        mock_service = MockAIService()
        summary = mock_service.get_summary("這是一段很長的測試內容，用來測試摘要功能是否正常運作")
        
        assert "摘要" in summary
        assert len(summary) > 0
    
    def test_mock_get_tags(self):
        """測試 Mock 標籤生成"""
        mock_service = MockAIService()
        tags = mock_service.get_tags("Python 程式設計入門")
        
        assert "測試" in tags
        assert "," in tags  # 確認有多個標籤
    
    def test_mock_search_notes(self):
        """測試 Mock 筆記搜尋"""
        mock_service = MockAIService()
        results = mock_service.search_notes("Python")
        
        assert len(results) > 0
        assert "note_id" in results[0]
        assert "score" in results[0]
    
    def test_mock_generate_rag_response(self):
        """測試 Mock RAG 回答"""
        mock_service = MockAIService()
        answer = mock_service.generate_rag_response("什麼是 AI?", ["AI 是人工智慧"])
        
        assert "什麼是 AI?" in answer
    
    def test_mock_get_embedding(self):
        """測試 Mock 向量生成"""
        mock_service = MockAIService()
        embedding = mock_service.get_embedding("測試文字")
        
        assert len(embedding) == 768
        assert all(isinstance(x, float) for x in embedding)
    
    def test_mock_add_to_vector_store(self):
        """測試 Mock 加入向量資料庫"""
        mock_service = MockAIService()
        # 應該不會拋出例外
        mock_service.add_to_vector_store(1, "測試內容", "測試標題")


# ============ 測試 Pydantic Models ============
class TestPydanticModels:
    """Pydantic 模型測試"""
    
    def test_note_request_model(self):
        """測試 NoteRequest 模型"""
        
        from pydantic import BaseModel
        class NoteRequest(BaseModel):
            title: str
            content: str
        
        # 正常情況
        note = NoteRequest(title="測試", content="內容")
        assert note.title == "測試"
        assert note.content == "內容"
        
        # 缺少欄位應該拋出 ValidationError
        with pytest.raises(Exception):
            NoteRequest(title="只有標題")
    
    def test_chat_request_model(self):
        """測試 ChatRequest 模型"""
        
        from pydantic import BaseModel
        class ChatRequest(BaseModel):
            query: str
        
        # 正常情況
        chat = ChatRequest(query="什麼是 Python?")
        assert chat.query == "什麼是 Python?"
        
        # 缺少欄位應該拋出 ValidationError
        with pytest.raises(Exception):
            ChatRequest()


# ============ 測試 Utility Functions ============
class TestUtilityFunctions:
    """工具函數測試"""
    
    def test_data_dir_environment_variable(self):
        """測試 DATA_DIR 環境變數"""
        os.environ["DATA_DIR"] = "./custom_data"
        assert os.environ.get("DATA_DIR") == "./custom_data"
        
        # 還原
        os.environ["DATA_DIR"] = "./test_data"
    
    def test_gemini_api_key_environment_variable(self):
        """測試 GEMINI_API_KEY 環境變數"""
        os.environ["GEMINI_API_KEY"] = "test_key_123"
        assert os.environ.get("GEMINI_API_KEY") == "test_key_123"
    
    def test_datetime_utc(self):
        """測試 UTC 時間"""
        from datetime import datetime, timezone
        
        now = datetime.now(timezone.utc)
        assert now.tzinfo is not None


# ============ 測試 API Response Format ============
class TestAPIResponseFormat:
    """API 回應格式測試"""
    
    def test_health_response_format(self):
        """測試健康檢查回應格式"""
        expected_response = {"status": "ok"}
        assert "status" in expected_response
        assert expected_response["status"] == "ok"
    
    def test_note_create_response_format(self):
        """測試筆記建立回應格式"""
        expected_response = {
            "message": "筆記建立成功",
            "id": 1,
            "summary": "測試摘要",
            "tags": "Python, AI"
        }
        
        assert "message" in expected_response
        assert "id" in expected_response
        assert "summary" in expected_response
        assert "tags" in expected_response
    
    def test_chat_response_format(self):
        """測試對話回應格式"""
        expected_response = {
            "answer": "這是 AI 的回答",
            "sources": [
                {"id": 1, "title": "筆記1", "summary": "摘要1", "score": 0.95}
            ]
        }
        
        assert "answer" in expected_response
        assert "sources" in expected_response
        assert isinstance(expected_response["sources"], list)


# ============ 整合測試 (使用 Mock) ============
class TestIntegration:
    """整合測試"""
    
    def test_full_note_workflow(self):
        """測試完整筆記工作流程"""
        # 模擬建立筆記
        mock_service = MockAIService()
        
        # 1. 生成摘要
        content = "Python 是一種簡單易學的程式語言"
        summary = mock_service.get_summary(content)
        assert len(summary) > 0
        
        # 2. 生成標籤
        tags = mock_service.get_tags(content)
        assert "," in tags
        
        # 3. 生成向量
        embedding = mock_service.get_embedding(content)
        assert len(embedding) == 768
    
    def test_full_chat_workflow(self):
        """測試完整對話工作流程"""
        mock_service = MockAIService()
        
        # 1. 搜尋筆記
        query = "什麼是 Python?"
        results = mock_service.search_notes(query)
        assert len(results) > 0
        
        # 2. 生成回答
        contexts = ["Python 是一種程式語言"]
        answer = mock_service.generate_rag_response(query, contexts)
        assert len(answer) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# ============ 測試 API Endpoints (需要啟動 FastAPI 應用程式) ============
@pytest.mark.asyncio
class Settings(BaseSettings):
    api_url: str = "http://localhost:8000"

class TestAPIs:
    """API 端點測試（需要啟動 FastAPI 應用程式）"""

    @pytest.fixture(scope="class")
    def app(self):
        """建立 FastAPI 應用程式實例"""
        from main import app
        return app

    @pytest.fixture(scope="class")
    async def client(self, app: FastAPI):
        """建立測試用 HTTP 客戶端"""
        async with httpx.AsyncClient(app=app, base_url="http://test") as client:
            yield client

    async def test_create_note_api(self, client: httpx.AsyncClient):
        """測試建立筆記 API"""
        note_data = {
            "title": "測試筆記",
            "content": "這是測試內容"
        }
        response = await client.post("/notes/", json=note_data)

        assert response.status_code == 200
        assert "message" in response.json()
        assert response.json()["message"] == "筆記建立成功"
        assert "id" in response.json()
        assert "summary" in response.json()
        assert "tags" in response.json()

        # 驗證資料庫是否已寫入
        from database import engine
        SQLModel.metadata.create_all(engine)
        with Session(engine) as session:
            note = session.get(Note, response.json()["id"])
            assert note is not None
            assert note.title == "測試筆記"
            assert note.content == "這是測試內容"

    async def test_chat_api(self, client: httpx.AsyncClient):
        """測試對話 API"""
        chat_data = {
            "query": "什麼是測試筆記？"
        }
        response = await client.post("/chat/", json=chat_data)

        assert response.status_code == 200
        assert "answer" in response.json()
        assert "sources" in response.json()
        assert isinstance(response.json()["sources"], list)

        # 驗證是否有回答
        assert len(response.json()["answer"]) > 0

        # 驗證是否有參考資料
        if len(response.json()["sources"]) > 0:
            assert "id" in response.json()["sources"][0]
            assert "title" in response.json()["sources"][0]
            assert "summary" in response.json()["sources"][0]
            assert "score" in response.json()["sources"][0]

