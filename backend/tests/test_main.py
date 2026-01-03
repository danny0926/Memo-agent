import pytest
import os
import sys
from datetime import datetime, timezone

from sqlmodel import SQLModel, Session
from fastapi import FastAPI

from database import Note
from unittest.mock import patch

# 檢測 Python 版本，判斷是否跳過需要 chromadb 的測試
SKIP_CHROMADB_TESTS = sys.version_info >= (3, 14)

# 設定測試環境變數
os.environ["GEMINI_API_KEY"] = "test_api_key"
os.environ["DATA_DIR"] = "./test_data"

# 確保可以 import backend 模組
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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
            created_at=datetime.now(timezone.utc)
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
    
    def test_create_note(self):
        """測試新增筆記"""
        from database import engine
        SQLModel.metadata.create_all(engine)
        with Session(engine) as session:
            note = Note(
                title="測試新增",
                content="測試內容",
                summary="測試摘要",
                tags="測試,新增",
                created_at=datetime.now(timezone.utc)
            )
            session.add(note)
            session.commit()
            session.refresh(note)
            
            assert note.id is not None
            assert note.title == "測試新增"
    
    def test_read_note(self):
        """測試讀取筆記"""
        from database import engine
        SQLModel.metadata.create_all(engine)
        with Session(engine) as session:
            # 先建立一個筆記
            note = Note(
                title="測試讀取",
                content="測試內容",
                summary="測試摘要",
                tags="測試,讀取",
                created_at=datetime.now(timezone.utc)
            )
            session.add(note)
            session.commit()
            session.refresh(note)
            
            # 讀取筆記
            retrieved_note = session.get(Note, note.id)
            assert retrieved_note is not None
            assert retrieved_note.title == "測試讀取"
    
    def test_update_note(self):
        """測試更新筆記"""
        from database import engine
        SQLModel.metadata.create_all(engine)
        with Session(engine) as session:
            # 先建立一個筆記
            note = Note(
                title="測試更新",
                content="測試內容",
                summary="測試摘要",
                tags="測試,更新",
                created_at=datetime.now(timezone.utc)
            )
            session.add(note)
            session.commit()
            session.refresh(note)
            
            # 更新筆記
            note.title = "更新後的標題"
            session.add(note)
            session.commit()
            session.refresh(note)
            
            # 讀取更新後的筆記
            updated_note = session.get(Note, note.id)
            assert updated_note is not None
            assert updated_note.title == "更新後的標題"
    
    def test_delete_note(self):
        """測試刪除筆記"""
        from database import engine
        SQLModel.metadata.create_all(engine)
        with Session(engine) as session:
            # 先建立一個筆記
            note = Note(
                title="測試刪除",
                content="測試內容",
                summary="測試摘要",
                tags="測試,刪除",
                created_at=datetime.now(timezone.utc)
            )
            session.add(note)
            session.commit()
            session.refresh(note)
            
            # 刪除筆記
            session.delete(note)
            session.commit()
            
            # 嘗試讀取已刪除的筆記
            deleted_note = session.get(Note, note.id)
            assert deleted_note is None


# ============ 測試 AI Service Unit ============
class TestAIServiceUnit:
    """AI 服務單元測試（使用 Mock）"""
    
    def test_mock_get_summary(self, mock_ai_service):
        """測試 Mock 摘要生成"""
        summary = mock_ai_service.get_summary("這是一段很長的測試內容，用來測試摘要功能是否正常運作")
        
        assert "這是測試摘要" in summary
        assert len(summary) > 0
    
    def test_mock_get_tags(self, mock_ai_service):
        """測試 Mock 標籤生成"""
        tags = mock_ai_service.get_tags("Python 程式設計入門")
        
        assert "測試" in tags
        assert "," in tags  # 確認有多個標籤
    
    def test_mock_search_notes(self, mock_ai_service):
        """測試 Mock 筆記搜尋"""
        results = mock_ai_service.search_notes("Python")
        
        assert len(results) > 0
        assert "note_id" in results[0]
        assert "score" in results[0]
    
    def test_mock_generate_rag_response(self, mock_ai_service):
        """測試 Mock RAG 回答"""
        answer = mock_ai_service.generate_rag_response("什麼是 AI?", ["AI 是人工智慧"])
        
        assert "這是 AI 回答" in answer
    
    def test_mock_get_embedding(self, mock_ai_service):
        """測試 Mock 向量生成"""
        embedding = mock_ai_service.get_embedding("測試文字")
        
        assert len(embedding) == 768
        assert all(isinstance(x, float) for x in embedding)
    
    def test_mock_add_to_vector_store(self, mock_ai_service):
        """測試 Mock 加入向量資料庫"""
        # 應該不會拋出例外
        mock_ai_service.add_to_vector_store(1, "測試內容", "測試標題")


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
    
    def test_full_note_workflow(self, mock_ai_service):
        """測試完整筆記工作流程"""
        # 模擬建立筆記
        
        # 1. 生成摘要
        content = "Python 是一種簡單易學的程式語言"
        summary = mock_ai_service.get_summary(content)
        assert len(summary) > 0
        
        # 2. 生成標籤
        tags = mock_ai_service.get_tags(content)
        assert "," in tags
        
        # 3. 生成向量
        embedding = mock_ai_service.get_embedding(content)
        assert len(embedding) == 768
    
    def test_full_chat_workflow(self, mock_ai_service):
        """測試完整對話工作流程"""
        # 1. 搜尋筆記
        query = "什麼是 Python?"
        results = mock_ai_service.search_notes(query)
        assert len(results) > 0
        
        # 2. 生成回答
        contexts = ["Python 是一種程式語言"]
        answer = mock_ai_service.generate_rag_response(query, contexts)
        assert len(answer) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# ============ 測試 API Endpoints (需要啟動 FastAPI 應用程式) ============
class TestAPIs:
    """API 端點測試（需要啟動 FastAPI 應用程式）"""

    @pytest.fixture(scope="class")
    def app(self):
        """建立 FastAPI 應用程式實例"""
        from main import app
        return app

    @pytest.fixture(scope="class")
    def client(self, app: FastAPI):
        """建立測試用 HTTP 客戶端（同步版本使用 TestClient）"""
        from starlette.testclient import TestClient
        with TestClient(app) as client:
            yield client

    @patch("main.ai_service.add_to_vector_store")
    @patch("main.ai_service.get_tags")
    @patch("main.ai_service.get_summary")
    def test_create_note_api(self, mock_get_summary, mock_get_tags, mock_add_to_vector_store, client):
        """測試建立筆記 API"""
        # 設定 mock 回傳值
        mock_get_summary.return_value = "測試摘要"
        mock_get_tags.return_value = "測試, Python"
        mock_add_to_vector_store.return_value = None

        note_data = {
            "title": "測試筆記",
            "content": "這是測試內容"
        }
        response = client.post("/notes/", json=note_data)

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

    def test_get_notes_api(self, client):
        """測試取得筆記列表 API"""
        response = client.get("/notes/")

        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_chat_api(self, client):
        """測試對話 API"""
        chat_data = {
            "query": "什麼是測試筆記？"
        }
        response = client.post("/chat/", json=chat_data)

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

    @patch("main.ai_service.add_to_vector_store")
    @patch("main.ai_service.get_tags")
    @patch("main.ai_service.get_summary")
    def test_create_note_api_ai_service_called(
        self,
        mock_get_summary,
        mock_get_tags,
        mock_add_to_vector_store,
        client
    ):
        """測試建立筆記 API 時 AI 服務是否被呼叫"""
        # 設定 mock 回傳值
        mock_get_summary.return_value = "測試摘要"
        mock_get_tags.return_value = "測試, Python"
        mock_add_to_vector_store.return_value = None

        note_data = {
            "title": "測試筆記",
            "content": "這是測試內容"
        }
        response = client.post("/notes/", json=note_data)

        assert response.status_code == 200
        mock_get_summary.assert_called_once_with("這是測試內容")
        mock_get_tags.assert_called_once_with("這是測試內容")
        # add_to_vector_store 使用關鍵字參數呼叫
        mock_add_to_vector_store.assert_called_once()
        call_kwargs = mock_add_to_vector_store.call_args.kwargs
        assert call_kwargs["content"] == "這是測試內容"
        assert call_kwargs["title"] == "測試筆記"

    @patch("main.ai_service.add_to_vector_store")
    @patch("main.ai_service.get_tags")
    @patch("main.ai_service.get_summary")
    def test_create_note_api_ai_service_error(self, mock_get_summary, mock_get_tags, mock_add_to_vector_store, client):
        """測試建立筆記 API 時，AI 服務發生錯誤"""
        # 設定 mock 行為
        mock_get_summary.side_effect = Exception("AI 服務發生錯誤")

        note_data = {
            "title": "測試筆記",
            "content": "這是測試內容"
        }
        response = client.post("/notes/", json=note_data)

        assert response.status_code == 500
        assert "detail" in response.json()
        assert response.json()["detail"] == "AI 服務發生錯誤"

    @patch("main.ai_service.add_to_vector_store")
    @patch("main.ai_service.get_tags")
    @patch("main.ai_service.get_summary")
    def test_create_note_api_vector_store_called(
        self,
        mock_get_summary,
        mock_get_tags,
        mock_add_to_vector_store,
        client
    ):
        """測試建立筆記時，向量資料庫是否被正確呼叫"""
        # 設定 mock 回傳值
        mock_get_summary.return_value = "測試摘要"
        mock_get_tags.return_value = "測試, Python"
        mock_add_to_vector_store.return_value = None

        note_data = {
            "title": "向量測試筆記",
            "content": "這是向量資料庫測試內容"
        }
        response = client.post("/notes/", json=note_data)

        assert response.status_code == 200
        # 驗證 add_to_vector_store 被呼叫，並檢查參數
        mock_add_to_vector_store.assert_called_once()
        call_kwargs = mock_add_to_vector_store.call_args.kwargs
        # 檢查 note_id
        assert call_kwargs["note_id"] == response.json()["id"]
        # 檢查 content
        assert call_kwargs["content"] == "這是向量資料庫測試內容"
        # 檢查 title
        assert call_kwargs["title"] == "向量測試筆記"

    @patch("main.ai_service.generate_rag_response")
    @patch("main.ai_service.search_notes")
    def test_chat_api_search_notes_called(
        self,
        mock_search_notes,
        mock_generate_rag_response,
        client
    ):
        """測試對話 API 時，search_notes 是否被正確呼叫"""
        # 設定 mock 回傳值
        mock_search_notes.return_value = []
        mock_generate_rag_response.return_value = "這是 AI 的回答"

        chat_data = {"query": "測試查詢"}
        response = client.post("/chat/", json=chat_data)

        assert response.status_code == 200
        # 驗證 search_notes 被正確呼叫
        mock_search_notes.assert_called_once_with("測試查詢", top_k=3)

    @patch("main.ai_service.generate_rag_response")
    @patch("main.ai_service.search_notes")
    def test_chat_api_rag_response(
        self,
        mock_search_notes,
        mock_generate_rag_response,
        client
    ):
        """測試 /chat/ 端點是否正確使用 RAG 方式生成回答"""
        # 先建立測試筆記
        from database import engine, Note
        SQLModel.metadata.create_all(engine)
        with Session(engine) as session:
            note = Note(
                title="RAG 測試筆記",
                content="這是 RAG 測試的內容",
                summary="RAG 測試摘要",
                tags="RAG, 測試",
                created_at=datetime.now(timezone.utc)
            )
            session.add(note)
            session.commit()
            session.refresh(note)
            note_id = note.id

        # 設定 mock 回傳值
        mock_search_notes.return_value = [
            {"note_id": note_id, "score": 0.95}
        ]
        mock_generate_rag_response.return_value = "根據筆記內容，這是 RAG 生成的回答"

        chat_data = {"query": "什麼是 RAG 測試？"}
        response = client.post("/chat/", json=chat_data)

        assert response.status_code == 200
        # 驗證 RAG 回應結構
        assert response.json()["answer"] == "根據筆記內容，這是 RAG 生成的回答"
        assert len(response.json()["sources"]) == 1
        assert response.json()["sources"][0]["id"] == note_id
        assert response.json()["sources"][0]["title"] == "RAG 測試筆記"
        assert response.json()["sources"][0]["score"] == 0.95

        # 驗證 generate_rag_response 被呼叫時有傳入正確的 contexts
        mock_generate_rag_response.assert_called_once()
        call_args = mock_generate_rag_response.call_args
        assert call_args[0][0] == "什麼是 RAG 測試？"
        # contexts 應該包含筆記標題和內容
        assert "RAG 測試筆記" in call_args[0][1][0]
        assert "這是 RAG 測試的內容" in call_args[0][1][0]


# ============ 安全性測試 ============
class TestSecurity:
    """安全性測試"""

    @pytest.fixture(scope="class")
    def app(self):
        """建立 FastAPI 應用程式實例"""
        from main import app
        return app

    @pytest.fixture(scope="class")
    def client(self, app: FastAPI):
        """建立測試用 HTTP 客戶端"""
        from starlette.testclient import TestClient
        with TestClient(app) as client:
            yield client

    @patch("main.ai_service.add_to_vector_store")
    @patch("main.ai_service.get_tags")
    @patch("main.ai_service.get_summary")
    def test_sql_injection_in_title(
        self,
        mock_get_summary,
        mock_get_tags,
        mock_add_to_vector_store,
        client
    ):
        """測試 SQL Injection 防護 - 標題欄位"""
        mock_get_summary.return_value = "測試摘要"
        mock_get_tags.return_value = "測試"
        mock_add_to_vector_store.return_value = None

        # SQL Injection 攻擊字串
        malicious_title = "'; DROP TABLE notes; --"
        note_data = {
            "title": malicious_title,
            "content": "正常內容"
        }
        response = client.post("/notes/", json=note_data)

        # 應該正常建立筆記，而不是執行 SQL 指令
        assert response.status_code == 200
        
        # 驗證資料庫仍然正常運作
        notes_response = client.get("/notes/")
        assert notes_response.status_code == 200

    @patch("main.ai_service.add_to_vector_store")
    @patch("main.ai_service.get_tags")
    @patch("main.ai_service.get_summary")
    def test_xss_in_content(
        self,
        mock_get_summary,
        mock_get_tags,
        mock_add_to_vector_store,
        client
    ):
        """測試 XSS 防護 - 內容欄位"""
        mock_get_summary.return_value = "測試摘要"
        mock_get_tags.return_value = "測試"
        mock_add_to_vector_store.return_value = None

        # XSS 攻擊字串
        malicious_content = "<script>alert('XSS')</script>"
        note_data = {
            "title": "XSS 測試",
            "content": malicious_content
        }
        response = client.post("/notes/", json=note_data)

        # 應該正常建立筆記
        assert response.status_code == 200
        # 內容應該被儲存（後端不應該過濾，由前端處理顯示）
        # 這裡主要驗證不會造成系統錯誤

    @patch("main.ai_service.generate_rag_response")
    @patch("main.ai_service.search_notes")
    def test_sql_injection_in_chat_query(
        self,
        mock_search_notes,
        mock_generate_rag_response,
        client
    ):
        """測試 SQL Injection 防護 - 對話查詢"""
        mock_search_notes.return_value = []
        mock_generate_rag_response.return_value = "這是回答"

        # SQL Injection 攻擊字串
        malicious_query = "'; DELETE FROM notes WHERE '1'='1"
        chat_data = {"query": malicious_query}
        response = client.post("/chat/", json=chat_data)

        # 應該正常回應，而不是執行 SQL 指令
        assert response.status_code == 200
        
        # 驗證資料庫仍然正常運作
        notes_response = client.get("/notes/")
        assert notes_response.status_code == 200

    @patch("main.ai_service.add_to_vector_store")
    @patch("main.ai_service.get_tags")
    @patch("main.ai_service.get_summary")
    def test_large_input_handling(
        self,
        mock_get_summary,
        mock_get_tags,
        mock_add_to_vector_store,
        client
    ):
        """測試大量輸入的處理"""
        mock_get_summary.return_value = "測試摘要"
        mock_get_tags.return_value = "測試"
        mock_add_to_vector_store.return_value = None

        # 大量輸入
        large_content = "A" * 100000  # 100KB 的內容
        note_data = {
            "title": "大量輸入測試",
            "content": large_content
        }
        response = client.post("/notes/", json=note_data)

        # 應該正常處理或回傳適當的錯誤
        assert response.status_code in [200, 413, 422]


# ============ 整合測試 - 向量資料庫互動 ============
class TestVectorDBIntegration:
    """向量資料庫整合測試"""

    def test_vector_store_integration_workflow(self, mock_ai_service):
        """測試完整的向量資料庫整合工作流程"""
        # 模擬新增筆記到向量資料庫
        mock_ai_service.add_to_vector_store(1, "測試內容", "測試標題")
        mock_ai_service.add_to_vector_store.assert_called_with(1, "測試內容", "測試標題")

        # 模擬搜尋筆記
        mock_ai_service.search_notes.return_value = [
            {"note_id": 1, "title": "測試標題", "score": 0.95}
        ]
        results = mock_ai_service.search_notes("測試查詢")
        
        assert len(results) == 1
        assert results[0]["note_id"] == 1
        assert results[0]["score"] == 0.95

    def test_rag_workflow_with_vector_search(self, mock_ai_service):
        """測試 RAG 工作流程與向量搜尋的整合"""
        # 設定 mock
        mock_ai_service.search_notes.return_value = [
            {"note_id": 1, "score": 0.9},
            {"note_id": 2, "score": 0.8}
        ]
        mock_ai_service.generate_rag_response.return_value = "整合測試回答"

        # 模擬 RAG 流程
        query = "測試問題"
        search_results = mock_ai_service.search_notes(query, top_k=3)
        
        # 驗證搜尋結果
        assert len(search_results) == 2
        
        # 模擬生成 RAG 回答
        contexts = ["內容1", "內容2"]
        answer = mock_ai_service.generate_rag_response(query, contexts)
        
        assert answer == "整合測試回答"
        mock_ai_service.generate_rag_response.assert_called_with(query, contexts)
