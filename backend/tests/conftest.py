"""
Pytest 設定檔
"""
import os
import sys
import pytest
from unittest.mock import MagicMock

# 設定測試環境變數
os.environ["GEMINI_API_KEY"] = "test_api_key_for_testing"
os.environ["OPENAI_API_KEY"] = "test_openai_api_key_for_testing"
os.environ["DATA_DIR"] = "./test_data"

# 確保可以 import backend 模組
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock chromadb 以避免 Python 3.14 相容性問題
# chromadb 依賴 onnxruntime，而 onnxruntime 還不支援 Python 3.14
if sys.version_info >= (3, 14):
    mock_chromadb = MagicMock()
    mock_chromadb.PersistentClient.return_value.get_or_create_collection.return_value = MagicMock()
    sys.modules['chromadb'] = mock_chromadb


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """設定測試環境"""
    import shutil
    import time
    
    # 清理舊的測試資料（如果存在）
    if os.path.exists("./test_data"):
        try:
            shutil.rmtree("./test_data")
        except PermissionError:
            pass  # 忽略權限錯誤
    
    # 建立測試資料目錄
    os.makedirs("./test_data", exist_ok=True)
    
    yield
    
    # 清理測試資料
    from database import engine
    engine.dispose()  # 關閉資料庫連線
    
    # 等待一下讓資源釋放
    time.sleep(0.5)
    
    # 嘗試清理，失敗也沒關係
    if os.path.exists("./test_data"):
        try:
            shutil.rmtree("./test_data")
        except PermissionError:
            print("警告：無法刪除 test_data 目錄，可能被其他進程佔用")


@pytest.fixture
def mock_ai_service():
    """提供 Mock AI 服務的 fixture"""
    from unittest.mock import MagicMock
    
    mock_service = MagicMock()
    mock_service.get_summary.return_value = "這是測試摘要"
    mock_service.get_tags.return_value = "測試, Python"
    mock_service.get_embedding.return_value = [0.1] * 768
    mock_service.search_notes.return_value = [
        {"note_id": 1, "title": "測試筆記", "score": 0.95}
    ]
    mock_service.generate_rag_response.return_value = "這是 AI 回答"
    
    # 分塊相關的 mock
    def mock_chunk_text(text, chunk_size=1500, overlap=200):
        """模擬分塊功能"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - overlap if end < len(text) else end
        return chunks
    
    mock_service.chunk_text = mock_chunk_text
    mock_service.add_chunks_to_vector_store.return_value = 10
    mock_service.process_long_document.return_value = {
        "total_chunks": 10,
        "success_chunks": 10,
        "content_length": 15000
    }
    mock_service.get_hierarchical_summary.return_value = "這是階層式摘要"
    
    return mock_service
