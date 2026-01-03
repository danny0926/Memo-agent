"""
Pytest 設定檔
"""
import os
import sys
import pytest

# 設定測試環境變數
os.environ["GEMINI_API_KEY"] = "test_api_key_for_testing"
os.environ["DATA_DIR"] = "./test_data"

# 確保可以 import backend 模組
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """設定測試環境"""
    # 建立測試資料目錄
    os.makedirs("./test_data", exist_ok=True)
    
    yield
    
    # 清理測試資料
    import shutil
    if os.path.exists("./test_data"):
        shutil.rmtree("./test_data")


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
    
    return mock_service
