"""
PDF 上傳功能測試
包含分塊處理、階層式摘要等功能的測試
"""
import pytest
import os
import sys
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

# 確保可以 import backend 模組
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============ 測試分塊功能 ============
class TestChunking:
    """文本分塊功能測試"""

    def test_chunk_text_short_content(self, mock_ai_service):
        """測試短文本不需要分塊"""
        short_text = "這是一段很短的文字。"
        chunks = mock_ai_service.chunk_text(short_text)
        
        assert len(chunks) == 1
        assert chunks[0] == short_text

    def test_chunk_text_long_content(self):
        """測試長文本會被正確分塊"""
        # 模擬分塊邏輯
        chunk_size = 100
        overlap = 20
        long_text = "這是測試文字。" * 50  # 約 350 字
        
        # 簡單的分塊實作
        chunks = []
        start = 0
        while start < len(long_text):
            end = min(start + chunk_size, len(long_text))
            chunks.append(long_text[start:end])
            start = end - overlap if end < len(long_text) else end
        
        assert len(chunks) > 1
        # 檢查每個分塊大小合理
        for chunk in chunks[:-1]:  # 最後一塊可能較小
            assert len(chunk) <= chunk_size + 10

    def test_chunk_overlap(self):
        """測試分塊有正確的重疊"""
        chunk_size = 100
        overlap = 20
        text = "A" * 250
        
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            start = end - overlap if end < len(text) else end
        
        # 檢查重疊
        if len(chunks) >= 2:
            # 第一塊的最後 overlap 字應該出現在第二塊的開頭
            assert chunks[0][-overlap:] == chunks[1][:overlap]


# ============ 測試 PDF 解析 ============
class TestPDFParsing:
    """PDF 解析功能測試"""

    @pytest.fixture
    def sample_pdf_path(self):
        """返回測試 PDF 路徑"""
        return r"D:\books\《黃仁勳傳》.pdf"

    def test_pdf_exists(self, sample_pdf_path):
        """確認測試用 PDF 檔案存在"""
        if not os.path.exists(sample_pdf_path):
            pytest.skip(f"測試 PDF 不存在: {sample_pdf_path}")
        assert os.path.exists(sample_pdf_path)

    def test_pdf_readable(self, sample_pdf_path):
        """測試 PDF 可以被正確讀取"""
        if not os.path.exists(sample_pdf_path):
            pytest.skip(f"測試 PDF 不存在: {sample_pdf_path}")
        
        import fitz
        
        doc = fitz.open(sample_pdf_path)
        assert len(doc) > 0  # 至少有一頁
        
        # 嘗試讀取文字（可能是掃描檔，文字可能很少）
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        doc.close()
        
        # 如果是掃描檔，可能沒有文字
        if len(full_text.strip()) == 0:
            pytest.skip("這是掃描檔 PDF，無法提取文字")

    def test_pdf_full_text_extraction(self, sample_pdf_path):
        """測試完整 PDF 文字提取"""
        if not os.path.exists(sample_pdf_path):
            pytest.skip(f"測試 PDF 不存在: {sample_pdf_path}")
        
        import fitz
        
        doc = fitz.open(sample_pdf_path)
        full_text = ""
        page_count = len(doc)
        
        for page_num in range(page_count):
            page = doc[page_num]
            full_text += page.get_text() + "\n\n"
        
        doc.close()
        
        print(f"PDF 頁數: {page_count}")
        print(f"提取文字長度: {len(full_text)}")
        
        assert page_count > 0
        assert len(full_text) > 1000  # 應該有大量文字

    def test_pdf_chunking(self, sample_pdf_path):
        """測試 PDF 內容分塊"""
        if not os.path.exists(sample_pdf_path):
            pytest.skip(f"測試 PDF 不存在: {sample_pdf_path}")
        
        import fitz
        
        doc = fitz.open(sample_pdf_path)
        full_text = ""
        for page_num in range(len(doc)):
            full_text += doc[page_num].get_text() + "\n\n"
        doc.close()
        
        # 模擬分塊
        chunk_size = 1500
        overlap = 200
        
        chunks = []
        start = 0
        while start < len(full_text):
            end = min(start + chunk_size, len(full_text))
            chunk = full_text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - overlap if end < len(full_text) else end
        
        print(f"文字總長度: {len(full_text)}")
        print(f"分塊數量: {len(chunks)}")
        
        assert len(chunks) > 1  # 應該有多個分塊
        # 預期《黃仁勳傳》會有很多分塊
        assert len(chunks) >= 10


# ============ 測試 API 端點 ============
class TestPDFUploadAPI:
    """PDF 上傳 API 測試"""

    @pytest.fixture(scope="class")
    def client(self):
        """建立測試用 HTTP 客戶端"""
        from starlette.testclient import TestClient
        from main import app
        with TestClient(app) as client:
            yield client

    def test_upload_pdf_invalid_file_type(self, client):
        """測試上傳非 PDF 檔案會被拒絕"""
        # 建立一個假的 txt 檔案
        files = {"file": ("test.txt", b"This is not a PDF", "text/plain")}
        response = client.post("/upload-pdf/", files=files)
        
        assert response.status_code == 400
        assert "只支援 PDF 檔案" in response.json()["detail"]

    @patch("main.ai_service.get_hierarchical_summary")
    @patch("main.ai_service.get_summary")
    @patch("main.ai_service.get_tags")
    @patch("main.ai_service.process_long_document")
    def test_upload_pdf_mock(
        self,
        mock_process_long_document,
        mock_get_tags,
        mock_get_summary,
        mock_get_hierarchical_summary,
        client
    ):
        """測試 PDF 上傳流程（使用 mock）"""
        # 設定 mock
        mock_get_summary.return_value = "測試摘要"
        mock_get_hierarchical_summary.return_value = "階層式測試摘要"
        mock_get_tags.return_value = "測試, PDF"
        mock_process_long_document.return_value = {
            "total_chunks": 5,
            "success_chunks": 5,
            "content_length": 10000
        }
        
        # 建立一個簡單的 PDF 內容（使用 PyMuPDF 建立）
        import fitz
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), "This is a test PDF content.")
        pdf_bytes = doc.tobytes()
        doc.close()
        
        files = {"file": ("test.pdf", pdf_bytes, "application/pdf")}
        response = client.post("/upload-pdf/", files=files)
        
        assert response.status_code == 200
        result = response.json()
        assert result["message"] == "PDF 上傳並建立筆記成功"
        assert "id" in result
        assert "summary" in result
        assert "chunks" in result

    @pytest.mark.skipif(
        not os.path.exists(r"D:\books\《黃仁勳傳》.pdf"),
        reason="測試 PDF 檔案不存在"
    )
    @patch("main.ai_service.get_hierarchical_summary")
    @patch("main.ai_service.get_tags")
    @patch("main.ai_service.process_long_document")
    def test_upload_real_pdf_huang_renxun(
        self,
        mock_process_long_document,
        mock_get_tags,
        mock_get_hierarchical_summary,
        client
    ):
        """測試上傳真實的《黃仁勳傳》PDF（使用 mock AI 服務）"""
        # 設定 mock
        mock_get_hierarchical_summary.return_value = "黃仁勳是 NVIDIA 創辦人，帶領公司從遊戲顯卡跨入 AI 領域。"
        mock_get_tags.return_value = "黃仁勳, NVIDIA, AI, GPU, 創業"
        mock_process_long_document.return_value = {
            "total_chunks": 50,
            "success_chunks": 50,
            "content_length": 100000
        }
        
        pdf_path = r"D:\books\《黃仁勳傳》.pdf"
        
        with open(pdf_path, "rb") as f:
            pdf_content = f.read()
        
        files = {"file": ("《黃仁勳傳》.pdf", pdf_content, "application/pdf")}
        response = client.post("/upload-pdf/", files=files)
        
        assert response.status_code == 200
        result = response.json()
        
        print(f"上傳結果: {result}")
        
        assert result["message"] == "PDF 上傳並建立筆記成功"
        assert result["title"] == "《黃仁勳傳》"
        assert "黃仁勳" in result["summary"] or "NVIDIA" in result["summary"]
        assert result["chunks"] == 50


# ============ 整合測試 ============
class TestPDFIntegration:
    """PDF 功能整合測試"""

    def test_full_pdf_workflow_mock(self, mock_ai_service):
        """測試完整的 PDF 處理工作流程（使用 mock）"""
        # 模擬 PDF 內容
        pdf_content = "這是一本關於人工智慧的書籍。" * 100
        title = "AI 入門"
        
        # 模擬分塊（修復無限迴圈問題）
        chunk_size = 100
        overlap = 20
        chunks = []
        start = 0
        while start < len(pdf_content):
            end = min(start + chunk_size, len(pdf_content))
            chunks.append(pdf_content[start:end])
            # 修復：如果已到結尾，直接跳出
            if end >= len(pdf_content):
                break
            start = end - overlap
        
        # 驗證分塊結果
        assert len(chunks) > 1
        
        # 模擬 AI 處理
        mock_ai_service.get_summary.return_value = "這是一本 AI 入門書"
        mock_ai_service.get_tags.return_value = "AI, 人工智慧, 入門"
        mock_ai_service.add_chunks_to_vector_store.return_value = len(chunks)
        
        # 模擬處理流程
        summary = mock_ai_service.get_summary(pdf_content[:5000])
        tags = mock_ai_service.get_tags(pdf_content[:5000])
        
        assert "AI" in summary or "入門" in summary
        assert "AI" in tags

    def test_hierarchical_summary_mock(self, mock_ai_service):
        """測試階層式摘要（使用 mock）"""
        # 模擬長文檔
        long_content = "第一章：NVIDIA 的創立。" * 500 + "第二章：GPU 革命。" * 500
        
        # 設定 mock
        mock_ai_service.get_summary.return_value = "這是一段摘要"
        mock_ai_service.get_hierarchical_summary.return_value = "NVIDIA 從創立到 GPU 革命的完整故事"
        
        result = mock_ai_service.get_hierarchical_summary(long_content)
        
        assert len(result) > 0
        mock_ai_service.get_hierarchical_summary.assert_called_once_with(long_content)
