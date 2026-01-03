# 🧠 Memo-Agent

基於 RAG (Retrieval-Augmented Generation) 的個人知識庫系統。

## ✨ 功能特色

- **📝 智慧筆記**: 新增 Markdown 筆記，AI 自動生成摘要和標籤
- **🔍 語意搜尋**: 使用自然語言搜尋相關筆記
- **💬 AI 對話**: 與你的知識庫對話，獲得基於筆記內容的回答
- **🐳 容器化部署**: Docker Compose 一鍵啟動

## 🛠️ 技術架構

- **Backend**: FastAPI + Python 3.10
- **Frontend**: Streamlit
- **Database**: SQLite (via SQLModel)
- **Vector Store**: ChromaDB
- **LLM**: Google Gemini 1.5 Flash

## 🚀 快速開始

### 前置需求

- Docker & Docker Compose
- Gemini API Key ([取得 API Key](https://makersuite.google.com/app/apikey))

### 安裝步驟

1. **Clone 專案**
   ```bash
   git clone https://github.com/YOUR_USERNAME/CLI_agent.git
   cd CLI_agent
   ```

2. **設定環境變數**
   ```bash
   cp .env.example .env
   # 編輯 .env 檔案，填入你的 GEMINI_API_KEY
   ```

3. **啟動服務**
   ```bash
   docker-compose up -d --build
   ```

4. **開啟瀏覽器**
   - Frontend UI: http://localhost:8501
   - Backend API: http://localhost:8000
   - API 文件: http://localhost:8000/docs

## 📁 專案結構

```
├── backend/
│   ├── main.py          # FastAPI 主程式
│   ├── database.py      # SQLModel 資料模型
│   ├── ai_service.py    # Gemini AI 服務
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── app.py           # Streamlit 介面
│   ├── requirements.txt
│   └── Dockerfile
├── data/                # 資料持久化目錄
├── docker-compose.yml
├── .env.example
├── SPEC.md              # 專案規格書
└── README.md
```

## 📡 API 端點

| Method | Endpoint | 說明 |
|--------|----------|------|
| POST | `/notes/` | 建立新筆記 |
| GET | `/notes/` | 取得所有筆記 |
| POST | `/chat/` | RAG 對話 |
| GET | `/health/` | 健康檢查 |

## 🔧 本地開發

### 使用虛擬環境

```bash
# 建立虛擬環境
python -m venv .venv

# 啟用虛擬環境 (Windows)
.venv\Scripts\activate

# 啟用虛擬環境 (Linux/Mac)
source .venv/bin/activate

# 安裝依賴
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
```

### 執行服務

```bash
# 啟動 Backend (在一個終端機)
cd backend
uvicorn main:app --reload

# 啟動 Frontend (在另一個終端機)
cd frontend
streamlit run app.py
```

## 📄 License

MIT License
