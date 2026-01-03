# Project Specification: Memo-Agent (Containerized AI Second Brain)

## 1. 系統概述 (System Overview)
這是一個基於 RAG (Retrieval-Augmented Generation) 的個人知識庫系統。
- 使用者透過 Web 介面新增 Markdown 筆記。
- 系統自動使用 LLM (Gemini) 進行總結、打標籤，並進行向量化存儲。
- 使用者可透過自然語言對話，系統檢索相關筆記並回答問題。
- **部署方式**：全系統完全 Docker 化，支援一鍵啟動。

## 2. 技術架構 (Tech Stack)
- **Backend API**: Python 3.10+, FastAPI
- **Frontend UI**: Streamlit
- **Database (Metadata)**: SQLite (via SQLModel)
- **Vector Store (Embeddings)**: ChromaDB (Local Persistent Client)
- **LLM Service**: Google Gemini 1.5 Flash (via `google-generativeai` SDK)
- **Containerization**: Docker, Docker Compose

## 3. 資料模型 (Data Schema)
必須使用 `SQLModel` 定義 Table。

### Table: `Note`
| Field | Type | Description |
| :--- | :--- | :--- |
| `id` | `int` | Primary Key, Auto-increment |
| `title` | `str` | 筆記標題 |
| `content` | `str` | 筆記原始內容 (Markdown) |
| `summary` | `str` | AI 自動生成的摘要 (約 50-100 字) |
| `tags` | `str` | AI 自動生成的標籤 (逗號分隔字串, e.g., "Python, AI, Study") |
| `created_at` | `datetime` | 建立時間 (UTC) |

*注意：ChromaDB 的儲存與 SQLite 分開，但必須透過 `id` 進行邏輯關聯。*

## 4. 核心功能 (User Stories & Logic)

### 4.1 新增筆記 (Create Note)
- **Input**: User 輸入 Title, Content。
- **Process**:
    1. 呼叫 Gemini API：生成 `summary` 和 `tags`。
    2. 呼叫 Gemini API (Embedding)：將 `content` 轉為向量。
    3. **Transaction**:
        - 寫入 SQLite (`Note` table)。
        - 寫入 ChromaDB (Collection: `notes`)，Metadata 包含 `note_id`, `title`。
- **Output**: 成功訊息與生成的摘要。

### 4.2 語意搜尋 (Semantic Search)
- **Input**: User 輸入 Query。
- **Process**:
    1. 將 Query 轉為向量。
    2. 查詢 ChromaDB，取回最相似的 Top 3 筆記。
- **Output**: 筆記列表 (包含標題、摘要、相似度分數)。

### 4.3 AI 對話 (RAG Chat)
- **Input**: User 輸入 Query。
- **Process**:
    1. 執行「語意搜尋」取得 Top 3 筆記內容 (Context)。
    2. 組合 Prompt：
       ```text
       你是我的個人知識助手。請根據以下參考資料回答使用者的問題。
       如果參考資料沒有答案，請說不知道。
       
       參考資料：
       {context_from_chromadb}
       
       使用者問題：{user_query}
       ```
    3. 呼叫 Gemini 生成回答。
- **Output**: AI 的回答串流 (Streaming response is preferred if possible, otherwise normal text)。

## 5. API 介面 (API Endpoints)
FastAPI 服務運行於 Port `8000`。

- `POST /notes/`: 建立筆記。
- `GET /notes/`: 取得所有筆記清單 (依時間排序)。
- `POST /chat/`: 接收 `{query: str}`，回傳 `{answer: str, sources: list}`。
- `GET /health/`: 健康檢查，回傳 `{"status": "ok"}`。

## 6. Docker 化架構 (Dockerization)
系統必須拆分為兩個服務容器，由 `docker-compose.yml` 管理。

### 6.1 Backend Container (`backend`)
- **Base Image**: `python:3.10-slim`
- **Workdir**: `/app/backend`
- **Command**: 使用 `uvicorn` 啟動 FastAPI server (`host 0.0.0.0`).
- **Volumes**: 
    - 掛載 `./data` 到 `/app/data` (確保 SQLite 和 ChromaDB 資料持久化)。
    - 掛載 `./backend` (開發時方便 hot-reload，生產環境則 COPY)。

### 6.2 Frontend Container (`frontend`)
- **Base Image**: `python:3.10-slim`
- **Workdir**: `/app/frontend`
- **Command**: 使用 `streamlit run app.py`。
- **Environment**:
    - `API_URL`: 指向 Backend 容器 (例如 `http://backend:8000`).
- **Ports**: Expose `8501`.

### 6.3 Docker Compose
- 定義 `backend` 和 `frontend` 兩個 services。
- 設定 Shared Network。
- 透過 `.env` 檔案傳遞 `GEMINI_API_KEY` 給 Backend。
- 確保 Frontend 依賴於 Backend 啟動 (`depends_on`).

## 7. 開發指引 (Implementation Guidelines)
1. **錯誤處理**: 所有 AI API 呼叫 (Gemini) 必須包含 `try-except`，若失敗需回傳清晰錯誤訊息，不可讓 Server Crash。
2. **目錄結構**:
/ 
├── docker-compose.yml 
├── backend/ │ 
├── Dockerfile │ 
├── main.py │
├── database.py (SQLite setup) │ 
├── ai_service.py (Gemini & Chroma logic) 
│ └── requirements.txt 
└── frontend/ 
├── Dockerfile 
├── app.py 
└── requirements.txt
3. **依賴管理**: Backend 與 Frontend 需有各自獨立的 `requirements.txt`。