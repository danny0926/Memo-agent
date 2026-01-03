import os
import os
from typing import List, Dict, Any
from openai import OpenAI
import chromadb
from dotenv import load_dotenv

load_dotenv()


class AIService:
    """AI 服務類別，整合 OpenAI API 和向量資料庫"""

    def __init__(self):
        # 設定 OpenAI API
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY 環境變數未設定")

        self.client = OpenAI(api_key=api_key)
        self.chat_model = "gpt-4o-mini"  # 使用較經濟的模型
        self.embedding_model = "text-embedding-3-small"
        # 設定向量資料庫
        data_dir = os.environ.get("DATA_DIR", "./data")
        chroma_path = os.path.join(data_dir, "chroma")
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.notes_collection = self.chroma_client.get_or_create_collection(name="notes")
        print(f"使用 ChromaDB 向量儲存，路徑: {chroma_path}")

    def get_summary(self, content: str) -> str:
        """使用 OpenAI 生成摘要（約 50-100 字）"""
        try:
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": "你是一個摘要助手，請用繁體中文生成簡潔的摘要。"},
                    {"role": "user", "content": f"請為以下內容生成一個簡潔的摘要，約 50-100 字：\n\n{content}"}
                ],
                max_tokens=200
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"生成摘要時發生錯誤: {e}")
            return content[:100] + "..." if len(content) > 100 else content

    def get_tags(self, content: str) -> str:
        """使用 OpenAI 生成標籤（逗號分隔）"""
        try:
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": "你是一個標籤生成助手，請用繁體中文生成相關標籤。"},
                    {"role": "user", "content": f"請為以下內容生成 3-5 個相關標籤，使用逗號分隔：\n\n{content}\n\n標籤（格式：標籤1, 標籤2, 標籤3）："}
                ],
                max_tokens=100
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"生成標籤時發生錯誤: {e}")
            return "未分類"

    def get_embedding(self, text: str) -> List[float]:
        """使用 OpenAI Embedding API 將文字轉為向量"""
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"生成向量時發生錯誤: {e}")
            raise

    def add_to_vector_store(self, note_id: int, content: str, title: str):
        """將筆記內容加入向量資料庫"""
        try:
            embedding = self.get_embedding(content)
            self.notes_collection.add(
                ids=[str(note_id)],
                embeddings=[embedding],
                metadatas=[{"note_id": note_id, "title": title}],
                documents=[content]
            )
        except Exception as e:
            print(f"加入向量資料庫時發生錯誤: {e}")
            raise

    def search_notes(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """語意搜尋筆記"""
        try:
            query_embedding = self.get_embedding(query)
            results = self.notes_collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )

            search_results = []
            if results and results['ids'] and results['ids'][0]:
                for i, id_ in enumerate(results['ids'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    distance = results['distances'][0][i] if results['distances'] else 0
                    search_results.append({
                        "note_id": metadata.get("note_id"),
                        "title": metadata.get("title"),
                        "score": 1 - distance  # 轉換距離為相似度分數
                    })

            return search_results
        except Exception as e:
            print(f"搜尋筆記時發生錯誤: {e}")
            return []

    def generate_rag_response(self, query: str, contexts: List[str]) -> str:
        """使用 RAG 方式生成回答"""
        try:
            context_text = "\n\n---\n\n".join(contexts) if contexts else "（無相關參考資料）"
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是我的個人知識助手。請根據提供的參考資料回答使用者的問題。如果參考資料沒有答案，請說不知道。請用繁體中文回答。"
                    },
                    {
                        "role": "user",
                        "content": f"參考資料：\n{context_text}\n\n使用者問題：{query}"
                    }
                ],
                max_tokens=1000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"生成回答時發生錯誤: {e}")
            return "抱歉，生成回答時發生錯誤，請稍後再試。"
