import os
import json
import math
from typing import List, Dict, Any, Optional
from openai import OpenAI


class SimpleVectorStore:
    """簡單的向量資料庫（使用 JSON 檔案儲存）
    
    這是一個輕量級的實作，適合開發和小規模使用。
    生產環境建議使用 ChromaDB 或其他向量資料庫。
    """
    
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.data_file = os.path.join(storage_path, "vectors.json")
        os.makedirs(storage_path, exist_ok=True)
        self.data = self._load_data()
    
    def _load_data(self) -> Dict:
        """載入現有資料"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {"vectors": [], "metadata": [], "documents": [], "ids": []}
        return {"vectors": [], "metadata": [], "documents": [], "ids": []}
    
    def _save_data(self):
        """儲存資料到檔案"""
        with open(self.data_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
    
    def add(self, ids: List[str], embeddings: List[List[float]], 
            metadatas: List[Dict], documents: List[str]):
        """新增向量資料"""
        for i, id_ in enumerate(ids):
            # 如果 ID 已存在，先移除舊資料
            if id_ in self.data["ids"]:
                idx = self.data["ids"].index(id_)
                self.data["ids"].pop(idx)
                self.data["vectors"].pop(idx)
                self.data["metadata"].pop(idx)
                self.data["documents"].pop(idx)
            
            self.data["ids"].append(id_)
            self.data["vectors"].append(embeddings[i])
            self.data["metadata"].append(metadatas[i])
            self.data["documents"].append(documents[i])
        
        self._save_data()
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """計算餘弦相似度"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def query(self, query_embedding: List[float], n_results: int = 3) -> Dict:
        """查詢最相似的向量"""
        if not self.data["vectors"]:
            return {"ids": [[]], "metadatas": [[]], "distances": [[]], "documents": [[]]}
        
        # 計算所有向量的相似度
        similarities = []
        for i, vec in enumerate(self.data["vectors"]):
            sim = self._cosine_similarity(query_embedding, vec)
            similarities.append((i, sim))
        
        # 按相似度排序（降序）
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 取前 n_results 個
        top_results = similarities[:n_results]
        
        result_ids = []
        result_metadatas = []
        result_distances = []
        result_documents = []
        
        for idx, sim in top_results:
            result_ids.append(self.data["ids"][idx])
            result_metadatas.append(self.data["metadata"][idx])
            result_distances.append(1 - sim)  # 轉換為距離
            result_documents.append(self.data["documents"][idx])
        
        return {
            "ids": [result_ids],
            "metadatas": [result_metadatas],
            "distances": [result_distances],
            "documents": [result_documents]
        }


class AIService:
    """AI 服務類別，整合 OpenAI API 和向量資料庫"""
    
    def __init__(self):
        # 載入環境變數
        from dotenv import load_dotenv
        load_dotenv()
        
        # 設定 OpenAI API
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY 環境變數未設定")
        
        self.client = OpenAI(api_key=api_key)
        self.chat_model = "gpt-4o-mini"  # 使用較經濟的模型
        self.embedding_model = "text-embedding-3-small"
        
        # 設定向量資料庫
        data_dir = os.environ.get("DATA_DIR", "./data")
        vector_store_path = os.path.join(data_dir, "vector_store")
        self.vector_store = SimpleVectorStore(vector_store_path)
        print(f"使用簡單的 JSON 向量儲存，路徑: {vector_store_path}")
    
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
            self.vector_store.add(
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
            results = self.vector_store.query(
                query_embedding=query_embedding,
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
        context_text = "\n\n---\n\n".join(contexts) if contexts else "（無相關參考資料）"
        
        try:
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
