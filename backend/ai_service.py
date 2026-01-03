import os
from typing import List, Dict, Any, Tuple
from openai import OpenAI
import chromadb
from dotenv import load_dotenv
import re

load_dotenv()


class AIService:
    """AI 服務類別，整合 OpenAI API 和向量資料庫"""

    # 分塊設定
    CHUNK_SIZE = 1500  # 每個分塊的字數
    CHUNK_OVERLAP = 200  # 分塊重疊的字數

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
        """語意搜尋筆記（返回 note_id 和 title）"""
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

    def search_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        語意搜尋並直接返回 chunk 內容（RAG 用）
        
        Args:
            query: 搜尋查詢
            top_k: 返回的結果數量
        
        Returns:
            包含 chunk 內容和 metadata 的列表
        """
        try:
            query_embedding = self.get_embedding(query)
            results = self.notes_collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )

            search_results = []
            if results and results['ids'] and results['ids'][0]:
                for i, id_ in enumerate(results['ids'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    distance = results['distances'][0][i] if results['distances'] else 0
                    document = results['documents'][0][i] if results['documents'] else ""
                    
                    search_results.append({
                        "chunk_id": id_,
                        "note_id": metadata.get("note_id"),
                        "title": metadata.get("title"),
                        "chunk_index": metadata.get("chunk_index", 0),
                        "total_chunks": metadata.get("total_chunks", 1),
                        "content": document,
                        "score": 1 - distance
                    })

            return search_results
        except Exception as e:
            print(f"搜尋 chunks 時發生錯誤: {e}")
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

    def chunk_text(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """
        將長文本分割成較小的區塊
        
        Args:
            text: 要分割的文本
            chunk_size: 每個區塊的字數（預設使用 CHUNK_SIZE）
            overlap: 區塊重疊的字數（預設使用 CHUNK_OVERLAP）
        
        Returns:
            分割後的文本區塊列表
        """
        chunk_size = chunk_size or self.CHUNK_SIZE
        overlap = overlap or self.CHUNK_OVERLAP
        
        # 清理文本
        text = re.sub(r'\s+', ' ', text).strip()
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # 如果不是最後一塊，嘗試在句號、問號、驚嘆號處斷開
            if end < len(text):
                # 在 chunk_size 範圍內找最後一個句子結尾
                last_period = max(
                    text.rfind('。', start, end),
                    text.rfind('！', start, end),
                    text.rfind('？', start, end),
                    text.rfind('. ', start, end),
                    text.rfind('! ', start, end),
                    text.rfind('? ', start, end),
                    text.rfind('\n', start, end)
                )
                if last_period > start + chunk_size // 2:
                    end = last_period + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # 下一塊的起始位置（考慮重疊）
            start = end - overlap if end < len(text) else end
        
        return chunks

    def add_chunks_to_vector_store(self, note_id: int, chunks: List[str], title: str) -> int:
        """
        將多個文本區塊加入向量資料庫
        
        Args:
            note_id: 筆記 ID
            chunks: 文本區塊列表
            title: 筆記標題
        
        Returns:
            成功加入的區塊數量
        """
        success_count = 0
        for i, chunk in enumerate(chunks):
            try:
                chunk_id = f"{note_id}_chunk_{i}"
                embedding = self.get_embedding(chunk)
                self.notes_collection.add(
                    ids=[chunk_id],
                    embeddings=[embedding],
                    metadatas=[{
                        "note_id": note_id,
                        "title": title,
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    }],
                    documents=[chunk]
                )
                success_count += 1
            except Exception as e:
                print(f"加入區塊 {i} 時發生錯誤: {e}")
        
        return success_count

    def process_long_document(self, content: str, title: str, note_id: int) -> Dict[str, Any]:
        """
        處理長文檔：分塊並加入向量資料庫
        
        Args:
            content: 文檔內容
            title: 文檔標題
            note_id: 筆記 ID
        
        Returns:
            處理結果，包含區塊數量等資訊
        """
        # 分塊
        chunks = self.chunk_text(content)
        
        # 加入向量資料庫
        success_count = self.add_chunks_to_vector_store(note_id, chunks, title)
        
        return {
            "total_chunks": len(chunks),
            "success_chunks": success_count,
            "content_length": len(content)
        }

    def get_hierarchical_summary(self, content: str, max_chunk_size: int = 8000) -> str:
        """
        對長文檔進行階層式摘要
        
        Args:
            content: 文檔內容
            max_chunk_size: 每次摘要的最大字數
        
        Returns:
            最終摘要
        """
        # 如果內容不長，直接摘要
        if len(content) <= max_chunk_size:
            return self.get_summary(content)
        
        # 分成多個區塊分別摘要
        chunks = self.chunk_text(content, chunk_size=max_chunk_size, overlap=500)
        
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            print(f"正在摘要第 {i+1}/{len(chunks)} 區塊...")
            try:
                summary = self.get_summary(chunk)
                chunk_summaries.append(f"[第{i+1}部分] {summary}")
            except Exception as e:
                print(f"區塊 {i+1} 摘要失敗: {e}")
                continue
        
        # 合併所有摘要再做一次總結
        combined_summaries = "\n\n".join(chunk_summaries)
        
        if len(combined_summaries) > max_chunk_size:
            # 如果合併後仍然太長，遞迴處理
            return self.get_hierarchical_summary(combined_summaries, max_chunk_size)
        
        # 生成最終摘要
        try:
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": "你是一個摘要助手。請根據以下多個部分的摘要，生成一個完整、連貫的總結。請用繁體中文，約 100-200 字。"},
                    {"role": "user", "content": f"請根據以下分段摘要，生成一個完整的總結：\n\n{combined_summaries}"}
                ],
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"生成最終摘要時發生錯誤: {e}")
            return chunk_summaries[0] if chunk_summaries else "無法生成摘要"
