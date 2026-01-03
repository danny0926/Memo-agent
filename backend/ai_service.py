import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
from chromadb import Client

# Gemini API
SCOPES = ['https://www.googleapis.com/auth/cloud-platform']
SERVICE_ACCOUNT_FILE = 'path/to/service_account_key.json'

credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES)
gemini_service = build('gemini', 'v1', credentials=credentials)

# ChromaDB
CHROMA_DB_URL = 'http://localhost:8000'
chroma_client = Client(CHROMA_DB_URL)

def get_summary(content):
    # 呼叫 Gemini API 生成摘要
    request = gemini_service.summarize().body(content=content)
    response = request.execute()
    return response['summary']

def get_tags(content):
    # 呼叫 Gemini API 生成標籤
    request = gemini_service.tag().body(content=content)
    response = request.execute()
    return response['tags']

def embed_content(content):
    # 呼叫 Gemini API 將內容轉為向量
    request = gemini_service.embed().body(content=content)
    response = request.execute()
    return response['vector']

def search_notes(query):
    # 將查詢轉為向量
    query_vector = embed_content(query)
    # 查詢 ChromaDB
    results = chroma_client.search(query_vector, k=3)
    return results
