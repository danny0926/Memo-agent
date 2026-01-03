import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_create_note():
    response = client.post("/notes/", json={"title": "Test Note", "content": "This is a test note."})
    assert response.status_code == 200
    assert response.json() == {"message": "Note created successfully"}

def test_get_notes():
    response = client.get("/notes/")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_chat():
    response = client.post("/chat/", json={"query": "What is the meaning of life?"})
    assert response.status_code == 200
    assert isinstance(response.json(), dict)
    assert "answer" in response.json()
    assert "sources" in response.json()

def test_health():
    response = client.get("/health/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
