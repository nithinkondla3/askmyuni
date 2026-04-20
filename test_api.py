from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

with patch("rag_chain.FAISS") as mock_faiss:
    mock_faiss.load_local.return_value = MagicMock()
    from main import app

client = TestClient(app)

def test_query_returns_response():
    with patch("main.ask", return_value="This is a test answer"):
        response = client.post("/query", json={"question": "What is RMIT?"})
        assert response.status_code == 200
        assert response.json()["answer"] != ""