import sys
from unittest.mock import patch, MagicMock

# Block rag_chain from loading entirely
sys.modules["rag_chain"] = MagicMock()

from main import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_query_returns_response():
    with patch("main.ask", return_value=("This is a test answer", [1, 2])):
        response = client.post("/query", json={"question": "What is RMIT?"})
        assert response.status_code == 200