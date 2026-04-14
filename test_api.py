from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_query_returns_answer():
    response = client.post(
        "/query",
        json={"question": "What is RMIT?"}
    )
    assert response.status_code == 200
    assert len(response.json()["answer"]) > 0