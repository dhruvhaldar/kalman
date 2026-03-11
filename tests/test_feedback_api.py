import pytest
from fastapi.testclient import TestClient
from api.index import app

client = TestClient(app)

def test_feedback_endpoint():
    response = client.post(
        "/api/feedback",
        json={
            "A": [[0, 1], [-2, -3]],
            "B": [[0], [1]],
            "poles": [[-1, 1], [-1, -1]],
            "discrete": False
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert "K" in data
    assert "eig_open" in data
    assert "eig_closed" in data

    K = data["K"]
    assert len(K) == 1
    assert len(K[0]) == 2
