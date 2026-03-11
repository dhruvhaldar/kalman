from fastapi.testclient import TestClient
from api.index import app

client = TestClient(app)

def test_analyze_endpoint():
    response = client.post(
        "/api/analyze",
        json={
            "A": [[0, 1], [-2, -3]],
            "B": [[0], [1]],
            "C": [[1, 0]],
            "D": [[0]],
            "discrete": False
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert data["controllable"] == True
    assert data["observable"] == True
    assert data["stable"] == True
    assert data["minimal"] == True

def test_lqr_endpoint():
    response = client.post(
        "/api/lqr",
        json={
            "A": [[0, 1], [-2, -3]],
            "B": [[0], [1]],
            "Q": [[1, 0], [0, 1]],
            "R": [[1]],
            "discrete": False
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert "K" in data
    assert "P" in data

    K = data["K"]
    assert len(K) == 1
    assert len(K[0]) == 2

def test_kalman_endpoint():
    response = client.post(
        "/api/kalman",
        json={
            "A": [[1, 0.1], [0, 1]],
            "B": [[0], [0.1]],
            "C": [[1, 0]],
            "D": [[0]],
            "Q": [[0.01, 0], [0, 0.01]],
            "R": [[0.1]],
            "x0": [0, 0],
            "P0": [[1, 0], [0, 1]],
            "time_steps": 10
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert "x_true" in data
    assert "y_meas" in data
    assert "x_est" in data
    assert "P_est" in data

    assert len(data["x_true"]) == 10
    assert len(data["x_est"]) == 10
