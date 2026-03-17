def test_health_returns_200(client):
    resp = client.get("/health")
    assert resp.status_code == 200


def test_health_body(client):
    resp = client.get("/health")
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_health_method_not_allowed(client):
    resp = client.post("/health")
    assert resp.status_code == 405
