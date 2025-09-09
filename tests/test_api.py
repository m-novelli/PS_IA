def _build_minimal_features(schema: dict) -> dict:
    feats = {}
    if schema["num"]:
        feats[schema["num"][0]] = 30
    if schema["cat"]:
        feats[schema["cat"][0]] = "M"
    if schema["txt"]:
        feats[schema["txt"][0]] = "analista de dados com python e sql"
    return feats

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "artifact" in body

def test_schema(client):
    r = client.get("/schema")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["num"], list)
    assert "threshold_default" in body

def test_predict_one(client):
    schema = client.get("/schema").json()
    payload = {"meta": {"external_id": "abc-123"},
               "features": _build_minimal_features(schema)}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200, r.text
    pred = r.json()["prediction"]
    assert 0.0 <= pred["prob_next_phase"] <= 1.0
    assert pred["label"] in (0, 1)

def test_predict_batch(client):
    schema = client.get("/schema").json()
    payload = {"items": [
        {"meta": {"external_id": "a"}, "features": _build_minimal_features(schema)},
        {"meta": {"external_id": "b"}, "features": _build_minimal_features(schema)}
    ]}
    r = client.post("/predict-batch", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    assert "results" in data and len(data["results"]) == 2
