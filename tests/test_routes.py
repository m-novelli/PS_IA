import pytest

def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "threshold_default" in data
    assert "schema_in" in data


def test_schema_endpoint(client):
    resp = client.get("/schema")
    assert resp.status_code == 200
    data = resp.json()
    assert "num" in data
    assert "cat" in data
    assert "txt" in data
    assert "threshold_default" in data


def test_predict_endpoint_success(client):
    payload = {
        "features": {"foo": 123, "bar": "abc"},
        "meta": {"external_id": "cand1"}
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "prediction" in data
    assert "model" in data
    assert isinstance(data["prediction"]["prob_next_phase"], float)


def test_predict_endpoint_invalid_payload(client):
    # manda payload vazio → deve falhar
    resp = client.post("/predict", json={})
    assert resp.status_code == 422  # erro de validação do Pydantic





def test_rank_and_suggest_success(client):
    payload = {
        "vaga": {"descricao": "Engenheiro de Dados"},
        "candidatos": [
            {"meta": {"external_id": "cand1"}, "candidato": {"skill": "SQL"}},
            {"meta": {"external_id": "cand2"}, "candidato": {"skill": "Python"}},
        ],
    }
    resp = client.post("/rank-and-suggest?include_questions=true", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert "questions" in data and data["questions"] is not None
    assert data["questions"].get("common_questions")



def test_rank_and_suggest_no_candidates(client):
    payload = {"vaga": {"descricao": "Engenheiro de Dados"}, "candidatos": []}
    resp = client.post("/rank-and-suggest", json=payload)
    assert resp.status_code == 400
    data = resp.json()
    assert data["detail"] == "Nenhum candidato enviado."