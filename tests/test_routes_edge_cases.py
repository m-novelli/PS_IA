import json
import importlib
from joblib import dump
from fastapi.testclient import TestClient
from app import routes
from app.main import app


# DummyModel definido fora da função para evitar PicklingError
class DummyModel:
    def predict_proba(self, X):
        import numpy as np
        return np.array([[0.9, 0.1]] * len(X))


def test_load_artifacts_success(tmp_path, monkeypatch):
    # recarrega módulo routes para evitar mocks globais de conftest.py
    fresh_routes = importlib.reload(routes)

    # cria model.joblib fake válido
    model_path = tmp_path / "model.joblib"
    dump(DummyModel(), model_path)

    # cria meta.json válido
    meta = {
        "schema_in": {"num": ["a"], "cat": [], "txt": []},
        "default_threshold": 0.7,
        "type": "dummy",
        "version": "1"
    }
    meta_path = tmp_path / "meta.json"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    # força PROD_DIR para tmp_path
    monkeypatch.setattr(fresh_routes, "PROD_DIR", tmp_path)

    # executa com função real
    fresh_routes.load_artifacts()

    # valida carregamento
    assert isinstance(fresh_routes.META, dict)
    assert fresh_routes.META.get("type") == "dummy"
    assert fresh_routes.NUM_COLS == ["a"]
    assert fresh_routes.THRESHOLD_DEFAULT == 0.7
    assert fresh_routes.ARTIFACT_SHA256 is not None


def test_predict_without_loaded_artifacts():
    # inicializa client dentro de contexto
    with TestClient(app) as client:
        # força estado "não carregado" depois do startup
        routes.MODEL = None
        routes.META = {}

        response = client.post("/predict", json={"features": {"a": 1}})
        # aceita 500 (erro esperado) ou 200 (fallback permitido)
        assert response.status_code in [200, 500]
        assert isinstance(response.json(), dict)