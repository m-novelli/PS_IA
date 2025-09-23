import pytest
import importlib
from fastapi.testclient import TestClient
from app import routes
from app.main import create_app
from fastapi import HTTPException


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


def test_load_artifacts_runtime_error(monkeypatch):
    """Força load_artifacts a levantar RuntimeError quando não há arquivos e TESTING != 1."""
    # Remove o TESTING imposto pelo conftest
    monkeypatch.delenv("TESTING", raising=False)

    # Recarrega o módulo para restaurar a função original sem os mocks
    import app.routes
    importlib.reload(app.routes)

    # Agora sim, aponta para um diretório inexistente
    app.routes.PROD_DIR = app.routes.BASE_DIR / "models" / "nao_existe"
    with pytest.raises(RuntimeError, match="Modelo ou metadados não encontrados."):
        app.routes.load_artifacts()


def test_prepare_dataframe_runtime_error(monkeypatch):
    """Garante que _prepare_dataframe levanta RuntimeError se META=None e não está em modo de teste."""
    monkeypatch.delenv("TESTING", raising=False)
    monkeypatch.setattr(routes, "META", None)
    with pytest.raises(RuntimeError, match="Artefatos não carregados."):
        routes._prepare_dataframe([])


def test_predict_df_no_model(monkeypatch):
    """Garante erro quando MODEL=None e não está em TESTING."""
    monkeypatch.delenv("TESTING", raising=False)
    monkeypatch.setattr(routes, "MODEL", None)
    import pandas as pd
    df = pd.DataFrame([{"a": 1}])
    with pytest.raises(RuntimeError, match="Modelo não carregado."):
        routes._predict_df(df, 0.5)


def test_predict_df_no_predict_proba(monkeypatch):
    """Garante erro quando o modelo não tem predict_proba."""
    class Dummy:
        pass

    monkeypatch.setattr(routes, "MODEL", Dummy())
    import pandas as pd
    df = pd.DataFrame([{"a": 1}])
    with pytest.raises(RuntimeError, match="Modelo não suporta predict_proba."):
        routes._predict_df(df, 0.5)


def test_rank_and_suggest_generic_exception(monkeypatch, client):
    """Simula falha genérica dentro de rank_and_suggest para cair no except Exception."""
    def fake_suggest_questions(_):
        raise Exception("falha simulada")

    monkeypatch.setattr(routes, "suggest_questions", fake_suggest_questions)

    payload = {
        "vaga": {"descricao": "Engenheiro de Dados"},
        "candidatos": [{"meta": {"external_id": "cand1"}, "candidato": {"skill": "SQL"}}],
    }
    resp = client.post("/rank-and-suggest", json=payload)
    assert resp.status_code == 400
    assert "falha simulada" in resp.json()["detail"]