# tests/test_routes_extra.py
import pandas as pd
import pytest

import app.routes as routes
from app.suggest import SuggestQuestionsResponse


# =========================
# /health e /schema
# =========================
def test_health_smoke(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert "status" in body
    assert "schema_in" in body
    assert "threshold_default" in body


def test_schema_fields(client):
    r = client.get("/schema")
    assert r.status_code == 200
    body = r.json()
    # estes campos devem sempre existir
    for k in ["cat", "txt", "threshold_default", "version"]:
        assert k in body


# =========================
# _prepare_dataframe
# =========================
def test_prepare_dataframe_requires_meta(monkeypatch):
    # força META ausente
    routes.META = None
    with pytest.raises(RuntimeError, match="Artefatos não carregados"):
        routes._prepare_dataframe([routes.PredictItem(meta=None, features={"a": 1})])


def test_prepare_dataframe_full_warnings(monkeypatch):
    # configura schema mínimo
    routes.META = {"schema_in": {"cat": ["sexo"], "txt": ["cv"]}}
    routes.CAT_COLS[:] = ["sexo"]
    routes.TXT_COLS[:] = ["cv"]
    # força exibição "full"
    monkeypatch.setenv("SHOW_WARNINGS", "full")

    df, warns = routes._prepare_dataframe([
        routes.PredictItem(meta={"external_id": "x"}, features={"cv": "texto", "extra": 123})
    ])
    assert isinstance(df, pd.DataFrame)
    assert isinstance(warns, list)
    # deve haver missing (sexo) ou unknown (extra)
    assert any("missing" in w or "unknown" in w for w in warns)


# =========================
# _predict_df
# =========================
class _NoProbaModel:
    pass


class _DummyModel:
    def predict_proba(self, X):
        import numpy as np
        # prob fixo 0.5 para todo mundo
        return np.c_[1 - np.full((len(X),), 0.5), np.full((len(X),), 0.5)]


def test_predict_df_model_not_loaded():
    routes.MODEL = None
    with pytest.raises(RuntimeError, match="Modelo não carregado"):
        routes._predict_df(pd.DataFrame([{"a": 1}]), 0.5)


def test_predict_df_no_predict_proba():
    routes.MODEL = _NoProbaModel()
    with pytest.raises(RuntimeError, match="não suporta predict_proba"):
        routes._predict_df(pd.DataFrame([{"a": 1}]), 0.5)


def test_predict_df_nan_fill(monkeypatch):
    routes.MODEL = _DummyModel()
    routes.TXT_COLS[:] = ["cv"]
    proba, label = routes._predict_df(pd.DataFrame([{"cv": None}]), 0.6)
    assert len(proba) == 1 and len(label) == 1


# =========================
# /predict
# =========================
def test_predict_uses_threshold_default(client, monkeypatch):
    # evita dependência de modelo real
    monkeypatch.setattr(routes, "_predict_df", lambda df, thr: ([0.7], [1]))
    routes.META = {"schema_in": {"cat": [], "txt": []}}
    routes.CAT_COLS[:] = []
    routes.TXT_COLS[:] = []
    routes.NUM_COLS[:] = []

    body = {"meta": {"external_id": "1"}, "features": {}}
    r = client.post("/predict", json=body)
    assert r.status_code == 200
    assert r.json()["prediction"]["label"] in (0, 1)


def test_predict_model_error_returns_400(client, monkeypatch):
    def _boom(df, thr):
        raise RuntimeError("falha intencional")

    monkeypatch.setattr(routes, "_predict_df", _boom)
    routes.META = {"schema_in": {"cat": [], "txt": []}}
    routes.CAT_COLS[:] = []
    routes.TXT_COLS[:] = []
    routes.NUM_COLS[:] = []

    r = client.post("/predict", json={"features": {}})
    assert r.status_code == 400
    assert "falha intencional" in r.json()["detail"]


# =========================
# /rank-and-suggest
# =========================
def test_rank_empty_candidates_returns_400(client):
    r = client.post("/rank-and-suggest", json={"vaga": {}, "candidatos": []})
    assert r.status_code == 400
    assert "Nenhum candidato" in r.json()["detail"]


def test_rank_respects_topk_and_order(client, monkeypatch):
    routes.META = {"schema_in": {"cat": [], "txt": []}}
    routes.CAT_COLS[:] = []
    routes.TXT_COLS[:] = []

    # DataFrame com 3 linhas (3 candidatos)
    monkeypatch.setattr(routes, "_prepare_dataframe",
                        lambda items: (pd.DataFrame({"x": [1, 2, 3]}), []))
    # proba crescente pra checar ordenação desc
    monkeypatch.setattr(routes, "_predict_df",
                        lambda df, thr: ([0.1, 0.9, 0.5], [0, 1, 0]))

    payload = {
        "vaga": {},
        "candidatos": [
            {"meta": {"external_id": "a"}, "candidato": {}},
            {"meta": {"external_id": "b"}, "candidato": {}},
            {"meta": {"external_id": "c"}, "candidato": {}},
        ],
    }
    r = client.post("/rank-and-suggest?top_k=2", json=payload)
    assert r.status_code == 200
    ids = [x.get("external_id") or x.get("id") for x in r.json()["results"]]
    # esperado: b (0.9), c (0.5)
    assert ids == ["b", "c"]


def test_rank_include_questions_false(client, monkeypatch):
    routes.META = {"schema_in": {"cat": [], "txt": []}}
    routes.CAT_COLS[:] = []
    routes.TXT_COLS[:] = []

    monkeypatch.setattr(routes, "_prepare_dataframe",
                        lambda items: (pd.DataFrame({"x": [1]}), []))
    monkeypatch.setattr(routes, "_predict_df",
                        lambda df, thr: ([0.5], [0]))

    r = client.post(
        "/rank-and-suggest?include_questions=false",
        json={"vaga": {}, "candidatos": [{"meta": {}, "candidato": {}}]},
    )
    assert r.status_code == 200
    body = r.json()
    # não deve vir perguntas quando include_questions=false
    assert "questions" not in body or body["questions"] in (None, {})


def test_rank_include_questions_true_calls_suggest(client, monkeypatch):
    routes.META = {"schema_in": {"cat": [], "txt": []}}
    routes.CAT_COLS[:] = []
    routes.TXT_COLS[:] = []

    monkeypatch.setattr(routes, "_prepare_dataframe",
                        lambda items: (pd.DataFrame({"x": [1]}), []))
    monkeypatch.setattr(routes, "_predict_df",
                        lambda df, thr: ([0.5], [0]))

    called = {}

    def fake_suggest(req):
        called["ok"] = True
        return SuggestQuestionsResponse(common_questions=["Q"], per_candidate={"x": ["Q"]})

    monkeypatch.setattr(routes, "suggest_questions", fake_suggest)

    r = client.post(
        "/rank-and-suggest?include_questions=true",
        json={"vaga": {}, "candidatos": [{"meta": {}, "candidato": {}}]},
    )
    assert r.status_code == 200 and called.get("ok")
    assert r.json()["questions"]["common_questions"] == ["Q"]



def test_prepare_dataframe_requires_meta(monkeypatch):
    # força caminho de produção (sem bypass)
    monkeypatch.setenv("TESTING", "0")

    import app.routes as routes
    routes.META = None  # sem artefatos
    with pytest.raises(RuntimeError, match="Artefatos não carregados"):
        routes._prepare_dataframe([routes.PredictItem(meta=None, features={"a": 1})])


def test_prepare_dataframe_full_warnings(monkeypatch):
    import app.routes as routes
    # schema mínimo
    routes.META = {"schema_in": {"cat": ["sexo"], "txt": ["cv"]}}
    routes.CAT_COLS[:] = ["sexo"]
    routes.TXT_COLS[:] = ["cv"]

    # força warnings no próprio módulo
    routes.SHOW_WARNINGS = "full"  # <- em vez de setenv
    # (se quiser, mantenha também o env por simetria, mas o que vale é a var do módulo)
    monkeypatch.setenv("SHOW_WARNINGS", "full")

    monkeypatch.setenv("TESTING", "0")

    df, warns = routes._prepare_dataframe([
        routes.PredictItem(meta={"external_id": "x"}, features={"cv": "texto", "extra": 123})
    ])

    assert not df.empty
    assert isinstance(warns, list)
    assert any(("missing" in w) or ("unknown" in w) for w in warns)

    
