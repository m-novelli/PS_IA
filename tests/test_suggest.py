import pytest
from pydantic import ValidationError

from app import suggest
from app.suggest import (
    SuggestQuestionsRequest,
    SuggestQuestionsResponse,
    suggest_questions,
)

def _as_response(obj) -> SuggestQuestionsResponse:
    if isinstance(obj, SuggestQuestionsResponse):
        return obj
    return SuggestQuestionsResponse(**obj)

def test_models_happy_path_instantiation():
    req = SuggestQuestionsRequest(
        vaga={"descricao": "Análises e relatórios de dados", "requisitos": "SQL, Python", "atividades": "Dashboards"},
        candidatos=[
            {"external_id": "cand1", "cv": "Experiência com SQL e Python"},
            {"external_id": "cand2", "cv": "ETL, análise exploratória"},
        ],
    )
    assert isinstance(req, SuggestQuestionsRequest)

def test_models_validation_errors():
    with pytest.raises(ValidationError):
        SuggestQuestionsRequest(vaga={"descricao": "algo"})

def test_suggest_offline_two_candidates(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("TESTING", "1")

    req = SuggestQuestionsRequest(
        vaga={"descricao": "Análises de dados", "requisitos": "SQL", "atividades": "Relatórios"},
        candidatos=[
            {"external_id": "cand1", "cv": "SQL e Python"},
            {"external_id": "cand2", "cv": "ETL"},
        ],
    )
    resp = _as_response(suggest_questions(req))
    assert isinstance(resp, SuggestQuestionsResponse)
    assert "cand1" in resp.per_candidate and "cand2" in resp.per_candidate

def test_suggest_handles_empty_texts(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("TESTING", "1")

    req = SuggestQuestionsRequest(
        vaga={"descricao": "", "requisitos": "", "atividades": ""},
        candidatos=[{"external_id": "c1", "cv": ""}],
    )
    resp = _as_response(suggest_questions(req))
    assert "c1" in resp.per_candidate
    assert isinstance(resp.per_candidate["c1"], list)

def test_suggest_single_candidate(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("TESTING", "1")

    req = SuggestQuestionsRequest(
        vaga={"descricao": "Data Viz", "requisitos": "PowerBI", "atividades": "Dashboards"},
        candidatos=[{"external_id": "only1", "cv": "PowerBI e DAX"}],
    )
    resp = _as_response(suggest_questions(req))
    assert list(resp.per_candidate.keys()) == ["only1"]

def test_cut_limits_string_length():
    text = "a" * 2000
    result = suggest._cut(text, 100)
    assert result.startswith("a")
    assert result.endswith("...") or len(result) <= 100

def test_extract_json_valid():
    content = "texto {\"perguntas\": [\"Q1\", \"Q2\"]} fim"
    result = suggest._extract_json(content)
    assert isinstance(result, str)
    assert "\"Q1\"" in result

def test_extract_json_invalid_returns_string():
    content = "isso não é json válido"
    result = suggest._extract_json(content)
    assert result == content

def test_suggest_questions_with_valid_json_offline(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("TESTING", "1")

    req = SuggestQuestionsRequest(
        vaga={"descricao": "Engenheiro de Dados"},
        candidatos=[{"external_id": "c42", "cv": "Spark, Databricks"}],
    )
    result = suggest_questions(req)
    assert "c42" in result.per_candidate

def test_suggest_questions_online_path_with_mock(monkeypatch):
    """Simula o caminho online sem usar a API real (forçando via chave fake)."""
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    monkeypatch.delenv("TESTING", raising=False)

    class FakeChoice:
        def __init__(self):
            self.message = type(
                "m",
                (),
                {
                    "content": '{"common_questions": ["Qual sua experiência com Spark?"], '
                               '"per_candidate": {"c99": ["Qual sua experiência com Spark?"]}}'
                },
            )
    class FakeCompletion:
        choices = [FakeChoice()]
    class FakeCompletions:
        def create(self, *a, **k): return FakeCompletion()
    class FakeChat:
        def __init__(self): self.completions = FakeCompletions()
    class FakeClient:
        def __init__(self, *a, **k): pass
        @property
        def chat(self): return FakeChat()

    monkeypatch.setattr(suggest, "OpenAI", lambda *a, **k: FakeClient())

    req = SuggestQuestionsRequest(
        vaga={"descricao": "Engenharia de Dados"},
        candidatos=[{"external_id": "c99", "cv": "Spark, Databricks"}],
    )
    resp = suggest_questions(req)

    assert isinstance(resp, SuggestQuestionsResponse)
    assert resp.common_questions == ["Qual sua experiência com Spark?"]
    assert resp.per_candidate["c99"] == ["Qual sua experiência com Spark?"]


# tests/test_routes_rank_happy_paths.py
import numpy as np

def _install_dummy_model(routes, prob=0.7):
    class Dummy:
        def predict_proba(self, X):
            return np.c_[1-np.full((len(X),), prob), np.full((len(X),), prob)]
    routes.MODEL = Dummy()

def test_rank_and_suggest_with_questions_and_topk(client, monkeypatch):
    import app.routes as routes
    monkeypatch.setenv("TESTING", "0")  # usa caminho produção do prepare/warnings

    # META/schema e modelo
    routes.META = {"schema_in": {"cat": [], "txt": []}}
    routes.CAT_COLS[:] = []
    routes.TXT_COLS[:] = []
    _install_dummy_model(routes, prob=0.8)

    # mock de suggest_questions
    def fake_suggest(req):
        return type("R", (), {"dict": lambda self=None: {"common_questions": ["Q1"], "per_candidate": {"c1": ["Q2"]}}})()
    monkeypatch.setattr(routes, "suggest_questions", fake_suggest)

    payload = {
        "vaga": {"descricao": "Eng Dados"},
        "candidatos": [
            {"meta": {"external_id": "c1"}, "candidato": {"cv_pt": "texto1"}},
            {"meta": {"external_id": "c2"}, "candidato": {"cv_pt": "texto2"}},
            {"meta": {"external_id": "c3"}, "candidato": {"cv_pt": "texto3"}},
        ],
    }
    r = client.post("/rank-and-suggest?top_k=2&include_questions=true", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert len(body["results"]) == 2                  # cortou top_k
    assert body["questions"]["common_questions"]      # incluiu perguntas

# tests/test_suggest_json_fallback.py
from app import suggest

def test_online_path_bad_json(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    monkeypatch.delenv("TESTING", raising=False)

    class FakeChoice: 
        def __init__(self): 
            self.message = type("m",(object,),{"content": "não-json"})()
    class FakeCompletion: choices = [FakeChoice()]
    class FakeCompletions: 
        def create(self,*a,**k): return FakeCompletion()
    class FakeChat: 
        def __init__(self): self.completions = FakeCompletions()
    class FakeClient:
        @property
        def chat(self): return FakeChat()

    monkeypatch.setattr(suggest, "OpenAI", lambda *a, **k: FakeClient())
    req = suggest.SuggestQuestionsRequest(vaga={"descricao":"x"}, candidatos=[{"external_id":"c1","cv":"y"}])
    resp = suggest.suggest_questions(req)
    assert "c1" in resp.per_candidate  # cai no fallback
