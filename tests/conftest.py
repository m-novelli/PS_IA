import pytest
from fastapi.testclient import TestClient
from app.main import create_app
from app import routes
from app.suggest import SuggestQuestionsResponse

# Força o modo de teste (seu routes.py checa TESTING=1)
@pytest.fixture(autouse=True)
def _force_testing(monkeypatch):
    monkeypatch.setenv("TESTING", "1")

# Mock do carregamento de artefatos chamado no startup do app
@pytest.fixture(autouse=True)
def _mock_load_artifacts(monkeypatch):
    def fake_load_artifacts():
        routes.MODEL = None
        routes.META = {}
        routes.MODEL_PATH = None
        routes.META_PATH = None
        routes.CAT_COLS[:] = []
        routes.TXT_COLS[:] = []
        routes.NUM_COLS[:] = []
        routes.LOADED_AT = "test"
        routes.ARTIFACT_SHA256 = None
        routes.THRESHOLD_DEFAULT = 0.60
    monkeypatch.setattr(routes, "load_artifacts", fake_load_artifacts)

# Mock da geração de perguntas da LLM
@pytest.fixture(autouse=True)
def _mock_suggest_questions(monkeypatch):
    def fake_suggest(req):
        return SuggestQuestionsResponse(
            common_questions=[
                "Qual sua experiência com SQL?",
                "Como resolve problemas de dados?",
            ],
            per_candidate={
                "cand1": ["Qual foi seu maior desafio em dados?"],
                "cand2": ["Como você lida com qualidade de dados?"],
            },
        )
    monkeypatch.setattr(routes, "suggest_questions", fake_suggest)

@pytest.fixture
def app():
    return create_app()

@pytest.fixture
def client(app):
    return TestClient(app)

@pytest.fixture
def feature_schema():
    return {"num": [], "cat": [], "txt": []}