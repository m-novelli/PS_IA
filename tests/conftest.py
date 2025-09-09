import pytest
from api.main import create_app
from api import routes
from fastapi.testclient import TestClient

@pytest.fixture(scope="session", autouse=True)
def _load_artifacts_once():
    # carrega modelo/meta para popular NUM/CAT/TXT antes dos testes
    routes.load_artifacts()

@pytest.fixture(scope="session")
def app():
    return create_app()

@pytest.fixture(scope="session")
def client(app):
    return TestClient(app)

@pytest.fixture(scope="session")
def feature_schema():
    return {"num": routes.NUM_COLS, "cat": routes.CAT_COLS, "txt": routes.TXT_COLS}
