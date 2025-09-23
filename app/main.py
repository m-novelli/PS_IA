# app/main.py
import os
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from . import routes  # app.routes

# >>> logging estruturado
from .logging_config import setup_logging, logger
from .middlewares import RequestContextMiddleware

TAGS_METADATA = [
    {"name": "Health",  "description": "Sinalização e metadados do modelo/artefatos."},
    {"name": "Schema",  "description": "Esquema de entrada aceito pela API (somente colunas **cat/txt**)."},
    {"name": "Predict", "description": "Predição para 1 item."},
    {"name": "Ranking", "description": "Ranking de candidatos por vaga + (opcional) perguntas."},
]

# Metadados do modelo (compatível com MLflow Registry via env)
MODEL_URI    = os.getenv("MODEL_URI", "N/A")     # ex.: "models:/triagem-candidatos/Production"
MODEL_RUN_ID = os.getenv("MODEL_RUN_ID", "N/A")  # run_id do MLflow (se disponível)
MODEL_TAG    = os.getenv("MODEL_TAG", "Production")

def create_app() -> FastAPI:
    # configurar logging antes de criar o app
    setup_logging()

    app = FastAPI(
        title="Triagem de Candidatos - API",
        version="2.0.0",
        description=(
            "API de Match entre Candidatos para uma Vaga e Geração de Perguntas padronizadas para Entrevista"
        ),
        openapi_tags=TAGS_METADATA,
    )

    # middleware que injeta request_id, mede latência e loga http_request
    app.add_middleware(RequestContextMiddleware)

    @app.on_event("startup")
    def _load():
        routes.load_artifacts()
        # loga um evento de startup com identidade do modelo servido
        logger.info(
            "startup",
            model_uri=MODEL_URI,
            model_run_id=MODEL_RUN_ID,
            model_tag=MODEL_TAG,
            api_version=app.version,
        )

    app.include_router(routes.router, prefix="")

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse(url="/docs")

    return app

app = create_app()
