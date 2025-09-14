# app/main.py
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from . import routes  # app.routes

TAGS_METADATA = [
    {"name": "Health",  "description": "Sinalização e metadados do modelo/artefatos."},
    {"name": "Schema",  "description": "Esquema de entrada aceito pela API (somente colunas **cat/txt**)."},
    {"name": "Predict", "description": "Predição para 1 item."},
    {"name": "Ranking", "description": "Ranking de candidatos por vaga + (opcional) perguntas."},
]

def create_app() -> FastAPI:
    app = FastAPI(
        title="Triagem de Candidatos - API",
        version="2.0.0",
        description=(
            "API de scoring/triagem. **Atenção**: o modelo consome **apenas** as colunas cruas "
            "especificadas em `/schema` (categorias e textos). As features numéricas são "
            "geradas internamente no pipeline."
        ),
        openapi_tags=TAGS_METADATA,
    )

    @app.on_event("startup")
    def _load():
        routes.load_artifacts()

    app.include_router(routes.router, prefix="")

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse(url="/docs")

    return app

app = create_app()
