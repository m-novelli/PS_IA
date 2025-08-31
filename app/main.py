from fastapi import FastAPI
from .routes import router as predict_router

def create_app() -> FastAPI:
    app = FastAPI(
        title="Triagem de Candidatos - API",
        version="1.0.0",
        description="API para predição de avanço de candidatos para próximas fases."
    )
    app.include_router(predict_router, prefix="")
    return app

app = create_app()
