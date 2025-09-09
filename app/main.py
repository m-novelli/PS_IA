# api/main.py
from fastapi import FastAPI
from . import routes
from fastapi.responses import RedirectResponse

def create_app() -> FastAPI:
    app = FastAPI(
        title="Triagem de Candidatos - API",
        version="1.0.0",
        description="API para predição de avanço de candidatos para próximas fases."
    )

    @app.on_event("startup")
    def _load():
        routes.load_artifacts()

    app.include_router(routes.router, prefix="")

    @app.get("/", include_in_schema=False)
    def root():
        # redireciona para as docs
        return RedirectResponse(url="/docs")

    return app

app = create_app()
