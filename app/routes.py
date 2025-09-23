# app/routes.py
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from joblib import load
import pandas as pd
import json, os, hashlib, datetime as dt

# OpenAI: geração de perguntas
from app.suggest import suggest_questions, SuggestQuestionsRequest, SuggestQuestionsResponse

router = APIRouter()

# ================================
# Artefatos
# ================================
BASE_DIR = Path(__file__).resolve().parents[1]
PROD_DIR = BASE_DIR / "models" / "prod"

MODEL = None
META: dict | None = None
MODEL_PATH: Path | None = None
META_PATH: Path | None = None
LOADED_AT: str | None = None
ARTIFACT_SHA256: str | None = None

CAT_COLS: List[str] = []
TXT_COLS: List[str] = []
NUM_COLS: List[str] = []

THRESHOLD_DEFAULT: float = float(os.getenv("THRESHOLD_DEFAULT", "0.60"))
MAX_WARN_LIST: int = int(os.getenv("MAX_WARN_LIST", "30"))
SHOW_WARNINGS = os.getenv("SHOW_WARNINGS", "off").lower()

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def load_artifacts() -> None:
    global MODEL, META, MODEL_PATH, META_PATH, CAT_COLS, TXT_COLS, NUM_COLS
    global LOADED_AT, ARTIFACT_SHA256, THRESHOLD_DEFAULT

    MODEL_PATH = PROD_DIR / "model.joblib"
    META_PATH  = PROD_DIR / "meta.json"

    if not MODEL_PATH.exists() or not META_PATH.exists():
        if os.getenv("TESTING", "0") == "1":
            META = {}
            CAT_COLS, TXT_COLS, NUM_COLS = [], [], []
            LOADED_AT = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
            ARTIFACT_SHA256 = None
            return
        raise RuntimeError("Modelo ou metadados não encontrados.")

    MODEL = load(MODEL_PATH)
    META  = json.loads(META_PATH.read_text(encoding="utf-8"))

    schema_in = META.get("schema_in") or {}
    CAT_COLS[:] = list(schema_in.get("cat", []))
    TXT_COLS[:] = list(schema_in.get("txt", []))
    NUM_COLS[:] = list(schema_in.get("num", []))

    meta_thr = META.get("default_threshold")
    if isinstance(meta_thr, (int, float)):
        THRESHOLD_DEFAULT = float(meta_thr)

    LOADED_AT = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    ARTIFACT_SHA256 = _sha256(MODEL_PATH)

# ================================
# Schemas
# ================================
class PredictItem(BaseModel):
    meta: Optional[Dict[str, Any]] = None
    features: Dict[str, Any]

class PredictionOut(BaseModel):
    prob_next_phase: float
    label: int
    threshold: float

class PredictResponse(BaseModel):
    meta: Dict[str, Any] | None
    prediction: PredictionOut
    model: Dict[str, Any]

class CandidateIn(BaseModel):
    meta: Optional[Dict[str, Any]] = None
    candidato: Dict[str, Any]

class RankRequest(BaseModel):
    vaga: Dict[str, Any]
    candidatos: List[CandidateIn]

class RankItemOut(BaseModel):
    external_id: str | None = None
    prob_next_phase: float
    label: int

class RankAndSuggestResponse(BaseModel):
    job_meta: Dict[str, Any] | None = None
    top_k: int
    threshold_used: float
    results: List[RankItemOut]
    questions: Optional[SuggestQuestionsResponse] = None

# ================================
# Helpers
# ================================
def _coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    num_candidates = NUM_COLS or [
        c for c in df.columns if df[c].dropna().apply(lambda x: str(x).replace(".", "", 1).isdigit()).any()
    ]
    for col in num_candidates:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    text_candidates = CAT_COLS + TXT_COLS
    if not text_candidates:
        text_candidates = [c for c in df.columns if c not in num_candidates]

    for col in text_candidates:
        if col in df:
            df[col] = df[col].astype("string").fillna("")

    return df

def _prepare_dataframe(items: List[PredictItem]) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    if META is None and os.getenv("TESTING", "0") != "1":
        raise RuntimeError("Artefatos não carregados.")

    rows, warnings = [], []
    ordered_cols = list(NUM_COLS) + list(CAT_COLS) + list(TXT_COLS)
    expected = set(ordered_cols)

    for it in items:
        feats = it.features or {}
        row = {k: v for k, v in feats.items() if not expected or k in expected}
        rows.append(row)

    df = pd.DataFrame(rows)
    for c in ordered_cols:
        if c not in df.columns:
            df[c] = None
    if ordered_cols:
        df = df[ordered_cols]
    df = _coerce_dtypes(df)
    return df, warnings

def _predict_df(df: pd.DataFrame, threshold: float):
    if MODEL is None:
        if os.getenv("TESTING", "0") == "1":
            proba = [0.5] * len(df)
            label = [0] * len(df)
            return proba, label
        raise RuntimeError("Modelo não carregado.")

    if not hasattr(MODEL, "predict_proba"):
        raise RuntimeError("Modelo não suporta predict_proba.")

    if df.isna().any().any():
        df = df.fillna({c: "" for c in TXT_COLS}).fillna(0)

    proba = MODEL.predict_proba(df)[:, 1]
    label = (proba >= threshold).astype(int)
    return proba, label

def _merge_job_candidate(vaga: Dict[str, Any], cand: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(vaga)
    merged.update(cand)
    return merged

# ================================
# Endpoints
# ================================
@router.get("/health", tags=["Health"])
def health():
    return {
        "status": "ok",
        "artifact": MODEL_PATH.name if MODEL_PATH else None,
        "schema_in": {"num": NUM_COLS, "cat": CAT_COLS, "txt": TXT_COLS},
        "threshold_default": THRESHOLD_DEFAULT,
        "loaded_at": LOADED_AT,
        "artifact_sha256": ARTIFACT_SHA256,
    }

@router.get("/schema", tags=["Schema"])
def schema():
    return {
        "num": NUM_COLS,
        "cat": CAT_COLS,
        "txt": TXT_COLS,
        "threshold_default": THRESHOLD_DEFAULT,
    }

@router.post("/predict", response_model=PredictResponse, tags=["Predict"])
def predict_one(item: PredictItem, threshold: float = Query(None, ge=0.01, le=0.99)):
    try:
        thr = float(threshold) if threshold is not None else THRESHOLD_DEFAULT
        df, _ = _prepare_dataframe([item])
        proba, label = _predict_df(df, thr)
        return {
            "meta": item.meta or {},
            "prediction": {
                "prob_next_phase": float(proba[0]),
                "label": int(label[0]),
                "threshold": thr,
            },
            "model": {
                "name": META.get("type") if META else None,
                "artifact": MODEL_PATH.name if MODEL_PATH else None,
                "version": META.get("version") if META else None,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/predict-batch", tags=["Predict"])
def predict_batch(payload: Dict[str, List[PredictItem]], threshold: float = Query(None, ge=0.01, le=0.99)):
    items = payload.get("items", [])
    if not items:
        raise HTTPException(status_code=400, detail="Nenhum item enviado.")

    thr = float(threshold) if threshold is not None else THRESHOLD_DEFAULT
    df, _ = _prepare_dataframe(items)
    proba, label = _predict_df(df, thr)

    results = []
    for i, item in enumerate(items):
        results.append({
            "meta": item.meta or {},
            "prediction": {
                "prob_next_phase": float(proba[i]),
                "label": int(label[i]),
                "threshold": thr,
            }
        })
    return {"results": results}

@router.post("/rank-and-suggest", response_model=RankAndSuggestResponse, tags=["Ranking"])
def rank_and_suggest(
    req: RankRequest,
    threshold: float = Query(None, ge=0.01, le=0.99),
    top_k: int = Query(5, ge=1)
):
    try:
        if not req.candidatos:
            raise HTTPException(status_code=400, detail="Nenhum candidato enviado.")

        thr = float(threshold) if threshold is not None else THRESHOLD_DEFAULT

        # Combina vaga + candidatos
        items = []
        for c in req.candidatos:
            merged = _merge_job_candidate(req.vaga, c.candidato)
            items.append(PredictItem(meta=c.meta, features=merged))

        df, _ = _prepare_dataframe(items)
        proba, label = _predict_df(df, thr)

        scored = []
        for i, cand in enumerate(req.candidatos):
            scored.append(RankItemOut(
                external_id=cand.meta.get("external_id") if cand.meta else None,
                prob_next_phase=float(proba[i]),
                label=int(label[i]),
            ))

        scored.sort(key=lambda x: x.prob_next_phase, reverse=True)
        top = scored[:top_k]

        # Sugestão de perguntas (aqui já sabemos que há candidatos)
        q_request = SuggestQuestionsRequest(vaga=req.vaga, candidatos=[c.candidato for c in req.candidatos])
        q_response = suggest_questions(q_request)

        return RankAndSuggestResponse(
            job_meta=req.vaga,
            top_k=top_k,
            threshold_used=thr,
            results=top,
            questions=q_response,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))