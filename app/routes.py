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
# Artefatos (carregados no startup)
# ================================
BASE_DIR = Path(__file__).resolve().parents[1]
PROD_DIR = BASE_DIR / "models" / "prod"

MODEL = None
META: dict | None = None
MODEL_PATH: Path | None = None
META_PATH: Path | None = None
LOADED_AT: str | None = None
ARTIFACT_SHA256: str | None = None

NUM_COLS: List[str] = []
CAT_COLS: List[str] = []
TXT_COLS: List[str] = []

THRESHOLD_DEFAULT: float = float(os.getenv("THRESHOLD_DEFAULT", "0.60"))
MAX_BATCH: int = int(os.getenv("MAX_BATCH", "5000"))
MAX_WARN_LIST: int = int(os.getenv("MAX_WARN_LIST", "30"))
SHOW_WARNINGS = os.getenv("SHOW_WARNINGS", "off").lower()  # "off" | "summary" | "full"

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def load_artifacts() -> None:
    """Chamada no startup do FastAPI (ver app/main.py)."""
    global MODEL, META, MODEL_PATH, META_PATH, NUM_COLS, CAT_COLS, TXT_COLS
    global LOADED_AT, ARTIFACT_SHA256, THRESHOLD_DEFAULT

    MODEL_PATH = PROD_DIR / "model.joblib"
    META_PATH  = PROD_DIR / "meta.json"

    if not MODEL_PATH.exists():
        raise RuntimeError(f"Modelo não encontrado: {MODEL_PATH}")
    if not META_PATH.exists():
        raise RuntimeError(f"Metadados não encontrados: {META_PATH}")

    MODEL = load(MODEL_PATH)
    META  = json.loads(META_PATH.read_text(encoding="utf-8"))

    feats = META.get("features_used", {})
    NUM_COLS[:] = list(feats.get("num", []))
    CAT_COLS[:] = list(feats.get("cat", []))
    TXT_COLS[:] = list(feats.get("txt", []))

    # threshold default via meta (se houver)
    meta_thr = META.get("default_threshold")
    if isinstance(meta_thr, (int, float)):
        THRESHOLD_DEFAULT = float(meta_thr)

    LOADED_AT = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    ARTIFACT_SHA256 = _sha256(MODEL_PATH)

# ================================
# Schemas de entrada/saída
# ================================
class PredictItem(BaseModel):
    meta: Optional[Dict[str, Any]] = Field(default=None, description="IDs externos (ex.: external_id)")
    features: Dict[str, Any] = Field(description="Mapa coluna->valor (colunas brutas do dataset)")

class PredictBatch(BaseModel):
    items: List[PredictItem]

class PredictionOut(BaseModel):
    prob_next_phase: float
    label: int
    threshold: float

class PredictResponse(BaseModel):
    meta: Dict[str, Any] | None
    prediction: PredictionOut
    model: Dict[str, Any]
    # warnings só será incluído quando SHOW_WARNINGS != "off"

class PredictBatchItemOut(BaseModel):
    external_id: Any | None = None
    meta: Dict[str, Any] | None = None
    prediction: PredictionOut

class PredictBatchResponse(BaseModel):
    results: List[PredictBatchItemOut]
    model: Dict[str, Any]
    # warnings só será incluído quando SHOW_WARNINGS != "off"

# Ranking por vaga/candidatos
class CandidateIn(BaseModel):
    meta: Optional[Dict[str, Any]] = None
    candidato: Dict[str, Any]  # ex.: {"idade": 30, "cv_pt": "...", ...}

class RankRequest(BaseModel):
    vaga: Dict[str, Any]           # ex.: {"perfil_vaga.principais_atividades": "...", ...}
    candidatos: List[CandidateIn]

class RankItemOut(BaseModel):
    external_id: Any | None = None
    prob_next_phase: float
    label: int

class RankResponse(BaseModel):
    job_meta: Dict[str, Any] | None = None
    top_k: int
    results: List[RankItemOut]
    # warnings só será incluído quando SHOW_WARNINGS != "off"

# ================================
# Helpers
# ================================
def _coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    # num -> numeric; cat -> string; txt -> string sem NaN
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in CAT_COLS:
        if c in df.columns:
            df[c] = df[c].astype("string")
    for c in TXT_COLS:
        if c in df.columns:
            df[c] = df[c].fillna("").astype("string")
    return df

def _prepare_dataframe(items: List[PredictItem]) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Monta DataFrame exatamente nas colunas esperadas pelo pipeline."""
    if META is None:
        raise RuntimeError("Artefatos não carregados. Verifique o startup do app.")

    rows, warnings = [], []
    expected = set(NUM_COLS + CAT_COLS + TXT_COLS)

    for it in items:
        feat = dict(it.features)
        external_id = (it.meta or {}).get("external_id")

        missing = [c for c in expected if c not in feat]
        unknown = [k for k in feat.keys() if k not in expected]

        # Controle de verbosidade dos avisos
        if SHOW_WARNINGS == "full":
            if missing:
                warnings.append({"external_id": external_id, "missing": missing[:MAX_WARN_LIST]})
            if unknown:
                warnings.append({"external_id": external_id, "unknown": unknown[:MAX_WARN_LIST]})
        elif SHOW_WARNINGS == "summary":
            entry = {"external_id": external_id}
            if missing:
                entry["missing_count"] = len(missing)
            if unknown:
                entry["unknown_count"] = len(unknown)
            if len(entry) > 1:
                warnings.append(entry)
        # "off": não adiciona nada

        rows.append(feat)

    df = pd.DataFrame(rows)

    # Garante presença das colunas esperadas
    for c in expected:
        if c not in df.columns:
            df[c] = None

    keep_cols = [c for c in NUM_COLS if c in df.columns] \
              + [c for c in CAT_COLS if c in df.columns] \
              + [c for c in TXT_COLS if c in df.columns]
    if not keep_cols:
        raise ValueError("Nenhuma feature conhecida foi informada no payload.")

    df = df[keep_cols]
    df = _coerce_dtypes(df)
    return df, warnings

def _predict_df(df: pd.DataFrame, threshold: float):
    if MODEL is None:
        raise RuntimeError("Modelo não carregado.")
    if not hasattr(MODEL, "predict_proba"):
        raise RuntimeError("Modelo não suporta predict_proba.")
    # Defesa final
    if df.isna().any().any():
        df = df.fillna({c: "" for c in TXT_COLS}).fillna(0)
    proba = MODEL.predict_proba(df)[:, 1]
    label = (proba >= threshold).astype(int)
    return proba, label

def _merge_job_candidate(vaga: Dict[str, Any], cand: Dict[str, Any]) -> Dict[str, Any]:
    """Mescla vaga + candidato num dict de features com nomes já usados no treino."""
    merged = dict(vaga)
    merged.update(cand)
    return merged

# ================================
# Endpoints
# ================================
@router.get("/health")
def health():
    return {
        "status": "ok",
        "model_type": META.get("type") if META else None,
        "artifact": MODEL_PATH.name if MODEL_PATH else None,
        "version": META.get("version") if META else None,
        "primary_metric": META.get("primary_metric") if META else None,
        "metrics_holdout": META.get("metrics_holdout", {}) if META else {},
        "threshold_default": THRESHOLD_DEFAULT,
        "features_used": META.get("features_used", {}) if META else {},
        "loaded_at": LOADED_AT,
        "artifact_sha256": ARTIFACT_SHA256,
    }

@router.get("/schema")
def schema():
    return {
        "num": NUM_COLS,
        "cat": CAT_COLS,
        "txt": TXT_COLS,
        "threshold_default": THRESHOLD_DEFAULT,
        "version": META.get("version") if META else None,
    }

@router.post("/predict", response_model=PredictResponse)
def predict_one(
    item: PredictItem,
    threshold: float = Query(None, ge=0.01, le=0.99, description="Limiar para classificar como 1 (avança)")
):
    try:
        thr = float(threshold) if threshold is not None else THRESHOLD_DEFAULT
        df, warns = _prepare_dataframe([item])
        proba, label = _predict_df(df, thr)

        resp = {
            "meta": item.meta or {},
            "prediction": {
                "prob_next_phase": float(proba[0]),
                "label": int(label[0]),
                "threshold": thr
            },
            "model": {
                "name": META.get("type") if META else None,
                "artifact": MODEL_PATH.name if MODEL_PATH else None,
                "version": META.get("version") if META else None
            }
        }
        if SHOW_WARNINGS != "off" and warns:
            resp["warnings"] = warns
        return resp
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))





@router.post("/rank-and-suggest")
def rank_and_suggest(
    payload: Dict[str, Any],
    top_k: int = Query(5, ge=1, le=1000, description="Quantidade de candidatos a retornar"),
    threshold: float = Query(None, ge=0.01, le=0.99, description="Limiar para classificar label 1"),
    include_questions: bool = Query(False, description="Se True, gera perguntas via OpenAI")  # default False
):
    """Única chamada que retorna ranking e (opcionalmente) perguntas."""
    try:
        if "vaga" not in payload or "candidatos" not in payload:
            raise HTTPException(status_code=400, detail="Payload deve conter 'vaga' e 'candidatos'.")

        vaga = payload["vaga"]
        candidatos = payload["candidatos"]
        thr = float(threshold) if threshold is not None else THRESHOLD_DEFAULT

        # ranking
        items = []
        for c in candidatos:
            features = _merge_job_candidate(vaga, (c.get("candidato") or {}))
            items.append(PredictItem(meta=c.get("meta"), features=features))
        df, warns = _prepare_dataframe(items)
        proba, label = _predict_df(df, thr)

        ranking = []
        for i, c in enumerate(candidatos):
            ranking.append({
                "external_id": ((c.get("meta") or {}).get("external_id")),
                "prob_next_phase": float(proba[i]),
                "label": int(label[i]),
            })
        ranking.sort(key=lambda x: x["prob_next_phase"], reverse=True)
        top = ranking[:top_k]

        questions = None
        if include_questions:
            # monta payload textual p/ LLM
            idx_by_id = {}
            for c in candidatos:
                eid = (c.get("meta") or {}).get("external_id")
                cv  = (c.get("candidato") or {}).get("cv_pt", "")
                if eid:
                    idx_by_id[eid] = cv

            cand_llm = [{"external_id": t["external_id"], "cv": idx_by_id.get(t["external_id"], "")} for t in top]
            req_llm = SuggestQuestionsRequest(
                vaga={
                    "descricao": vaga.get("perfil_vaga.principais_atividades", "") or vaga.get("descricao", ""),
                    "requisitos": vaga.get("perfil_vaga.competencia_tecnicas_e_comportamentais", "") or vaga.get("requisitos", ""),
                    "atividades": vaga.get("perfil_vaga.principais_atividades", "") or vaga.get("atividades", "")
                },
                candidatos=cand_llm
            )
            questions = suggest_questions(req_llm).dict()

        resp = {
            "job_meta": vaga,
            "top_k": top_k,
            "threshold_used": thr,
            "ranking": top,
            "questions": questions
        }
        if SHOW_WARNINGS != "off" and warns:
            resp["warnings"] = warns
        return resp

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
