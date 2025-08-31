from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from pathlib import Path
from joblib import load
import pandas as pd
import json
import os

router = APIRouter()

# ================================
# Carrega artefatos e configuração
# ================================
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "xgboost_pipeline.joblib"
META_PATH  = BASE_DIR / "models" / "xgboost_meta.json"
CFG_PATH   = BASE_DIR / "configs" / "triagem_features.json"

if not MODEL_PATH.exists():
    raise RuntimeError(f"Modelo não encontrado: {MODEL_PATH}")
if not META_PATH.exists():
    raise RuntimeError(f"Metadados não encontrados: {META_PATH}")
if not CFG_PATH.exists():
    raise RuntimeError(f"Config de features não encontrada: {CFG_PATH}")

MODEL = load(MODEL_PATH)
META  = json.loads(META_PATH.read_text(encoding="utf-8"))
CFG   = json.loads(CFG_PATH.read_text(encoding="utf-8"))

NUM_COLS: List[str] = CFG.get("NUM_CANDIDATES", [])
CAT_COLS: List[str] = CFG.get("CAT_CANDIDATES", [])
TXT_COLS: List[str] = CFG.get("TXT_CANDIDATES", [])

THRESHOLD_DEFAULT = float(os.getenv("THRESHOLD_DEFAULT", "0.60"))

# ================================
# Schemas
# ================================
class PredictItem(BaseModel):
    meta: Optional[Dict[str, Any]] = Field(default=None, description="IDs externos e contexto (ex.: codigo_vaga)")
    features: Dict[str, Any] = Field(description="Mapa coluna->valor conforme triagem_features.json")

class PredictBatch(BaseModel):
    items: List[PredictItem]

# ================================
# Helpers
# ================================
def _prepare_dataframe(items: List[PredictItem]):
    """Monta DataFrame com colunas esperadas pelo modelo e retorna também warnings por faltas."""
    rows, warnings = [], []
    expected = set(NUM_COLS + CAT_COLS + TXT_COLS)

    for it in items:
        feat = dict(it.features)  # cópia
        # Concatena colunas de texto se existirem
        texts = [str(feat.get(c, "")) for c in TXT_COLS if c in feat]
        if len(texts) > 0:
            feat["__texto"] = " ".join(texts)

        # Log de colunas faltantes (apenas informativo; imputação cuida do resto)
        missing = [c for c in expected if c not in feat]
        if missing:
            warnings.append({"external_id": (it.meta or {}).get("external_id"), "missing": missing})

        rows.append(feat)

    df = pd.DataFrame(rows)

    # Mantém apenas o que o modelo conhece (ordem: num -> cat -> __texto)
    keep_cols = [c for c in NUM_COLS if c in df.columns] + \
                [c for c in CAT_COLS if c in df.columns]
    if "__texto" in df.columns:
        keep_cols += ["__texto"]

    # Se nada foi informado, falha explicitamente
    if not keep_cols:
        raise ValueError("Nenhuma feature conhecida foi informada no payload.")

    # Garante presença das colunas esperadas (preenche com NA se não veio)
    for c in NUM_COLS + CAT_COLS:
        if c not in df.columns:
            df[c] = None
            keep_cols.append(c) if c in (NUM_COLS + CAT_COLS) and c not in keep_cols else None

    df = df[keep_cols]
    return df, warnings

def _predict_df(df: pd.DataFrame, threshold: float):
    proba = MODEL.predict_proba(df)[:, 1]
    label = (proba >= threshold).astype(int)
    return proba, label

# ================================
# Endpoints
# ================================
@router.get("/health")
def health():
    return {"status": "ok", "model": META.get("type", "unknown"), "version": MODEL_PATH.name}

@router.post("/predict")
def predict_one(
    item: PredictItem,
    threshold: float = Query(THRESHOLD_DEFAULT, ge=0.01, le=0.99, description="limiar para classificar como 1 (avança)")
):
    try:
        df, warns = _prepare_dataframe([item])
        proba, label = _predict_df(df, threshold)
        return {
            "meta": item.meta or {},
            "prediction": {
                "prob_next_phase": float(proba[0]),
                "label": int(label[0]),
                "threshold": threshold
            },
            "model": {
                "name": META.get("type", "xgboost"),
                "artifact": MODEL_PATH.name,
                "metrics_holdout": META.get("metrics_holdout", {})
            },
            "warnings": warns
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/predict-batch")
def predict_batch(
    payload: PredictBatch,
    threshold: float = Query(THRESHOLD_DEFAULT, ge=0.01, le=0.99)
):
    try:
        df, warns = _prepare_dataframe(payload.items)
        proba, label = _predict_df(df, threshold)

        results = []
        for i, it in enumerate(payload.items):
            results.append({
                "meta": it.meta or {},
                "prediction": {
                    "prob_next_phase": float(proba[i]),
                    "label": int(label[i]),
                    "threshold": threshold
                }
            })

        return {
            "results": results,
            "model": {
                "name": META.get("type", "xgboost"),
                "artifact": MODEL_PATH.name,
                "metrics_holdout": META.get("metrics_holdout", {})
            },
            "warnings": warns
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
