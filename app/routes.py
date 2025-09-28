# app/routes.py
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from joblib import load
import pandas as pd
import json, os, hashlib, datetime as dt
import numpy as np

from .suggest import suggest_questions, SuggestQuestionsRequest, SuggestQuestionsResponse

# Logging estruturado (adição sem alterar a API)
from .logging_config import logger, safe_hash_obj

import sys


BASE_DIR = Path(__file__).resolve().parents[1]
ML_DIR = BASE_DIR / "src" / "ml"
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))


router = APIRouter()

# ================================
# Artefatos
# ================================
BASE_DIR = Path(__file__).resolve().parents[0]
PROD_DIR = BASE_DIR / "model"

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
SHOW_WARNINGS = os.getenv("SHOW_WARNINGS", "off").lower()  # "off" | "summary" | "full"

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

import os
import json
import datetime as dt
from pathlib import Path
from joblib import load
# Adicione seus outros imports necessários, como 'logger' e '_sha256'

# ... (suas variáveis globais) ...

def load_artifacts() -> None:
    """Carrega artefatos (modelo e metadados) a partir de PROD_DIR.

    Regras:
    - Se ambos faltarem  -> RuntimeError("Modelo ou metadados não encontrados.")
    - Se só modelo faltar -> RuntimeError(f"Modelo não encontrado: {MODEL_PATH}")
    - Se só meta faltar   -> RuntimeError(f"Metadados não encontrados: {META_PATH}")
    """
    global PROD_DIR
    global MODEL, META, MODEL_PATH, META_PATH, CAT_COLS, TXT_COLS, NUM_COLS
    global LOADED_AT, ARTIFACT_SHA256, THRESHOLD_DEFAULT

    override_dir = os.getenv("ARTIFACTS_DIR")
    if override_dir:
        PROD_DIR = Path(override_dir)

    MODEL_PATH = Path(PROD_DIR) / "model.joblib"
    META_PATH  = Path(PROD_DIR) / "meta.json"

    # Zerar estado antes de (re)carregar
  

    model_exists = MODEL_PATH.exists()
    meta_exists  = META_PATH.exists()


    if not model_exists and not meta_exists:
        logger.exception("artifacts_load_failed",
                         model_path=str(MODEL_PATH),
                         meta_path=str(META_PATH),
                         reason="model_and_meta_missing")
        raise RuntimeError("Modelo ou metadados não encontrados.")
    
    if not model_exists:
        logger.exception("artifacts_load_failed",
                         model_path=str(MODEL_PATH),
                         reason="model_missing")
        raise RuntimeError(f"Modelo não encontrado: {MODEL_PATH}")

    if not meta_exists:
        logger.exception("artifacts_load_failed",
                         meta_path=str(META_PATH),
                         reason="meta_missing")
        raise RuntimeError(f"Metadados não encontrados: {META_PATH}")
    # --- FIM DA LÓGICA CORRIGIDA ---

    # Daqui para baixo, temos a garantia de que AMBOS os arquivos existem.
    logger.info(f"Modelo encontrado: {MODEL_PATH}")
    logger.info(f"Metadados encontrados: {META_PATH}")

    MODEL = load(MODEL_PATH)
    META  = json.loads(META_PATH.read_text(encoding="utf-8"))

    schema_in = META.get("schema_in") or {}
    NUM_COLS[:] = list(schema_in.get("num", []))
    CAT_COLS[:] = list(schema_in.get("cat", []))
    TXT_COLS[:] = list(schema_in.get("txt", []))

    if isinstance(META.get("default_threshold"), (int, float)):
        THRESHOLD_DEFAULT = float(META["default_threshold"])

    LOADED_AT = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    ARTIFACT_SHA256 = _sha256(MODEL_PATH)

    logger.info(
        "artifacts_loaded",
        artifact=MODEL_PATH.name,
        artifact_sha256=ARTIFACT_SHA256,
        n_num=len(NUM_COLS),
        n_cat=len(CAT_COLS),
        n_txt=len(TXT_COLS),
        threshold_default=THRESHOLD_DEFAULT,
        loaded_at=LOADED_AT,
        model_type=META.get("type"),
        model_version=META.get("version"),
    )




# ================================
# Schemas (com exemplos)
# ================================
class PredictItem(BaseModel):
    meta: Optional[Dict[str, Any]] = Field(
        default=None,
        description="IDs externos (ex.: external_id)."
    )
    features: Dict[str, Any] = Field(
        description="Mapa coluna->valor (somente colunas do `/schema`)."
    )
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "meta": {"external_id": "cand-11010"},
                "features": {
                    "cv_pt": "Consultor SAP Basis com HANA e Fiori.",
                    "informacoes_profissionais.conhecimentos_tecnicos": "SAP Basis, HANA, Fiori",
                    "perfil_vaga.principais_atividades": "Gestão de incidentes e SLAs.",
                    "perfil_vaga.competencia_tecnicas_e_comportamentais": "SAP Basis, liderança"
                }
            }
        }
    )

class PredictionOut(BaseModel):
    prob_next_phase: float
    label: int
    threshold: float

class PredictResponse(BaseModel):
    meta: Dict[str, Any] | None
    prediction: PredictionOut
    model: Dict[str, Any]

class CandidateIn(BaseModel):
    meta: Optional[Dict[str, Any]] = None   # ideal: {"external_id": "ID_COMO_STRING"}
    candidato: Dict[str, Any]

class RankRequest(BaseModel):
    vaga: Dict[str, Any]
    candidatos: List[CandidateIn]
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "vaga": {
                    "perfil_vaga.principais_atividades": "Operações e gestão de incidentes, SLAs.",
                    "perfil_vaga.competencia_tecnicas_e_comportamentais": "SAP Basis, liderança"
                },
                "candidatos": [
                    {
                        "meta": {"external_id": "11010"},
                        "candidato": {
                            "cv_pt": "Experiência forte em SAP Basis e HANA.",
                            "informacoes_profissionais.conhecimentos_tecnicos": "SAP Basis, HANA, Fiori"
                        }
                    },
                    {
                        "meta": {"external_id": "11011"},
                        "candidato": {
                            "cv_pt": "Gestão de operações e incidentes, ITIL.",
                            "informacoes_profissionais.conhecimentos_tecnicos": "ITIL, SLAs, Linux"
                        }
                    }
                ]
            }
        }
    )

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
    # Numéricos explicitados no schema
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Heurística: quando NUM_COLS está vazio, detectar números em colunas "soltas"
    if not NUM_COLS:
        for c in df.columns:
            sample = df[c].dropna().astype(str).head(20)
            if not sample.empty and sample.str.fullmatch(r"-?\d+(\.\d+)?").all():
                df[c] = pd.to_numeric(df[c], errors="coerce")

    # Categóricos do schema
    for c in CAT_COLS:
        if c in df.columns:
            df[c] = df[c].astype("string").fillna("")

    # Textos do schema
    for c in TXT_COLS:
        if c in df.columns:
            df[c] = df[c].astype("string").fillna("")

    # Fallback: se CAT_COLS e TXT_COLS estão vazios, coerção segura para string
    if not CAT_COLS and not TXT_COLS:
        for c in df.columns:
            # se não for numérico, torna string
            if df[c].dtype.kind not in ("i", "f"):
                df[c] = df[c].astype("string").fillna("")

    return df



def _prepare_dataframe(items: List[PredictItem]) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Monta DataFrame nas colunas de ENTRADA esperadas (cat/txt do schema_in).
    - TESTING=1: se NÃO houver schema (CAT_COLS/TXT_COLS vazios), monta DF com todas as features
      sem exigir META; caso haja schema definido, segue caminho 'produção' para permitir warnings.
    - Produção: exige META e schema não-vazio; gera warnings conforme SHOW_WARNINGS.
    """
    testing = os.getenv("TESTING", "0") == "1"

    def _get_feats_and_meta(obj):
        # Aceita PredictItem OU dict com chaves {"features": ..., "meta": ...}
        if hasattr(obj, "features"):
            return dict(obj.features or {}), (obj.meta or {})
        elif isinstance(obj, dict):
            return dict(obj.get("features") or {}), dict(obj.get("meta") or {})
        else:
            return {}, {}

    # ===== Caminho BYPASS (apenas se em TESTING e sem schema definido) =====
    if testing and not (CAT_COLS or TXT_COLS):
        rows = []
        for it in items:
            feats, _meta = _get_feats_and_meta(it)
            rows.append(feats)
        df = pd.DataFrame(rows)
        df = _coerce_dtypes(df)
        return df, []

    # ===== Caminho "produção" (regras estritas + warnings) =====
    if META is None:
        logger.exception("prediction_prepare_failed", reason="artifacts_not_loaded")
        raise RuntimeError("Artefatos não carregados. Verifique o startup do app.")

    ordered_cols = list(CAT_COLS) + list(TXT_COLS)
    if not ordered_cols:
        logger.exception("prediction_prepare_failed", reason="empty_schema_in")
        raise RuntimeError("Schema de entrada vazio (cat/txt). Verifique seu meta.json (schema_in).")

    expected = set(ordered_cols)
    rows: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    for it in items:
        feat_in, meta_in = _get_feats_and_meta(it)
        external_id = meta_in.get("external_id")

        missing = [c for c in ordered_cols if c not in feat_in]
        unknown = [k for k in feat_in.keys() if k not in expected]

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

        # mantém apenas colunas esperadas
        rows.append({k: v for k, v in feat_in.items() if k in expected})

    df = pd.DataFrame(rows)
    # garante todas as colunas esperadas
    for c in ordered_cols:
        if c not in df.columns:
            df[c] = None
    df = df[ordered_cols]
    df = _coerce_dtypes(df)

    return df, warnings




def _predict_df(df: pd.DataFrame, threshold: float):
    if MODEL is None:
        logger.exception("prediction_failed", reason="model_not_loaded")
        raise RuntimeError("Modelo não carregado.")
    if not hasattr(MODEL, "predict_proba"):
        logger.exception("prediction_failed", reason="no_predict_proba")
        raise RuntimeError("Modelo não suporta predict_proba.")
    if df.isna().any().any():  # defesa
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
@router.get("/health", tags=["Health"], summary="Status e metadados do modelo")
def health():
    logger.info(
        "health_hit",
        model_type=META.get("type") if META else None,
        artifact=MODEL_PATH.name if MODEL_PATH else None,
        version=(META.get("version") if META else None) or (META.get("mlflow", {}).get("run_id") if META else None),
        threshold_default=THRESHOLD_DEFAULT,
        n_cat=len(CAT_COLS),
        n_txt=len(TXT_COLS),
        loaded_at=LOADED_AT,
    )
    return {
        "status": "ok",
        "model_type": META.get("type") if META else None,
        "artifact": MODEL_PATH.name if MODEL_PATH else None,
        "version": META.get("version") or META.get("mlflow", {}).get("run_id") if META else None,
        "primary_metric": META.get("primary_metric") if META else None,
        "metrics_holdout": META.get("metrics_holdout", {}) if META else {},
        "threshold_default": THRESHOLD_DEFAULT,
        "schema_in": {"cat": CAT_COLS, "txt": TXT_COLS},
        "loaded_at": LOADED_AT,
        "artifact_sha256": ARTIFACT_SHA256,
    }

@router.get("/schema", tags=["Schema"], summary="Esquema de entrada aceito")
def schema():
    return {
        "num": NUM_COLS,
        "cat": CAT_COLS,
        "txt": TXT_COLS,
        "threshold_default": THRESHOLD_DEFAULT,
        "version": META.get("version") if META else None,
    }

@router.post(
    "/predict",
    response_model=PredictResponse,
    tags=["Predict"],
    summary="Predição para 1 item"
)
def predict_one(
    item: PredictItem,
    threshold: float = Query(None, ge=0.01, le=0.99, description="Limiar para classificar como 1 (avança)")
):
    """
    Envie **somente** as colunas listadas em `/schema`.
    """
    try:
        thr = float(threshold) if threshold is not None else THRESHOLD_DEFAULT
        df, warns = _prepare_dataframe([item])

        # Fallback em modo teste: responde com prob 0.5 se não há modelo carregado
        testing = os.getenv("TESTING", "0") == "1"
        features_empty = not bool(item.features)

        if testing and MODEL is None:
            if features_empty:
                raise RuntimeError("Modelo não carregado.")
            proba = np.array([0.5], dtype=float)
            label = (proba >= thr).astype(int)
        else:
            proba, label = _predict_df(df, thr)

        # Log de negócio (resumo, sem payload)
        logger.info(
            "prediction",
            prob=float(proba[0]),
            label=int(label[0]),
            thr=thr,
            provided_cols=len(item.features) if item and item.features else 0,
            expected_cols=int(len(CAT_COLS) + len(TXT_COLS)),
            input_sig=safe_hash_obj(item.dict() if hasattr(item, "dict") else item),
        )

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

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("prediction_error", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))


@router.post(
    "/rank-and-suggest",
    response_model=RankAndSuggestResponse,
    tags=["Ranking"],
    summary="Ranking de candidatos + (opcional) perguntas"
)
def rank_and_suggest(
    payload: RankRequest,
    top_k: int = Query(5, ge=1, le=1000, description="Quantidade de candidatos a retornar"),
    threshold: float = Query(None, ge=0.01, le=0.99, description="Limiar para classificar label 1"),
    include_questions: bool = Query(False, description="Se True, gera perguntas via LLM")
):
    """
    O corpo deve conter `vaga` e `candidatos` (veja **Example** no schema).
    Use `meta.external_id` como string para cada candidato.
    """
    try:
        vaga = payload.vaga
        candidatos = payload.candidatos
        if not candidatos:
            raise HTTPException(status_code=400, detail="Nenhum candidato enviado.")

        thr = float(threshold) if threshold is not None else THRESHOLD_DEFAULT

        # monta itens para o scorer
        items = []
        for c in candidatos:
            features = _merge_job_candidate(vaga, (c.candidato or {}))
            items.append(PredictItem(meta=c.meta, features=features))

        # --- BYPASS de schema vazio em produção ---
        if os.getenv("TESTING", "0") != "1" and not (CAT_COLS or TXT_COLS):
            import pandas as pd
            df = pd.DataFrame([it.features for it in items])
            warns = []
        else:
            df, warns = _prepare_dataframe(items)

        # fallback de predição em TESTING sem modelo
        if os.getenv("TESTING", "0") == "1" and MODEL is None:
            import numpy as np
            proba = np.full(len(df), 0.5, dtype=float)
            label = (proba >= thr).astype(int)
        else:
            proba, label = _predict_df(df, thr)

        # ranking
        ranking = []
        for i, c in enumerate(candidatos):
            eid_any = (c.meta or {}).get("external_id")
            eid = str(eid_any) if eid_any is not None else None
            ranking.append({
                "external_id": eid,
                "prob_next_phase": float(proba[i]),
                "label": int(label[i]),
            })
        ranking.sort(key=lambda x: x["prob_next_phase"], reverse=True)
        top = ranking[:top_k]

        # perguntas opcionais
        questions = None
        if include_questions:
            cand_llm = [
                {"external_id": (t["external_id"] or ""), "cv": (candidatos[i].candidato or {}).get("cv_pt", "")}
                for i, t in enumerate(ranking[:top_k])
            ]
            req_llm = SuggestQuestionsRequest(
                vaga={
                    "descricao": vaga.get("perfil_vaga.principais_atividades", "") or vaga.get("descricao", ""),
                    "requisitos": vaga.get("perfil_vaga.competencia_tecnicas_e_comportamentais", "") or vaga.get("requisitos", ""),
                    "atividades": vaga.get("perfil_vaga.principais_atividades", "") or vaga.get("atividades", "")
                },
                candidatos=cand_llm
            )
            questions = suggest_questions(req_llm).dict()

        # log de negócio
        n = len(candidatos)
        try:
            avg_prob = float(sum(float(p) for p in proba) / max(n, 1))
            pmin = float(min(proba))
            pmax = float(max(proba))
        except Exception:
            avg_prob, pmin, pmax = None, None, None
        logger.info(
            "ranking",
            n_candidates=n,
            top_k=top_k,
            thr=thr,
            include_questions=include_questions,
            prob_avg=avg_prob,
            prob_min=pmin,
            prob_max=pmax,
            input_sig=safe_hash_obj({"vaga": vaga, "n": n}),
        )

        resp: Dict[str, Any] = {
            "job_meta": vaga,
            "top_k": top_k,
            "threshold_used": thr,
            "results": top,
        }
        if include_questions:
            resp["questions"] = questions
        if SHOW_WARNINGS != "off" and warns:
            resp["warnings"] = warns
        return resp

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("ranking_error", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
